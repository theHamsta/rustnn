//! Shared native wgpu buffers and CUDA synchronization.
//!
//! Vulkan uses opaque FDs on Linux and NT handles on Windows; D3D12 uses shared NT handles.
//! Vulkan devices must enable the platform external memory and semaphore extensions;
//! [`create_vulkan_device`] creates a wgpu device with those extensions enabled.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys};

use crate::error::{Error, Result};

#[cfg(any(target_os = "linux", windows))]
mod vulkan;
#[cfg(any(target_os = "linux", windows))]
pub use vulkan::create_vulkan_device;
#[cfg(windows)]
mod dx12;

pub(super) fn error(message: impl std::fmt::Display) -> Error {
    Error::GpuInterop(message.to_string())
}

#[derive(Debug, Clone)]
pub(crate) struct InteropDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl InteropDevice {
    pub(crate) fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self> {
        let result = Self { device, queue };
        result.cuda_device()?;
        Ok(result)
    }

    pub(crate) fn cuda_device(&self) -> Result<u32> {
        // SAFETY: probe the driver library before cudarc's lazy loader can panic.
        if !unsafe { sys::is_culib_present() } {
            return Err(error("CUDA driver library is unavailable"));
        }
        match self.device.adapter_info().backend {
            #[cfg(any(target_os = "linux", windows))]
            wgpu::Backend::Vulkan => vulkan::cuda_device(&self.device),
            #[cfg(windows)]
            wgpu::Backend::Dx12 => dx12::cuda_device(&self.device),
            _ => Err(error(
                "wgpu/CUDA interop requires Vulkan on Linux/Windows or D3D12 on Windows",
            )),
        }
    }

    pub(crate) fn allocate(&self, ctx: &Arc<CudaContext>, bytes: usize) -> Result<SharedTensor> {
        // All graphics-visible bytes, including padding, are initialized before wrapping in wgpu.
        let size = (bytes.max(1) as u64)
            .checked_add(3)
            .ok_or_else(|| error("buffer size overflow"))?
            & !3;
        if size > self.device.limits().max_buffer_size {
            return Err(error("shared tensor exceeds the wgpu buffer size limit"));
        }
        #[cfg(any(target_os = "linux", windows))]
        let (native, cuda) = match self.device.adapter_info().backend {
            wgpu::Backend::Vulkan => {
                let (allocation, cuda) = vulkan::Allocation::new(self, ctx, size)?;
                (NativeAllocation::Vulkan(Box::new(allocation)), cuda)
            }
            #[cfg(windows)]
            wgpu::Backend::Dx12 => {
                let (allocation, cuda) = dx12::Allocation::new(self, ctx, size)?;
                (NativeAllocation::Dx12(Box::new(allocation)), cuda)
            }
            _ => return Err(error("unsupported graphics backend")),
        };
        #[cfg(not(any(target_os = "linux", windows)))]
        return Err(error("unsupported graphics platform"));
        #[cfg(any(target_os = "linux", windows))]
        {
            let allocation = Arc::new(SharedAllocation { cuda, native });
            let buffer = allocation
                .native
                .wrap(&self.device, size, Arc::clone(&allocation));
            Ok(SharedTensor {
                allocation,
                buffer,
                size: bytes,
                graphics_value: 0,
                cuda_value: 0,
                device: self.clone(),
            })
        }
    }
}

#[cfg(any(target_os = "linux", windows))]
#[derive(Debug)]
enum NativeAllocation {
    Vulkan(Box<vulkan::Allocation>),
    #[cfg(windows)]
    Dx12(Box<dx12::Allocation>),
}

#[cfg(any(target_os = "linux", windows))]
impl NativeAllocation {
    fn wrap(&self, device: &wgpu::Device, size: u64, owner: Arc<SharedAllocation>) -> wgpu::Buffer {
        match self {
            Self::Vulkan(allocation) => allocation.wrap(device, size, owner),
            #[cfg(windows)]
            Self::Dx12(allocation) => allocation.wrap(device, size, owner),
        }
    }
    fn idle(&self) {
        match self {
            Self::Vulkan(allocation) => allocation.idle(),
            #[cfg(windows)]
            Self::Dx12(allocation) => allocation.idle(),
        }
    }
    fn acquire(&self, value: u64) -> Result<()> {
        match self {
            Self::Vulkan(allocation) => allocation.acquire(value),
            #[cfg(windows)]
            Self::Dx12(allocation) => allocation.acquire(value),
        }
    }
    fn release(&self, value: u64) -> Result<()> {
        match self {
            Self::Vulkan(allocation) => allocation.release(value),
            #[cfg(windows)]
            Self::Dx12(allocation) => allocation.release(value),
        }
    }
}

// Keep the feature compilable on targets without native CUDA graphics interop.
#[cfg(not(any(target_os = "linux", windows)))]
type NativeAllocation = UnsupportedAllocation;
#[cfg(not(any(target_os = "linux", windows)))]
#[derive(Debug)]
struct UnsupportedAllocation;
#[cfg(not(any(target_os = "linux", windows)))]
impl UnsupportedAllocation {
    fn idle(&self) {}
    fn acquire(&self, _: u64) -> Result<()> {
        Err(error("unsupported platform"))
    }
    fn release(&self, _: u64) -> Result<()> {
        Err(error("unsupported platform"))
    }
}

#[derive(Debug)]
struct SharedAllocation {
    // CUDA mappings/imports must be destroyed before their graphics backing resources.
    cuda: CudaExternal,
    native: NativeAllocation,
}

impl Drop for SharedAllocation {
    fn drop(&mut self) {
        // A retained wgpu buffer can outlive MLContext. Its callback retains this allocation.
        let _ = self.cuda.ctx.bind_to_thread();
        let _ = self.cuda.ctx.synchronize();
        self.native.idle();
    }
}

#[derive(Debug)]
pub(crate) struct SharedTensor {
    allocation: Arc<SharedAllocation>,
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) size: usize,
    graphics_value: u64,
    cuda_value: u64,
    device: InteropDevice,
}

impl SharedTensor {
    pub(crate) fn ptr(&self) -> sys::CUdeviceptr {
        self.allocation.cuda.ptr
    }

    pub(crate) fn export(&mut self, stream: &CudaStream) -> Result<()> {
        self.cuda_value = self
            .cuda_value
            .checked_add(1)
            .ok_or_else(|| error("CUDA timeline exhausted"))?;
        self.allocation.cuda.signal(stream, self.cuda_value)?;
        // This submission waits on CUDA before acquiring the native resource for graphics.
        self.allocation.native.acquire(self.cuda_value)
    }

    pub(crate) fn reclaim(&mut self, stream: &CudaStream) -> Result<()> {
        self.graphics_value = self
            .graphics_value
            .checked_add(1)
            .ok_or_else(|| error("graphics timeline exhausted"))?;
        // Flush pending wgpu uploads, and give D3D12 a known COMMON resource state.
        let mut encoder = self
            .device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.transition_resources(
            std::iter::once(wgpu::BufferTransition {
                buffer: &self.buffer,
                state: wgpu::BufferUses::empty(),
            }),
            std::iter::empty(),
        );
        self.device.queue.submit([encoder.finish()]);
        self.allocation.native.release(self.graphics_value)?;
        self.allocation.cuda.wait(stream, self.graphics_value)
    }
}

#[derive(Debug)]
struct CudaExternal {
    ptr: sys::CUdeviceptr,
    memory: sys::CUexternalMemory,
    // [graphics -> CUDA, CUDA -> graphics]
    semaphores: [sys::CUexternalSemaphore; 2],
    ctx: Arc<CudaContext>,
}

// CUDA handles are used only with their owning context bound to the calling thread.
unsafe impl Send for CudaExternal {}
unsafe impl Sync for CudaExternal {}

impl CudaExternal {
    fn empty(ctx: &Arc<CudaContext>) -> Self {
        Self {
            ptr: 0,
            memory: std::ptr::null_mut(),
            semaphores: [std::ptr::null_mut(); 2],
            ctx: ctx.clone(),
        }
    }

    fn memory_descriptor(
        kind: sys::CUexternalMemoryHandleType,
    ) -> sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: kind,
            handle: unsafe { std::mem::zeroed() },
            size: 0,
            flags: 0,
            reserved: [0; 16],
        }
    }

    fn semaphore_descriptor(
        kind: sys::CUexternalSemaphoreHandleType,
    ) -> sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
        sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
            type_: kind,
            handle: unsafe { std::mem::zeroed() },
            flags: 0,
            reserved: [0; 16],
        }
    }

    fn map(&mut self, size: u64) -> Result<()> {
        let desc = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            size,
            ..unsafe { std::mem::zeroed() }
        };
        // SAFETY: memory is an imported allocation of at least `size` bytes.
        unsafe { sys::cuExternalMemoryGetMappedBuffer(&mut self.ptr, self.memory, &desc).result() }
            .map_err(error)
    }

    fn signal(&self, stream: &CudaStream, value: u64) -> Result<()> {
        self.ctx.bind_to_thread().map_err(error)?;
        let mut params: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = unsafe { std::mem::zeroed() };
        params.params.fence.value = value;
        // SAFETY: semaphore is imported from this allocation's graphics timeline.
        unsafe {
            sys::cuSignalExternalSemaphoresAsync(
                &self.semaphores[1],
                &params,
                1,
                stream.cu_stream(),
            )
            .result()
        }
        .map_err(error)
    }

    fn wait(&self, stream: &CudaStream, value: u64) -> Result<()> {
        self.ctx.bind_to_thread().map_err(error)?;
        let mut params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = unsafe { std::mem::zeroed() };
        params.params.fence.value = value;
        // SAFETY: graphics submits the corresponding signal before this wait.
        unsafe {
            sys::cuWaitExternalSemaphoresAsync(&self.semaphores[0], &params, 1, stream.cu_stream())
                .result()
        }
        .map_err(error)
    }
}

impl Drop for CudaExternal {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        let _ = self.ctx.synchronize();
        // SAFETY: this object uniquely owns each non-null imported handle and mapped pointer.
        unsafe {
            if self.ptr != 0 {
                let _ = sys::cuMemFree_v2(self.ptr);
            }
            for semaphore in self.semaphores {
                if !semaphore.is_null() {
                    let _ = sys::cuDestroyExternalSemaphore(semaphore);
                }
            }
            if !self.memory.is_null() {
                let _ = sys::cuDestroyExternalMemory(self.memory);
            }
        }
    }
}

#[cfg(all(test, any(target_os = "linux", windows)))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an NVIDIA Vulkan/CUDA GPU"]
    fn vulkan_cuda_shared_buffer() {
        let _ = pretty_env_logger::try_init();
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
            .into_iter()
            .find(|a| a.get_info().vendor == 0x10de)
            .expect("NVIDIA Vulkan adapter required");
        let (device, queue) = create_vulkan_device(&adapter, &Default::default()).unwrap();
        let interop = InteropDevice::new(device.clone(), queue.clone()).unwrap();
        let cuda = CudaContext::new(interop.cuda_device().unwrap() as usize).unwrap();
        let stream = cuda.new_stream().unwrap();
        let mut shared = interop.allocate(&cuda, 1024).unwrap();
        for value in 1u8..=16 {
            // CUDA writes -> Vulkan copy -> CPU readback, without a host wait at handoff.
            unsafe {
                cudarc::driver::result::memset_d8_async(
                    shared.ptr(),
                    value,
                    1024,
                    stream.cu_stream(),
                )
            }
            .unwrap();
            shared.export(&stream).unwrap();
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 1024,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&shared.buffer, 0, &staging, 0, 1024);
            queue.submit([encoder.finish()]);
            // Vulkan writes -> CUDA readback. reclaim must flush pending queue.write_buffer.
            queue.write_buffer(&shared.buffer, 0, &[value + 1; 1024]);
            shared.reclaim(&stream).unwrap();
            let mut actual = [0u8; 1024];
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(
                    &mut actual,
                    shared.ptr(),
                    stream.cu_stream(),
                )
            }
            .unwrap();
            stream.synchronize().unwrap();
            assert_eq!(actual, [value + 1; 1024]);
            let (sender, receiver) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap()
                });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            receiver.recv().unwrap().unwrap();
            assert_eq!(
                &*staging.slice(..).get_mapped_range().unwrap(),
                &[value; 1024]
            );
        }
        // wgpu's callback must keep the native allocation alive after the ML view is dropped.
        shared.export(&stream).unwrap();
        let buffer = shared.buffer.clone();
        drop(shared);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&buffer, 0, None);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
}
