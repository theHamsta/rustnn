use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use cudarc::driver::{CudaContext, result, sys};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE},
        Graphics::{Direct3D12::*, Dxgi::Common::*},
    },
    core::PCWSTR,
};

use super::{CudaExternal, InteropDevice, SharedAllocation, error};
use crate::error::Result;

type Api = wgpu::hal::api::Dx12;

pub(super) fn cuda_device(device: &wgpu::Device) -> Result<u32> {
    let hal = unsafe { device.as_hal::<Api>() }
        .ok_or_else(|| error("Windows wgpu/CUDA interop requires D3D12"))?;
    let raw = hal.raw_device();
    if unsafe { raw.GetNodeCount() } != 1 {
        return Err(error(
            "CUDA interop does not support linked-node D3D12 adapters",
        ));
    }
    let luid = unsafe { raw.GetAdapterLuid() };
    let mut expected = [0u8; 8];
    expected[..4].copy_from_slice(&luid.LowPart.to_ne_bytes());
    expected[4..].copy_from_slice(&luid.HighPart.to_ne_bytes());
    result::init().map_err(error)?;
    for index in 0..result::device::get_count().map_err(error)? {
        let cuda = result::device::get(index).map_err(error)?;
        let mut actual = [0i8; 8];
        let mut node_mask = 0;
        unsafe { sys::cuDeviceGetLuid(actual.as_mut_ptr(), &mut node_mask, cuda).result() }
            .map_err(error)?;
        if actual.map(|b| b as u8) == expected && node_mask == 1 {
            if super::super::TrtxContext::list_devices_for_interop().contains(&(index as u32)) {
                return Ok(index as u32);
            }
            return Err(error(
                "matching CUDA device does not meet TensorRT-RTX requirements",
            ));
        }
    }
    Err(error("no CUDA device matches the D3D12 adapter LUID"))
}

struct SharedHandle(HANDLE);
impl Drop for SharedHandle {
    fn drop(&mut self) {
        let _ = unsafe { CloseHandle(self.0) };
    }
}

#[derive(Debug)]
pub(super) struct Allocation {
    _device: wgpu::Device,
    queue: ID3D12CommandQueue,
    resource: ID3D12Resource,
    fences: [ID3D12Fence; 2],
    idle_fence: ID3D12Fence,
    idle_value: AtomicU64,
}

impl Allocation {
    pub(super) fn new(
        device: &InteropDevice,
        ctx: &Arc<CudaContext>,
        size: u64,
    ) -> Result<(Self, CudaExternal)> {
        ctx.bind_to_thread().map_err(error)?;
        let hal =
            unsafe { device.device.as_hal::<Api>() }.ok_or_else(|| error("expected D3D12"))?;
        let raw = hal.raw_device();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            raw.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
        }
        .map_err(error)?;
        let resource = resource.ok_or_else(|| error("D3D12 returned no shared resource"))?;
        let fences = [
            unsafe { raw.CreateFence(0, D3D12_FENCE_FLAG_SHARED) }.map_err(error)?,
            unsafe { raw.CreateFence(0, D3D12_FENCE_FLAG_SHARED) }.map_err(error)?,
        ];
        let allocation = Self {
            _device: device.device.clone(),
            queue: hal.raw_queue().clone(),
            resource,
            fences,
            idle_fence: unsafe { raw.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.map_err(error)?,
            idle_value: AtomicU64::new(0),
        };
        let mut cuda = CudaExternal::empty(ctx);
        // GENERIC_ALL, as required for shared D3D12 objects. CUDA does not consume NT handles.
        let memory_handle = SharedHandle(
            unsafe {
                raw.CreateSharedHandle(&allocation.resource, None, 0x1000_0000, PCWSTR::null())
            }
            .map_err(error)?,
        );
        let mut memory = CudaExternal::memory_descriptor(
            sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
        );
        memory.handle.win32.handle = memory_handle.0.0;
        memory.size = unsafe { raw.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes;
        memory.flags = sys::CUDA_EXTERNAL_MEMORY_DEDICATED;
        unsafe { sys::cuImportExternalMemory(&mut cuda.memory, &memory).result() }
            .map_err(error)?;
        cuda.map(size)?;
        for index in 0..2 {
            let handle = SharedHandle(
                unsafe {
                    raw.CreateSharedHandle(
                        &allocation.fences[index],
                        None,
                        0x1000_0000,
                        PCWSTR::null(),
                    )
                }
                .map_err(error)?,
            );
            let mut semaphore = CudaExternal::semaphore_descriptor(
                sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
            );
            semaphore.handle.win32.handle = handle.0.0;
            unsafe {
                sys::cuImportExternalSemaphore(&mut cuda.semaphores[index], &semaphore).result()
            }
            .map_err(error)?;
        }
        // wgpu's HAL wrapping contract requires initialized memory. Complete this initial clear
        // before wrapping; all subsequent transfers use shared GPU fence waits/signals.
        unsafe { sys::cuMemsetD8_v2(cuda.ptr, 0, size as usize).result() }.map_err(error)?;
        ctx.synchronize().map_err(error)?;
        Ok((allocation, cuda))
    }

    pub(super) fn wrap(
        &self,
        device: &wgpu::Device,
        size: u64,
        owner: Arc<SharedAllocation>,
    ) -> wgpu::Buffer {
        // Unlike Vulkan, the D3D12 HAL buffer retains a COM reference to the committed resource.
        // That reference keeps its memory alive even after the CUDA imports have been released.
        drop(owner);
        unsafe {
            let buffer = wgpu::hal::dx12::Device::buffer_from_raw(self.resource.clone(), size);
            device.create_buffer_from_hal::<Api>(
                buffer,
                &wgpu::BufferDescriptor {
                    label: Some("rustnn shared D3D12/CUDA tensor"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )
        }
    }

    pub(super) fn acquire(&self, value: u64) -> Result<()> {
        unsafe { self.queue.Wait(&self.fences[1], value) }.map_err(error)
    }
    pub(super) fn release(&self, value: u64) -> Result<()> {
        unsafe { self.queue.Signal(&self.fences[0], value) }.map_err(error)
    }
    pub(super) fn idle(&self) {
        let value = self.idle_value.fetch_add(1, Ordering::Relaxed) + 1;
        // A null event makes SetEventOnCompletion block until the queue reaches this fence.
        unsafe {
            if self.queue.Signal(&self.idle_fence, value).is_ok() {
                let _ = self
                    .idle_fence
                    .SetEventOnCompletion(value, HANDLE::default());
            }
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.idle();
    }
}
