#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use ash::{vk, vk::Handle};
use cudarc::driver::{CudaContext, result, sys};

use super::{CudaExternal, InteropDevice, SharedAllocation, error};
use crate::error::Result;

type Api = wgpu::hal::api::Vulkan;

fn required_extensions() -> [&'static std::ffi::CStr; 2] {
    #[cfg(target_os = "linux")]
    {
        [
            ash::khr::external_memory_fd::NAME,
            ash::khr::external_semaphore_fd::NAME,
        ]
    }
    #[cfg(windows)]
    {
        [
            ash::khr::external_memory_win32::NAME,
            ash::khr::external_semaphore_win32::NAME,
        ]
    }
}

fn memory_handle_type() -> vk::ExternalMemoryHandleTypeFlags {
    #[cfg(target_os = "linux")]
    {
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD
    }
    #[cfg(windows)]
    {
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32
    }
}

fn semaphore_handle_type() -> vk::ExternalSemaphoreHandleTypeFlags {
    #[cfg(target_os = "linux")]
    {
        vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD
    }
    #[cfg(windows)]
    {
        vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32
    }
}

/// Create a Vulkan wgpu device with CUDA-exportable memory and semaphore support.
///
/// The adapter must support timeline semaphores and the platform's external memory and
/// semaphore extensions (`_fd` on Linux, `_win32` on Windows). This helper requests
/// `VULKAN_EXTERNAL_MEMORY_FD` or `VULKAN_EXTERNAL_MEMORY_WIN32` and also enables the
/// semaphore extension ordinary wgpu device creation does not request. Handles are
/// opaque FDs or NT handles, never DMA-BUFs or KMT handles.
pub fn create_vulkan_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Result<(wgpu::Device, wgpu::Queue)> {
    // SAFETY: only inspect the adapter; no raw resources escape this function.
    let hal =
        unsafe { adapter.as_hal::<Api>() }.ok_or_else(|| error("expected a Vulkan adapter"))?;
    let instance = hal.shared_instance().raw_instance();
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(hal.raw_physical_device()) }
            .map_err(error)?;
    for required in required_extensions() {
        if !extensions
            .iter()
            .any(|ext| unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) } == required)
        {
            return Err(error(format!(
                "adapter lacks {}",
                required.to_string_lossy()
            )));
        }
    }
    let mut descriptor = descriptor.clone();
    #[cfg(target_os = "linux")]
    {
        descriptor.required_features |= wgpu::Features::VULKAN_EXTERNAL_MEMORY_FD;
    }
    #[cfg(windows)]
    {
        descriptor.required_features |= wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32;
    }
    if !adapter.features().contains(descriptor.required_features)
        || !descriptor.required_limits.check_limits(&adapter.limits())
    {
        return Err(error(
            "requested Vulkan device features or limits are unsupported",
        ));
    }
    // SAFETY: the callback only adds an extension whose support was checked above.
    let open = unsafe {
        hal.open_with_callback(
            descriptor.required_features,
            &descriptor.required_limits,
            &descriptor.memory_hints,
            Some(Box::new(|args| {
                let extension = required_extensions()[1];
                if !args.extensions.contains(&extension) {
                    args.extensions.push(extension);
                }
            })),
        )
    }
    .map_err(error)?;
    drop(hal);
    // SAFETY: the HAL device was opened from precisely this adapter with these requirements.
    unsafe { adapter.create_device_from_hal::<Api>(open, &descriptor) }.map_err(error)
}

pub(super) fn cuda_device(device: &wgpu::Device) -> Result<u32> {
    let hal = unsafe { device.as_hal::<Api>() }.ok_or_else(|| error("expected a Vulkan device"))?;
    for required in required_extensions() {
        if !hal.enabled_device_extensions().contains(&required) {
            return Err(error(format!(
                "Vulkan device did not enable {}; use create_vulkan_device",
                required.to_string_lossy()
            )));
        }
    }
    let instance = hal.shared_instance().raw_instance();
    let physical = hal.raw_physical_device();
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline);
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    if timeline.timeline_semaphore == 0 {
        return Err(error("Vulkan timeline semaphores are required"));
    }
    let mut sem_type =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let query = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(semaphore_handle_type())
        .push_next(&mut sem_type);
    let mut caps = vk::ExternalSemaphoreProperties::default();
    unsafe {
        instance.get_physical_device_external_semaphore_properties(physical, &query, &mut caps)
    };
    if !caps
        .external_semaphore_features
        .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
    {
        return Err(error(
            "Vulkan timeline semaphores cannot export the required platform handle",
        ));
    }
    result::init().map_err(error)?;
    for index in 0..result::device::get_count().map_err(error)? {
        let cuda = result::device::get(index).map_err(error)?;
        let uuid = result::device::get_uuid(cuda).map_err(error)?;
        if uuid.bytes.map(|b| b as u8) == id.device_uuid {
            if super::super::TrtxContext::list_devices_for_interop().contains(&(index as u32)) {
                return Ok(index as u32);
            }
            return Err(error(
                "matching CUDA device does not meet TensorRT-RTX requirements",
            ));
        }
    }
    Err(error("no CUDA device matches the Vulkan device UUID"))
}

pub(super) struct Allocation {
    // Keep wgpu's logical device alive until the raw resources have been destroyed.
    _device: wgpu::Device,
    raw: ash::Device,
    queue: vk::Queue,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    semaphores: [vk::Semaphore; 2],
    completion: vk::Semaphore,
    submitted: AtomicU64,
    pool: vk::CommandPool,
    release: vk::CommandBuffer,
    acquire: vk::CommandBuffer,
}

impl std::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanAllocation")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

impl Allocation {
    pub(super) fn new(
        device: &InteropDevice,
        ctx: &Arc<CudaContext>,
        size: u64,
    ) -> Result<(Self, CudaExternal)> {
        ctx.bind_to_thread().map_err(error)?;
        let hal =
            unsafe { device.device.as_hal::<Api>() }.ok_or_else(|| error("expected Vulkan"))?;
        let raw = hal.raw_device().clone();
        let instance = hal.shared_instance().raw_instance();
        let mut allocation = Self {
            _device: device.device.clone(),
            raw,
            queue: hal.raw_queue(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            semaphores: [vk::Semaphore::null(); 2],
            completion: vk::Semaphore::null(),
            submitted: AtomicU64::new(0),
            pool: vk::CommandPool::null(),
            release: vk::CommandBuffer::null(),
            acquire: vk::CommandBuffer::null(),
        };
        let mut completion_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        allocation.completion = unsafe {
            allocation.raw.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut completion_type),
                None,
            )
        }
        .map_err(error)?;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let query = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(memory_handle_type());
        let mut caps = vk::ExternalBufferProperties::default();
        unsafe {
            instance.get_physical_device_external_buffer_properties(
                hal.raw_physical_device(),
                &query,
                &mut caps,
            )
        };
        if !caps
            .external_memory_properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        {
            return Err(error(
                "Vulkan storage buffers cannot export the required platform handle",
            ));
        }
        let mut external =
            vk::ExternalMemoryBufferCreateInfo::default().handle_types(memory_handle_type());
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        allocation.buffer = unsafe { allocation.raw.create_buffer(&info, None) }.map_err(error)?;
        let requirements = unsafe {
            allocation
                .raw
                .get_buffer_memory_requirements(allocation.buffer)
        };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(hal.raw_physical_device()) };
        let memory_type = (0..memory_properties.memory_type_count)
            .find(|&index| {
                requirements.memory_type_bits & (1 << index) != 0
                    && memory_properties.memory_types[index as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| error("no device-local exportable buffer memory type"))?;
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(memory_handle_type());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(allocation.buffer);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        allocation.memory =
            unsafe { allocation.raw.allocate_memory(&info, None) }.map_err(error)?;
        unsafe {
            allocation
                .raw
                .bind_buffer_memory(allocation.buffer, allocation.memory, 0)
        }
        .map_err(error)?;
        let mut cuda = CudaExternal::empty(ctx);
        allocation.import_memory(instance, &mut cuda, requirements.size)?;
        cuda.map(size)?;
        for index in 0..2 {
            let mut export =
                vk::ExportSemaphoreCreateInfo::default().handle_types(semaphore_handle_type());
            let mut timeline =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            let info = vk::SemaphoreCreateInfo::default()
                .push_next(&mut export)
                .push_next(&mut timeline);
            allocation.semaphores[index] =
                unsafe { allocation.raw.create_semaphore(&info, None) }.map_err(error)?;
            allocation.import_semaphore(instance, &mut cuda, index)?;
        }
        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(hal.queue_family_index());
        allocation.pool =
            unsafe { allocation.raw.create_command_pool(&pool_info, None) }.map_err(error)?;
        let commands = unsafe {
            allocation.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(allocation.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(3),
            )
        }
        .map_err(error)?;
        allocation.release = commands[0];
        allocation.acquire = commands[1];
        for (command, src, dst, initialize) in [
            (
                allocation.release,
                hal.queue_family_index(),
                vk::QUEUE_FAMILY_EXTERNAL,
                false,
            ),
            (
                allocation.acquire,
                vk::QUEUE_FAMILY_EXTERNAL,
                hal.queue_family_index(),
                false,
            ),
            (
                commands[2],
                hal.queue_family_index(),
                vk::QUEUE_FAMILY_EXTERNAL,
                true,
            ),
        ] {
            unsafe {
                allocation
                    .raw
                    .begin_command_buffer(
                        command,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
                    )
                    .map_err(error)?;
                if initialize {
                    allocation
                        .raw
                        .cmd_fill_buffer(command, allocation.buffer, 0, size, 0);
                }
                let acquiring = src == vk::QUEUE_FAMILY_EXTERNAL;
                let barrier = vk::BufferMemoryBarrier::default()
                    .buffer(allocation.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .src_queue_family_index(src)
                    .dst_queue_family_index(dst)
                    .src_access_mask(if acquiring {
                        vk::AccessFlags::empty()
                    } else {
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
                    })
                    .dst_access_mask(if acquiring {
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
                    } else {
                        vk::AccessFlags::empty()
                    });
                allocation.raw.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
                allocation.raw.end_command_buffer(command).map_err(error)?;
            }
        }
        // Initialize the complete graphics-visible range and transfer initial ownership to CUDA.
        // Allocation is the only blocking graphics operation; normal handoffs remain GPU-side.
        let values = [1];
        let semaphores = [allocation.completion];
        let commands = [commands[2]];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let info = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline);
        unsafe {
            allocation
                .raw
                .queue_submit(allocation.queue, &[info], vk::Fence::null())
        }
        .map_err(error)?;
        allocation.submitted.store(1, Ordering::Relaxed);
        allocation.wait_idle()?;
        Ok((allocation, cuda))
    }

    #[cfg(target_os = "linux")]
    fn import_memory(
        &self,
        instance: &ash::Instance,
        cuda: &mut CudaExternal,
        size: u64,
    ) -> Result<()> {
        let api = ash::khr::external_memory_fd::Device::new(instance, &self.raw);
        let fd = unsafe {
            api.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(self.memory)
                    .handle_type(memory_handle_type()),
            )
        }
        .map_err(error)?;
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let mut desc = CudaExternal::memory_descriptor(
            sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
        );
        desc.handle.fd = fd.as_raw_fd();
        desc.size = size;
        desc.flags = sys::CUDA_EXTERNAL_MEMORY_DEDICATED;
        unsafe { sys::cuImportExternalMemory(&mut cuda.memory, &desc).result() }.map_err(error)?;
        // Successful CUDA FD imports consume ownership; failed imports leave it with us.
        let _ = fd.into_raw_fd();
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn import_semaphore(
        &self,
        instance: &ash::Instance,
        cuda: &mut CudaExternal,
        index: usize,
    ) -> Result<()> {
        let api = ash::khr::external_semaphore_fd::Device::new(instance, &self.raw);
        let fd = unsafe {
            api.get_semaphore_fd(
                &vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(self.semaphores[index])
                    .handle_type(semaphore_handle_type()),
            )
        }
        .map_err(error)?;
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let mut desc = CudaExternal::semaphore_descriptor(sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD);
        desc.handle.fd = fd.as_raw_fd();
        unsafe { sys::cuImportExternalSemaphore(&mut cuda.semaphores[index], &desc).result() }
            .map_err(error)?;
        let _ = fd.into_raw_fd();
        Ok(())
    }

    #[cfg(windows)]
    fn import_memory(
        &self,
        instance: &ash::Instance,
        cuda: &mut CudaExternal,
        size: u64,
    ) -> Result<()> {
        use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
        let api = ash::khr::external_memory_win32::Device::new(instance, &self.raw);
        let handle = unsafe {
            api.get_memory_win32_handle(
                &vk::MemoryGetWin32HandleInfoKHR::default()
                    .memory(self.memory)
                    .handle_type(memory_handle_type()),
            )
        }
        .map_err(error)?;
        let handle = unsafe { OwnedHandle::from_raw_handle(handle as *mut std::ffi::c_void) };
        let mut desc = CudaExternal::memory_descriptor(
            sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
        );
        desc.handle.win32.handle = handle.as_raw_handle();
        desc.size = size;
        desc.flags = sys::CUDA_EXTERNAL_MEMORY_DEDICATED;
        // CUDA retains the allocation, but does not consume an NT handle.
        unsafe { sys::cuImportExternalMemory(&mut cuda.memory, &desc).result() }.map_err(error)
    }

    #[cfg(windows)]
    fn import_semaphore(
        &self,
        instance: &ash::Instance,
        cuda: &mut CudaExternal,
        index: usize,
    ) -> Result<()> {
        use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
        let api = ash::khr::external_semaphore_win32::Device::new(instance, &self.raw);
        let handle = unsafe {
            api.get_semaphore_win32_handle(
                &vk::SemaphoreGetWin32HandleInfoKHR::default()
                    .semaphore(self.semaphores[index])
                    .handle_type(semaphore_handle_type()),
            )
        }
        .map_err(error)?;
        let handle = unsafe { OwnedHandle::from_raw_handle(handle as *mut std::ffi::c_void) };
        let mut desc = CudaExternal::semaphore_descriptor(sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32);
        desc.handle.win32.handle = handle.as_raw_handle();
        unsafe { sys::cuImportExternalSemaphore(&mut cuda.semaphores[index], &desc).result() }
            .map_err(error)
    }

    pub(super) fn wrap(
        &self,
        device: &wgpu::Device,
        size: u64,
        owner: Arc<SharedAllocation>,
    ) -> wgpu::Buffer {
        // SAFETY: initialization completed above; callback keeps native and CUDA storage alive.
        unsafe {
            let buffer = wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
                self.buffer,
                Box::new(move || drop(owner)),
            );
            device.create_buffer_from_hal::<Api>(
                buffer,
                &wgpu::BufferDescriptor {
                    label: Some("rustnn shared Vulkan/CUDA tensor"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )
        }
    }

    pub(super) fn release(&self, value: u64) -> Result<()> {
        let next = self
            .submitted
            .load(Ordering::Relaxed)
            .checked_add(1)
            .ok_or_else(|| error("native timeline exhausted"))?;
        let values = [value, next];
        let semaphores = [self.semaphores[0], self.completion];
        let commands = [self.release];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let info = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline);
        unsafe {
            self.raw
                .queue_submit(self.queue, &[info], vk::Fence::null())
        }
        .map_err(error)?;
        self.submitted.store(next, Ordering::Relaxed);
        Ok(())
    }

    pub(super) fn acquire(&self, value: u64) -> Result<()> {
        let next = self
            .submitted
            .load(Ordering::Relaxed)
            .checked_add(1)
            .ok_or_else(|| error("native timeline exhausted"))?;
        let values = [value];
        let signals = [next];
        let semaphores = [self.semaphores[1]];
        let completion = [self.completion];
        let commands = [self.acquire];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&values)
            .signal_semaphore_values(&signals);
        let info = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .wait_semaphores(&semaphores)
            .wait_dst_stage_mask(&stages)
            .signal_semaphores(&completion)
            .push_next(&mut timeline);
        unsafe {
            self.raw
                .queue_submit(self.queue, &[info], vk::Fence::null())
        }
        .map_err(error)?;
        self.submitted.store(next, Ordering::Relaxed);
        Ok(())
    }

    fn wait_idle(&self) -> Result<()> {
        let value = self.submitted.load(Ordering::Relaxed);
        if value == 0 {
            return Ok(());
        }
        // Host waits on our own completion semaphore need no external queue lock. In
        // particular, the final wgpu buffer callback may run during queue housekeeping.
        let semaphores = [self.completion];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.raw.wait_semaphores(&info, u64::MAX) }.map_err(error)
    }

    pub(super) fn idle(&self) {
        let _ = self.wait_idle();
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.idle();
        unsafe {
            if !self.completion.is_null() {
                self.raw.destroy_semaphore(self.completion, None);
            }
            if !self.pool.is_null() {
                self.raw.destroy_command_pool(self.pool, None);
            }
            for semaphore in self.semaphores {
                if !semaphore.is_null() {
                    self.raw.destroy_semaphore(semaphore, None);
                }
            }
            if !self.buffer.is_null() {
                self.raw.destroy_buffer(self.buffer, None);
            }
            if !self.memory.is_null() {
                self.raw.free_memory(self.memory, None);
            }
        }
    }
}
