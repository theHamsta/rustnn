//! Hardware integration tests. Run explicitly with --ignored on an NVIDIA GPU.
#![cfg(all(
    feature = "trtx-runtime",
    feature = "wgpu-native",
    any(target_os = "linux", windows)
))]

use rustnn::{
    mlcontext::{
        GpuDevice, MLContext, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
        MLTensorDescriptor, RustNNOptions, WgpuDevice,
    },
    mlgraphbuilder::MLGraphBuilder,
    operator_enums::MLOperandDataType,
};

const ELEMENTS: usize = 256;

fn compute(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, expression: &str) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("interop test compute"),
        source: wgpu::ShaderSource::Wgsl(format!(
            "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
             @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
             let i = id.x; if i < {ELEMENTS}u {{ data[i] = {expression}; }} }}"
        ).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((ELEMENTS as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
}

fn round_trip(backend: wgpu::Backends) {
    let _ = pretty_env_logger::try_init();
    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = backend;
    let instance = wgpu::Instance::new(instance_descriptor);
    let Some(adapter) = pollster::block_on(instance.enumerate_adapters(backend))
        .into_iter()
        .find(|a| a.get_info().vendor == 0x10de)
    else {
        log::warn!(
            "skipping wgpu test because no Nvidia adapter found for wgpu backend {backend:?}"
        );
        return;
    };

    let descriptor = wgpu::DeviceDescriptor {
        label: Some("rustnn CUDA interop test"),
        ..Default::default()
    };
    let (device, queue) = if backend == wgpu::Backends::VULKAN {
        rustnn::backends::trtx::wgpu_interop::create_vulkan_device(&adapter, &descriptor).unwrap()
    } else {
        pollster::block_on(adapter.request_device(&descriptor)).unwrap()
    };

    for cuda_graphs in [false, true] {
        let mut options = RustNNOptions::default();
        options.trtx.cuda_graphs = cuda_graphs;
        // SAFETY: this test uses this device's queue on one thread and submits all exported
        // resource uses before returning ownership. No other thread submits on this queue.
        let gpu = unsafe { WgpuDevice::new(device.clone(), queue.clone()) };
        let mut context =
            MLContext::create_from_gpu_device(GpuDevice::WgpuDevice(gpu), options).unwrap();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let input = builder
            .input(
                "input",
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![ELEMENTS as u64]),
            )
            .unwrap();
        let output = builder.relu(input).unwrap();
        let mut graph = builder
            .build(&MLNamedOperands::from([("output", output)]))
            .unwrap();
        let mut desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![ELEMENTS as u64]);
        desc.set_readable(true);
        desc.set_writable(true);
        let input = context.create_exportable_tensor(&desc).unwrap();
        let output = context.create_exportable_tensor(&desc).unwrap();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interop readback"),
            size: (ELEMENTS * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for iteration in 0..8 {
            let export = context.export_to_gpu(&input).unwrap();
            // SAFETY: use is submitted now, and no buffer references are reused after return.
            let buffer = unsafe { export.as_wgpu_buffer() }.unwrap();
            compute(
                &device,
                &queue,
                buffer,
                &format!("f32(i % 16u) - 8.0 + {iteration}.0"),
            );
            assert!(context.write_tensor(&input, &[0f32; ELEMENTS]).is_err());
            let mut resize = input.clone();
            assert!(
                context
                    .rustnn_resize_tensor(&mut resize, &[ELEMENTS as u64])
                    .is_err()
            );
            context.destroy_gpu_buffer(export).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("output", &output)]),
                )
                .unwrap();

            // Re-export the input immediately after enqueue: inference must finish reading
            // before this compute overwrites it. No host synchronization occurs here.
            let input_export = context.export_to_gpu(&input).unwrap();
            compute(
                &device,
                &queue,
                unsafe { input_export.as_wgpu_buffer() }.unwrap(),
                "-999.0",
            );
            context.destroy_gpu_buffer(input_export).unwrap();

            let export = context.export_to_gpu(&output).unwrap();
            let buffer = unsafe { export.as_wgpu_buffer() }.unwrap();
            compute(&device, &queue, buffer, "data[i] * 2.0");
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (ELEMENTS * 4) as u64);
            queue.submit([encoder.finish()]);
            context.destroy_gpu_buffer(export).unwrap();
            let (sender, receiver) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap()
                });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            receiver.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range().unwrap();
            let actual: &[f32] = bytemuck::cast_slice(&mapped);
            let expected: Vec<f32> = (0..ELEMENTS)
                .map(|i| ((i % 16) as f32 - 8.0 + iteration as f32).max(0.0) * 2.0)
                .collect();
            assert_eq!(
                actual, expected,
                "iteration={iteration}, cuda_graphs={cuda_graphs}"
            );
            drop(mapped);
            staging.unmap();
            // Check graphics writes are visible through CUDA's host-read path, too.
            let mut actual = vec![0f32; ELEMENTS];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
        let mut resize = input.clone();
        assert!(
            context
                .rustnn_resize_tensor(&mut resize, &[(ELEMENTS * 2) as u64])
                .is_err()
        );
    }
}

#[test]
fn vulkan_cuda_trtx_round_trip() {
    round_trip(wgpu::Backends::VULKAN);
}

#[cfg(windows)]
#[test]
fn d3d12_cuda_trtx_round_trip() {
    round_trip(wgpu::Backends::DX12);
}
