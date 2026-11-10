//! Render -> shared ML input -> style transfer -> shared ML output -> window.
#[path = "support/fast_style_transfer.rs"]
mod fast_style_transfer;

use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use fast_style_transfer::*;
use rustnn::mlcontext::{
    GpuDevice, MLContext, MLGraph, MLGraphBuilder, MLNamedOperands, MLNamedTensors, MLTensor,
    MLTensorDescriptor, RustNNOptions, WgpuDevice,
};
use rustnn::operator_enums::MLOperandDataType;
use tokio::sync::Mutex;
use tracing_subscriber::{EnvFilter, prelude::*};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

#[derive(Debug, Parser)]
#[command(about = "Style a spinning wgpu triangle with MLContext graphics interop")]
struct Args {
    #[arg(long, value_enum, default_value_t = StyleModel::StarryNight)]
    model: StyleModel,
    #[arg(long, default_value = DEFAULT_WEIGHTS_BASE_URL)]
    weights_base_url: String,
    /// Exit after presenting this many frames (useful for smoke testing).
    #[arg(long)]
    frames: Option<u64>,
}

struct Demo {
    // All queue submissions and MLContext ownership handoffs happen on the event thread.
    context: MLContext<'static>,
    graph: MLGraph<'static>,
    input: MLTensor,
    output: MLTensor,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scene: wgpu::TextureView,
    triangle: wgpu::RenderPipeline,
    pack: wgpu::ComputePipeline,
    present: wgpu::RenderPipeline,
    rotation: wgpu::Buffer,
    rotation_binding: wgpu::BindGroup,
    started: Instant,
    presented: u64,
}

fn pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    vertex: &str,
    fragment: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(vertex),
        layout: None,
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl Demo {
    #[tracing::instrument(name = "initialize_style_transfer", skip_all)]
    fn new(window: Arc<Window>, args: &Args) -> Result<Self> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
            .into_iter()
            .find(|a| a.get_info().vendor == 0x10de && a.is_surface_supported(&surface))
            .context("no NVIDIA Vulkan adapter can present to this window")?;
        println!("Adapter: {}", adapter.get_info().name);
        #[cfg(any(target_os = "linux", windows))]
        let (device, queue) = WgpuDevice::create_vulkan_device(&adapter, &Default::default())?;
        #[cfg(not(any(target_os = "linux", windows)))]
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))?;

        // SAFETY: the queue belongs to device; only this thread submits work. All shared
        // buffer uses are submitted before returning each export to MLContext.
        let gpu = unsafe { WgpuDevice::new(device.clone(), queue.clone()) };
        let mut context = MLContext::create_from_gpu_device(
            GpuDevice::WgpuDevice(gpu),
            RustNNOptions::default(),
        )?;
        println!(
            "Loading {} weights and compiling model...",
            args.model.title()
        );
        let runtime = tokio::runtime::Runtime::new()?;
        let mut graph = runtime.block_on(async {
            let builder = Mutex::new(MLGraphBuilder::new(&mut context)?);
            let url = format!(
                "{}/{}",
                args.weights_base_url.trim_end_matches('/'),
                args.model.id()
            );
            let loaded = load_weights(&reqwest::Client::new(), &builder, &url, false).await?;
            let mut builder = builder.lock().await;
            let output = build_fast_style_transfer_graph(&mut builder, &loaded)?;
            ensure!(
                builder.rustnn_operand_shape(output)? == OUTPUT_SHAPE,
                "unexpected output shape"
            );
            builder
                .build(&MLNamedOperands::from([("output", output)]))
                .context("compile style transfer graph")
        })?;
        let desc = MLTensorDescriptor::new(MLOperandDataType::Float32, INPUT_SHAPE.to_vec());
        let input = context.create_exportable_tensor(&desc)?;
        let output = context.create_exportable_tensor(&desc)?;
        // Warm up compilation before animation starts. Newly allocated shared tensors are zeroed.
        context.dispatch(
            &mut graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("output", &output)]),
        )?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("surface has no supported configuration")?;
        // The model consumes and produces RGB byte values. Prefer a non-sRGB swapchain
        // so presentation does not apply a second gamma encoding.
        config.format = surface
            .get_capabilities(&adapter)
            .formats
            .into_iter()
            .find(|f| !f.is_srgb())
            .unwrap_or(config.format);
        surface.configure(&device, &config);
        let scene = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("triangle at model resolution"),
                size: wgpu::Extent3d {
                    width: WIDTH as u32,
                    height: HEIGHT as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle, NCHW packing and presentation"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("support/wgpu_style_transfer.wgsl").into(),
            ),
        });
        let triangle = pipeline(
            &device,
            &shader,
            "triangle_vertex",
            "triangle_fragment",
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let present = pipeline(
            &device,
            &shader,
            "screen_vertex",
            if config.format.is_srgb() {
                "screen_fragment_srgb"
            } else {
                "screen_fragment"
            },
            config.format,
        );
        let pack = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RGB texture to float32 NCHW"),
            layout: None,
            module: &shader,
            entry_point: Some("pack"),
            compilation_options: Default::default(),
            cache: None,
        });
        let rotation = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rotation"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rotation_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rotation"),
            layout: &triangle.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: rotation.as_entire_binding(),
            }],
        });
        Ok(Self {
            context,
            graph,
            input,
            output,
            device,
            queue,
            surface,
            config,
            window,
            scene,
            triangle,
            pack,
            present,
            rotation,
            rotation_binding,
            started: Instant::now(),
            presented: 0,
        })
    }

    fn resize(&mut self) {
        let size = self.window.inner_size();
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[tracing::instrument(name = "style_transfer_frame", skip_all, fields(payload = self.presented))]
    fn render(&mut self) -> Result<()> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                bail!("presentation surface lost; restart the demo")
            }
            wgpu::CurrentSurfaceTexture::Validation => bail!("surface validation failed"),
        };
        let angle = self.started.elapsed().as_secs_f32();
        self.queue.write_buffer(
            &self.rotation,
            0,
            bytemuck::cast_slice(&[angle, 0.0, 0.0, 0.0]),
        );
        let input_export = self.context.export_to_gpu(&self.input)?;
        {
            // SAFETY: the bind group and encoder stay in this scope, and their work is
            // submitted before destroy_gpu_buffer returns ownership to inference.
            let buffer = unsafe { input_export.as_wgpu_buffer() }.context("expected wgpu input")?;
            let binding = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shared model input"),
                layout: &self.pack.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.scene),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.08,
                                g: 0.10,
                                b: 0.16,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.triangle);
                pass.set_bind_group(0, &self.rotation_binding, &[]);
                pass.draw(0..3, 0..1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pack);
                pass.set_bind_group(0, &binding, &[]);
                pass.dispatch_workgroups(
                    (WIDTH as u32).div_ceil(8),
                    (HEIGHT as u32).div_ceil(8),
                    1,
                );
            }
            self.queue.submit([encoder.finish()]);
        }
        self.context.destroy_gpu_buffer(input_export)?;
        self.context.dispatch(
            &mut self.graph,
            &MLNamedTensors::from([("input", &self.input)]),
            &MLNamedTensors::from([("output", &self.output)]),
        )?;
        let output_export = self.context.export_to_gpu(&self.output)?;
        {
            // SAFETY: as above, no shared buffer or bind group is reused after return.
            let buffer =
                unsafe { output_export.as_wgpu_buffer() }.context("expected wgpu output")?;
            let binding = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shared model output"),
                layout: &self.present.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            let view = frame.texture.create_view(&Default::default());
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                // Keep the fixed square model image square when the window is resized.
                let side = self.config.width.min(self.config.height) as f32;
                pass.set_viewport(
                    (self.config.width as f32 - side) / 2.0,
                    (self.config.height as f32 - side) / 2.0,
                    side,
                    side,
                    0.0,
                    1.0,
                );
                pass.set_pipeline(&self.present);
                pass.set_bind_group(0, &binding, &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue.submit([encoder.finish()]);
        }
        self.context.destroy_gpu_buffer(output_export)?;
        self.window.pre_present_notify();
        self.queue.present(frame);
        self.presented += 1;
        if self.presented.is_multiple_of(60) {
            self.window.set_title(&format!(
                "rustnn style transfer | {:.1} FPS",
                self.presented as f64 / self.started.elapsed().as_secs_f64()
            ));
        }
        Ok(())
    }
}

struct App {
    args: Args,
    demo: Option<Demo>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.demo.is_some() {
            return;
        }
        let result = (|| {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title("rustnn | spinning triangle style transfer")
                        .with_inner_size(LogicalSize::new(810, 810)),
                )?,
            );
            Demo::new(window, &self.args)
        })();
        match result {
            Ok(demo) => {
                demo.window.request_redraw();
                self.demo = Some(demo);
            }
            Err(error) => {
                self.error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(demo) = self.demo.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit()
            }
            WindowEvent::Resized(_) => demo.resize(),
            WindowEvent::RedrawRequested => {
                if let Err(error) = demo.render() {
                    self.error = Some(error);
                    event_loop.exit();
                    return;
                }
                if self.args.frames.is_some_and(|n| demo.presented >= n) {
                    println!(
                        "Presented {} styled frames in {:.2}s",
                        demo.presented,
                        demo.started.elapsed().as_secs_f64()
                    );
                    event_loop.exit();
                } else {
                    demo.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let fmt = tracing_subscriber::fmt::layer()
        .compact()
        .with_writer(std::io::stderr)
        .without_time()
        .with_filter(env_filter);
    tracing_subscriber::registry()
        .with(nvtx::tracing::NvtxLayer::default())
        .with(fmt)
        .init();
    let span = tracing::info_span!("wgpu_style_transfer");
    let _enter = span.enter();
    let args = Args::parse();
    ensure!(args.frames != Some(0), "--frames must be at least 1");
    let mut app = App {
        args,
        demo: None,
        error: None,
    };
    EventLoop::new()?.run_app(&mut app)?;
    if let Some(error) = app.error {
        return Err(error);
    }
    Ok(())
}
