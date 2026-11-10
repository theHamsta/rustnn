// Fast Style Transfer using the public MLContext builder and tensor API.
#[path = "support/fast_style_transfer.rs"]
mod fast_style_transfer;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use fast_style_transfer::*;
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use tokio::sync::Mutex;
const CHANNELS: usize = 3;
const IMAGE_ELEMENT_COUNT: usize = CHANNELS * HEIGHT * WIDTH;
const PIPELINE_DEPTH: usize = 2;

#[derive(Parser, Debug)]
#[command(about = "Load Fast Style Transfer weights into RustNN")]
struct Args {
    #[arg(short, long, help = "Input image path")]
    input: PathBuf,
    #[arg(short, long, help = "Output image path")]
    output: PathBuf,
    #[arg(long, default_value_t = 1, help = "Number of inferences to run")]
    num_inferences: usize,
    #[arg(
        long,
        help = "Disable pipelined tensor buffering and read each result before dispatching the next inference"
    )]
    disable_pipelining: bool,
    #[arg(long, value_enum, default_value_t = StyleModel::StarryNight, visible_alias = "model-id")]
    model: StyleModel,
    #[arg(long, default_value = DEFAULT_WEIGHTS_BASE_URL)]
    weights_base_url: String,
    #[arg(long)]
    sequential_weights: bool,
}

fn image_to_tensor(image: &image::RgbImage) -> Vec<f32> {
    let resized = image::imageops::resize(
        image,
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );
    let plane = HEIGHT * WIDTH;
    let mut tensor = vec![0.0f32; IMAGE_ELEMENT_COUNT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let idx = y * WIDTH + x;
            for channel in 0..CHANNELS {
                tensor[channel * plane + idx] = pixel[channel] as f32;
            }
        }
    }
    tensor
}

fn load_input_image(path: &Path) -> Result<Vec<f32>> {
    let image = image::open(path)
        .with_context(|| format!("decode input image {}", path.display()))?
        .to_rgb8();
    Ok(image_to_tensor(&image))
}

fn save_output_image(path: &PathBuf, output: &[f32]) -> Result<()> {
    ensure!(
        output.len() == IMAGE_ELEMENT_COUNT,
        "output tensor length {}, expected {}",
        output.len(),
        IMAGE_ELEMENT_COUNT
    );
    let plane = HEIGHT * WIDTH;
    let mut image = image::RgbImage::new(WIDTH as u32, HEIGHT as u32);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = y * WIDTH + x;
            let pixel = image::Rgb([
                output[idx].round().clamp(0.0, 255.0) as u8,
                output[plane + idx].round().clamp(0.0, 255.0) as u8,
                output[2 * plane + idx].round().clamp(0.0, 255.0) as u8,
            ]);
            image.put_pixel(x as u32, y as u32, pixel);
        }
    }
    image
        .save(path)
        .with_context(|| format!("write output image {}", path.display()))
}

fn run_inferences(
    context: &mut MLContext<'_>,
    graph: &mut rustnn::mlcontext::MLGraph<'_>,
    input_data: &[f32],
    num_inferences: usize,
    pipelined: bool,
) -> Result<Vec<f32>> {
    let mut input_descriptor =
        MLTensorDescriptor::new(MLOperandDataType::Float32, INPUT_SHAPE.to_vec());
    input_descriptor.set_writable(true);
    let mut output_descriptor =
        MLTensorDescriptor::new(MLOperandDataType::Float32, OUTPUT_SHAPE.to_vec());
    output_descriptor.set_readable(true);

    let tensor_count = if pipelined { PIPELINE_DEPTH } else { 1 };
    let mut input_tensors = Vec::with_capacity(tensor_count);
    let mut output_tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        input_tensors.push(
            context
                .create_tensor(&input_descriptor)
                .context("create input tensor")?,
        );
        output_tensors.push(
            context
                .create_tensor(&output_descriptor)
                .context("create output tensor")?,
        );
    }

    let mut output_data = vec![0.0f32; IMAGE_ELEMENT_COUNT];
    let mut starts = vec![None; tensor_count];
    let mut next_to_read = 0;

    for i in 0..num_inferences {
        let slot = i % tensor_count;
        context
            .write_tensor(&input_tensors[slot], input_data)
            .context("write input tensor")?;
        starts[slot] = Some(Instant::now());
        let inputs = MLNamedTensors::from([("input", &input_tensors[slot])]);
        let outputs = MLNamedTensors::from([("output", &output_tensors[slot])]);
        context
            .dispatch(graph, &inputs, &outputs)
            .context("dispatch graph")?;

        if i + 1 >= tensor_count {
            let completed_slot = next_to_read % tensor_count;
            context
                .read_tensor(&output_tensors[completed_slot], &mut output_data)
                .context("read output tensor")?;
            next_to_read += 1;
        }
    }

    while next_to_read < num_inferences {
        let completed_slot = next_to_read % tensor_count;
        context
            .read_tensor(&output_tensors[completed_slot], &mut output_data)
            .context("drain output tensor")?;
        println!(
            "inference {} took {:?}",
            next_to_read + 1,
            starts[completed_slot]
                .take()
                .expect("inference start time must be recorded")
                .elapsed()
        );
        next_to_read += 1;
    }

    Ok(output_data)
}

#[tokio::main]
async fn main() -> Result<()> {
    pretty_env_logger::init();
    let args = Args::parse();
    ensure!(args.num_inferences >= 1, "--num-inferences must be >= 1");

    let input = &args.input;
    println!("Input image: {}", input.display());
    println!("Output image: {}", args.output.display());
    println!("Requested inferences: {}", args.num_inferences);
    println!(
        "Inference tensor buffering: {}",
        if args.disable_pipelining {
            "disabled"
        } else {
            "pipeline depth 3"
        }
    );
    println!("Model: {} ({})", args.model.id(), args.model.title());
    println!("Input shape: {:?}", INPUT_SHAPE);
    println!("Output shape: {:?}", OUTPUT_SHAPE);

    let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true);
    let mut context = MLContext::create(&options).context(
        "create MLContext; run this example with an enabled RustNN backend feature enabled and make sure the runtime libraries can be found",
    )?;
    let builder = Mutex::new(MLGraphBuilder::new(&mut context).context("create MLGraphBuilder")?);
    let client = reqwest::Client::new();
    let model_base_url = format!(
        "{}/{}",
        args.weights_base_url.trim_end_matches('/'),
        args.model.id()
    );

    let loaded = load_weights(&client, &builder, &model_base_url, args.sequential_weights).await?;
    let total_bytes: usize = loaded.iter().map(|weight| weight.byte_len).sum();

    println!(
        "Loaded {} Fast Style Transfer weight tensors ({} bytes total).",
        loaded.len(),
        total_bytes
    );

    println!("Building Fast Style Transfer graph...");
    let mut graph = {
        let mut builder = builder.lock().await;
        let output_operand = build_fast_style_transfer_graph(&mut builder, &loaded)?;
        let inferred_output_shape = builder
            .rustnn_operand_shape(output_operand)
            .context("infer output shape")?;
        ensure!(
            inferred_output_shape == OUTPUT_SHAPE,
            "inferred output shape {:?}, expected {:?}",
            inferred_output_shape,
            OUTPUT_SHAPE
        );
        let outputs = MLNamedOperands::from([("output", output_operand)]);
        builder.build(&outputs).context("build MLGraph")?
    };

    let input_data = load_input_image(input)?;
    let output_data = run_inferences(
        &mut context,
        &mut graph,
        &input_data,
        args.num_inferences,
        !args.disable_pipelining,
    )?;
    save_output_image(&args.output, &output_data)?;
    println!("Wrote output image: {}", args.output.display());

    Ok(())
}
