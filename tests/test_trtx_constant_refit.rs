//! Reproduce the int32-full-range WPT graphs with other constant data types.
//! The length matrix repeats the same values around 64 elements and at larger sizes
//! to distinguish size-dependent refit behavior from dtype and value effects.
//! Run explicitly with the TensorRT library paths set by the Makefile.
#![cfg(feature = "trtx-runtime")]

use rustnn::converters::TrtxConverter;
use rustnn::graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedTensors, MLPowerPreference,
    MLTensorDescriptor,
};
use rustnn::mlcontextoptions::RustNNOptions;
use rustnn::operators::Operation;

const SELECTION_A: [i32; 8] = [
    16777216,
    16777217,
    -16777216,
    -16777217,
    i32::MIN,
    i32::MIN + 1,
    i32::MAX,
    i32::MAX - 1,
];
const SELECTION_B: [i32; 8] = [
    16777217,
    16777216,
    -16777217,
    -16777216,
    i32::MIN + 1,
    i32::MIN,
    i32::MAX - 1,
    i32::MAX,
];
const DIV_A: [i32; 6] = [-33554435, 33554435, i32::MIN, i32::MAX, -7, 7];
const DIV_B: [i32; 6] = [2, -2, 1, 1, 3, -3];
const BINARY_OPS: [&str; 7] = ["min", "max", "div", "add", "sub", "mul", "pow"];

fn additional_arithmetic(op: &str) -> bool {
    matches!(op, "add" | "sub" | "mul" | "pow")
}

fn encode(dtype: DataType, values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| match dtype {
            DataType::Float32 => (v as f32).to_le_bytes().to_vec(),
            // Literal conversion of the WPT values: large values become infinities.
            DataType::Float16 => half::f16::from_f32(v as f32).to_le_bytes().to_vec(),
            DataType::Int32 => v.to_le_bytes().to_vec(),
            DataType::Uint32 => (v as u32).to_le_bytes().to_vec(),
            DataType::Int64 => (v as i64).to_le_bytes().to_vec(),
            DataType::Uint64 => (v as u64).to_le_bytes().to_vec(),
            DataType::Int8 => vec![v as i8 as u8],
            DataType::Uint8 => vec![v as u8],
            _ => unreachable!(),
        })
        .collect()
}

fn graph(dtype: DataType, op: &str, finite: bool, length: Option<usize>) -> GraphInfo {
    let (mut a, mut b) = if op == "div" {
        (DIV_A.to_vec(), DIV_B.to_vec())
    } else {
        (SELECTION_A.to_vec(), SELECTION_B.to_vec())
    };
    if finite && additional_arithmetic(op) {
        // No overflow, infinities, negative powers, or zero divisors, even for i8.
        a = vec![2, 3, 4, 5, 6, 7, 8, 9];
        b = vec![1, 2, 1, 2, 1, 2, 1, 2];
    } else if finite {
        // Supplement the exact dtype substitution with values representable in f16.
        a = if op == "div" {
            vec![-35, 35, -64, 63, -7, 7]
        } else {
            vec![16, 17, -16, -17, -64, -63, 63, 62]
        };
        if op != "div" {
            b = vec![17, 16, -17, -16, -63, -64, 62, 63];
        }
    }
    if let Some(length) = length {
        // Repeat the same values so only tensor length changes in the size sweep.
        a = a.into_iter().cycle().take(length).collect();
        b = b.into_iter().cycle().take(length).collect();
    }
    let descriptor = OperandDescriptor {
        data_type: dtype,
        shape: to_dimension_vector(&[a.len() as u32]),
        pending_permutation: vec![],
    };
    GraphInfo {
        operands: [
            OperandKind::Constant,
            OperandKind::Constant,
            OperandKind::Output,
        ]
        .into_iter()
        .enumerate()
        .map(|(id, kind)| Operand {
            kind,
            descriptor: descriptor.clone(),
            name: (id == 2).then(|| "output".into()),
        })
        .collect(),
        output_operands: vec![2],
        operations: vec![
            Operation::from_json_attributes(op, &[0, 1], &[2], &serde_json::json!({})).unwrap(),
        ],
        constant_operand_ids_to_handles: [
            (
                0,
                ConstantData {
                    data: encode(dtype, &a),
                    label: None,
                },
            ),
            (
                1,
                ConstantData {
                    data: encode(dtype, &b),
                    label: None,
                },
            ),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

/// Force both graph constants to be refittable, bypassing rustnn's baked-constant
/// exclusions. This recreates the original failing build/refit sequence.
fn force_refit(graph: &GraphInfo, strip: bool) -> Result<(), String> {
    let logger = trtx::Logger::stderr().map_err(|e| e.to_string())?;
    let mut builder = trtx::Builder::new(&logger).map_err(|e| e.to_string())?;
    let mut network = builder.create_network(0).map_err(|e| e.to_string())?;
    TrtxConverter::build_network(graph, &mut network).map_err(|e| format!("lowering: {e}"))?;
    let mut config = builder.create_config().map_err(|e| e.to_string())?;
    config.set_flag(trtx::trtx_sys::BuilderFlag::kREFIT_INDIVIDUAL);
    if strip {
        config.set_flag(trtx::trtx_sys::BuilderFlag::kSTRIP_PLAN);
    }
    for id in [0, 1] {
        network
            .mark_weights_refittable(&id.to_string())
            .map_err(|e| format!("mark {id}: {e}"))?;
    }
    let plan = builder
        .build_serialized_network(&mut network, &mut config)
        .map_err(|e| format!("build: {e}"))?;
    let mut runtime = trtx::Runtime::new(&logger).map_err(|e| e.to_string())?;
    let engine = runtime
        .deserialize_cuda_engine(&plan)
        .map_err(|e| format!("load: {e}"))?;
    let mut refitter = trtx::Refitter::new(&engine, &logger).map_err(|e| e.to_string())?;
    // Keep every supplied buffer alive until refit_cuda_engine finishes.
    let mut refit_buffers = Vec::new();
    for id in [0, 1] {
        let name = id.to_string();
        let prototype = refitter.weights_prototype(&name);
        if prototype.count < 0 {
            return Err(format!("missing refit prototype for constant {id}"));
        }
        let data = &graph.constant_operand_ids_to_handles[&id].data;
        let storage_type: trtx::DataType = prototype.type_.clone().into();
        let storage = match (
            graph.operands[id as usize].descriptor.data_type,
            storage_type,
        ) {
            (DataType::Int8, trtx::DataType::kINT32) => data
                .iter()
                .flat_map(|&v| (v as i8 as i32).to_le_bytes())
                .collect(),
            (DataType::Uint8, trtx::DataType::kINT32) => data
                .iter()
                .flat_map(|&v| (v as i32).to_le_bytes())
                .collect(),
            _ => data.clone(),
        };
        if storage.len() != prototype.count as usize * storage_type.size_bits() / 8 {
            return Err(format!(
                "unexpected constant {id} storage: {storage_type:?}"
            ));
        }
        refit_buffers.push(storage);
        unsafe {
            refitter.set_named_weights_with_location(
                &name,
                trtx::trtx_sys::Weights {
                    type_: prototype.type_.clone(),
                    values: refit_buffers.last().unwrap().as_ptr().cast(),
                    count: prototype.count,
                },
                trtx::trtx_sys::nvinfer1::TensorLocation::kHOST,
            )
        }
        .map_err(|e| format!("set {id}: {e}"))?;
    }
    refitter
        .refit_cuda_engine()
        .map_err(|e| format!("refit: {e}"))
}

#[test]
#[ignore = "manual TensorRT SDK diagnostic; explicitly forces refit of baked constants"]
fn constant_refit_dtype_matrix() {
    check_constant_refit_dtype_matrix(None);
}

fn check_constant_refit_dtype_matrix(length: Option<usize>) {
    let mut options = RustNNOptions::default();
    options.trtx.engine_caching = false;
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, true)
            .with_rustnn_backend_hint(Backend::Trtx)
            .with_rustnn_options(options),
    )
    .unwrap();
    for dtype in [
        DataType::Float32,
        DataType::Float16,
        DataType::Int32,
        DataType::Uint32,
        DataType::Int64,
        DataType::Uint64,
        DataType::Int8,
        DataType::Uint8,
    ] {
        for op in BINARY_OPS {
            for finite in [false, true] {
                if finite
                    && dtype != DataType::Float16
                    && !additional_arithmetic(op)
                    && !matches!(dtype, DataType::Uint32 | DataType::Uint64)
                {
                    continue;
                }
                let normal_build = MLGraphBuilder::new(&mut context)
                    .unwrap()
                    .build_graph_info(graph(dtype, op, finite, length))
                    .map(|_| ());
                println!("BACKEND {dtype:?} {op} finite={finite}: {normal_build:?}");
                for strip in [false, true] {
                    let result = force_refit(&graph(dtype, op, finite, length), strip);
                    println!("REFIT {dtype:?} {op} finite={finite} strip={strip}: {result:?}");
                    // TensorRT-RTX 1.7 retains INT32 refit prototypes above 64 elements.
                    if matches!(dtype, DataType::Float32 | DataType::Float16)
                        || (matches!(dtype, DataType::Int32 | DataType::Uint32)
                            && length.is_some_and(|length| length > 64))
                    {
                        result.unwrap();
                    } else if let Err(error) = result {
                        assert!(
                            error.starts_with("missing refit prototype")
                                || (op == "pow" && error.starts_with("build:")),
                            "{error}"
                        );
                    }
                }
            }
        }
    }
}

fn decode(dtype: DataType, bytes: &[u8]) -> Vec<f64> {
    let width = match dtype {
        DataType::Float16 => 2,
        DataType::Int64 | DataType::Uint64 => 8,
        _ => 4,
    };
    bytes
        .chunks_exact(width)
        .map(|v| match dtype {
            DataType::Float32 => f32::from_le_bytes(v.try_into().unwrap()) as f64,
            DataType::Float16 => half::f16::from_le_bytes(v.try_into().unwrap()).to_f64(),
            DataType::Int32 => i32::from_le_bytes(v.try_into().unwrap()) as f64,
            DataType::Uint32 => u32::from_le_bytes(v.try_into().unwrap()) as f64,
            DataType::Int64 => i64::from_le_bytes(v.try_into().unwrap()) as f64,
            DataType::Uint64 => u64::from_le_bytes(v.try_into().unwrap()) as f64,
            _ => unreachable!(),
        })
        .collect()
}

#[test]
#[ignore = "requires a TensorRT-RTX GPU and SDK library paths"]
fn dtype_substitutions_execute_correctly() {
    check_dtype_substitutions_execute_correctly(None);
}

fn check_dtype_substitutions_execute_correctly(length: Option<usize>) {
    // Leave engine caching enabled. Rebuild identical topologies with different
    // bounded constants below to catch reuse of a stale constant-folded result.
    let options = RustNNOptions::default();
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, true)
            .with_rustnn_backend_hint(Backend::Trtx)
            .with_rustnn_options(options),
    )
    .unwrap();
    for dtype in [
        DataType::Float32,
        DataType::Float16,
        DataType::Int32,
        DataType::Uint32,
        DataType::Int64,
        DataType::Uint64,
    ] {
        for op in BINARY_OPS {
            for finite in [false, true] {
                if finite
                    && dtype != DataType::Float16
                    && !additional_arithmetic(op)
                    && !matches!(dtype, DataType::Uint32 | DataType::Uint64)
                {
                    continue;
                }
                let mut info = graph(dtype, op, finite, length);
                let bounded = matches!(dtype, DataType::Uint32 | DataType::Uint64)
                    || (additional_arithmetic(op)
                        && matches!(dtype, DataType::Int32 | DataType::Int64));
                if bounded {
                    // Both revisions fit every tested integer type; only the
                    // weights differ, so a topology-only cache must be bypassed.
                    let count = info.operands[0].descriptor.element_count().unwrap();
                    let revision = i32::from(finite) * 10;
                    let a: Vec<i32> = (0..count).map(|i| (i % 8) as i32 + 2 + revision).collect();
                    let b: Vec<i32> = (0..count).map(|i| i as i32 % 2 + 1).collect();
                    info.constant_operand_ids_to_handles
                        .get_mut(&0)
                        .unwrap()
                        .data = encode(dtype, &a);
                    info.constant_operand_ids_to_handles
                        .get_mut(&1)
                        .unwrap()
                        .data = encode(dtype, &b);
                }
                let a = decode(dtype, &info.constant_operand_ids_to_handles[&0].data);
                let b = decode(dtype, &info.constant_operand_ids_to_handles[&1].data);
                let mut expected: Vec<f64> = a
                    .iter()
                    .zip(&b)
                    .map(|(&a, &b)| match op {
                        "min" => a.min(b),
                        "max" => a.max(b),
                        "div" => a / b,
                        "add" => a + b,
                        "sub" => a - b,
                        "mul" => a * b,
                        "pow" => a.powf(b),
                        _ => unreachable!(),
                    })
                    .collect();
                for value in &mut expected {
                    *value = match dtype {
                        DataType::Float32 => (*value as f32) as f64,
                        DataType::Float16 => half::f16::from_f64(*value).to_f64(),
                        _ => value.trunc(),
                    };
                }
                let mut compiled = MLGraphBuilder::new(&mut context)
                    .unwrap()
                    .build_graph_info(info.clone())
                    .unwrap();
                let mut desc =
                    MLTensorDescriptor::new(dtype.try_into().unwrap(), vec![a.len() as u64]);
                desc.set_readable(true);
                let output = context.create_tensor(&desc).unwrap();
                context
                    .dispatch(
                        &mut compiled,
                        &MLNamedTensors::new(),
                        &MLNamedTensors::from([("output", &output)]),
                    )
                    .unwrap();
                let mut bytes = vec![0u8; info.operands[2].descriptor.byte_length().unwrap()];
                context.read_tensor(&output, &mut bytes).unwrap();
                let actual = decode(dtype, &bytes);
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        actual == expected || (actual.is_nan() && expected.is_nan()),
                        "{dtype:?} {op} finite={finite} index={index}: {actual} != {expected}"
                    );
                }
                println!("EXECUTE {dtype:?} {op} finite={finite}: correct");
            }
        }
    }
}

#[test]
#[ignore = "manual TensorRT SDK diagnostic; sweeps constant lengths around 64 elements"]
fn constant_refit_length_matrix() {
    for length in [63, 64, 65, 128, 1024] {
        println!("LENGTH {length}: forced refit");
        check_constant_refit_dtype_matrix(Some(length));
        println!("LENGTH {length}: execution");
        check_dtype_substitutions_execute_correctly(Some(length));
    }
}
