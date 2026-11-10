// The shared model is fixed to [1, 3, 540, 540], RGB float32 in byte range.
const SIZE: u32 = 540u;
const PLANE: u32 = SIZE * SIZE;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}
@group(0) @binding(0) var<uniform> rotation: vec4<f32>;

@vertex fn triangle_vertex(@builtin(vertex_index) i: u32) -> VertexOutput {
    let positions = array(vec2(0.0, 0.78), vec2(-0.68, -0.40), vec2(0.68, -0.40));
    let colors = array(vec3(1.0, 0.15, 0.08), vec3(0.12, 1.0, 0.25), vec3(0.15, 0.35, 1.0));
    let p = positions[i];
    let c = cos(rotation.x);
    let s = sin(rotation.x);
    var out: VertexOutput;
    out.position = vec4(c * p.x - s * p.y, s * p.x + c * p.y, 0.0, 1.0);
    out.color = colors[i];
    return out;
}
@fragment fn triangle_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}

// Separate entry points use independent inferred pipeline layouts.
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@compute @workgroup_size(8, 8) fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= SIZE || id.y >= SIZE { return; }
    let rgb = textureLoad(scene, vec2<i32>(id.xy), 0).rgb * 255.0;
    let i = id.y * SIZE + id.x;
    input[i] = rgb.r;
    input[PLANE + i] = rgb.g;
    input[2u * PLANE + i] = rgb.b;
}

struct ScreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex fn screen_vertex(@builtin(vertex_index) i: u32) -> ScreenOutput {
    let p = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0))[i];
    var out: ScreenOutput;
    out.position = vec4(p, 0.0, 1.0);
    out.uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}
@group(0) @binding(0) var<storage, read> output: array<f32>;
fn model_rgb(uv: vec2<f32>) -> vec3<f32> {
    let xy = min(vec2<u32>(clamp(uv, vec2(0.0), vec2(1.0)) * f32(SIZE)), vec2(SIZE - 1u));
    let i = xy.y * SIZE + xy.x;
    return clamp(vec3(output[i], output[PLANE + i], output[2u * PLANE + i]) / 255.0, vec3(0.0), vec3(1.0));
}
@fragment fn screen_fragment(in: ScreenOutput) -> @location(0) vec4<f32> {
    return vec4(model_rgb(in.uv), 1.0);
}
@fragment fn screen_fragment_srgb(in: ScreenOutput) -> @location(0) vec4<f32> {
    let rgb = model_rgb(in.uv);
    let linear = select(pow((rgb + 0.055) / 1.055, vec3(2.4)), rgb / 12.92, rgb <= vec3(0.04045));
    return vec4(linear, 1.0);
}
