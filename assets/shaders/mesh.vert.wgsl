struct Vertex {
	vx: f32,
    vy: f32,
    vz: f32,
    vpad: f32,
    normal: u32,
    tu: f32,
    tv: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>
};

@group(0) @binding(0)
var<storage, read> vertices: array<Vertex>;

@vertex
fn vertex(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    let vertex = vertices[in_vertex_index];

    let pos = vec3(vertex.vx, vertex.vy, vertex.vz);

    let nx = f32((vertex.normal >> 0u) & 0xFFu);
	let ny = f32((vertex.normal >> 8u) & 0xFFu);
	let nz = f32((vertex.normal >> 16u) & 0xFFu);
    let normal = vec3(nx, ny, nz) / 127.0 - 1.0;

    var out: VertexOutput;

    let offset = vec3(0.25, -0.75, 0.5);
    let scale = vec3(1.0, 1.0, 0.5);

    out.clip_position = vec4(pos * scale + offset, 1.0);
    out.color = normal * 0.5 + vec3(0.5);
    return out;
}