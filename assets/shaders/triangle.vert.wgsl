struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// TODO use const array, blocked by naga: https://github.com/gfx-rs/naga/issues/1829
var<private> vertices:array<vec3<f32>, 3> = array<vec3<f32>, 3>(
	vec3<f32>( 0.0,  0.5, 0.0),
	vec3<f32>( 0.5, -0.5, 0.0),
	vec3<f32>(-0.5, -0.5, 0.0),
);

@vertex
fn vertex(@builtin(vertex_index) in_vertex_index: u32,) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertices[in_vertex_index], 1.0);
    return out;
}