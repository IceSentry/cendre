struct Vertex {
    @location(0) position: vec3<f32>
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    // TODO use const array, blocked by naga: https://github.com/gfx-rs/naga/issues/1829
    let vertices = array(
        vec3( 0.0,  0.5, 0.0),
        vec3( 0.5, -0.5, 0.0),
        vec3(-0.5, -0.5, 0.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4(vertex.position, 1.0);
    return out;
}