#version 450

struct Vertex {
	float vx;
    float vy;
    float vz;
	uint n;
	float tu;
    float tv;
};

layout(binding = 0) readonly buffer Vertices {
    Vertex vertices[];
};

layout(location = 0) out vec3 color;

void main() {
	Vertex v = vertices[gl_VertexIndex];

	float nx = float((v.n >> 0) & 0xFF);
	float ny = float((v.n >> 8) & 0xFF);
	float nz = float((v.n >> 16) & 0xFF);

	vec3 position = vec3(v.vx, v.vy, v.vz);
	vec3 normal = vec3(nx, ny, nz) / 127.0 - 1.0;
	vec2 texcoord = vec2(v.tu, v.tv);

	gl_Position = vec4(position + vec3(0.25, -0.75, 0.5), 1.0);

	color = vec3(normal * 0.5 + vec3(0.5));
}