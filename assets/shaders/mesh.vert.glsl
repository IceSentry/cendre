#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require

#include "mesh.h"

layout(push_constant) uniform block {
	MeshDraw mesh_draw;
};

layout(binding = 0) readonly buffer Vertices {
	Vertex vertices[];
};

layout(location = 0) out vec4 color;

void main() {
	vec3 position = vec3(
        vertices[gl_VertexIndex].vx,
        vertices[gl_VertexIndex].vy,
        vertices[gl_VertexIndex].vz
    );
	vec3 normal = vec3(
        int(vertices[gl_VertexIndex].nx),
        int(vertices[gl_VertexIndex].ny),
        int(vertices[gl_VertexIndex].nz)
    );
    normal = normal / 127.0 - 1.0;
	vec2 texcoord = vec2(vertices[gl_VertexIndex].tu, vertices[gl_VertexIndex].tv);

    vec3 offset = vec3(mesh_draw.offset[0], mesh_draw.offset[1], 0.0);
    vec3 scale = vec3(mesh_draw.scale[0], mesh_draw.scale[1], 1.0);

    gl_Position = vec4(
        position * scale + offset * vec3(2.0, 2.0, 0.5) + vec3(-1.0, -1.0, 0.5),
        1.0
    );

	color = vec4(normal * 0.5 + vec3(0.5), 1.0);
}