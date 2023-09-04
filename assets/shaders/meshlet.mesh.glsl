#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_NV_mesh_shader: require
#extension GL_GOOGLE_include_directive: require

#include "mesh.h"

#define DEBUG 1

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

layout(binding = 0) readonly buffer Vertices {
	Vertex vertices[];
};

layout(binding = 1) readonly buffer Meshlets {
	Meshlet meshlets[];
};

layout(binding = 2) readonly buffer MeshletData {
	uint meshlet_data[];
};

in taskNV block {
	uint meshlet_indices[32];
};

layout(location = 0) out vec4 color[];

uint hash(uint a) {
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

bool cone_cull(vec4 cone, vec3 view) {
	return dot(cone.xyz, view) > cone.w;
}

void main() {
	// mesh index
	uint mi = meshlet_indices[gl_WorkGroupID.x];
	// thread index
	uint ti = gl_LocalInvocationID.x;

#if DEBUG
	uint mhash = hash(mi);
	vec3 mcolor = vec3(
		float(mhash & 255),
		float((mhash >> 8) & 255),
		float((mhash >> 16) & 255)
	) / 255.0;
#endif

	uint vertex_count = uint(meshlets[mi].vertex_count);
	uint vertex_offset = meshlets[mi].data_offset;
	uint index_offset = vertex_offset + vertex_count;

	for (uint i = ti; i < vertex_count; i += 32) {
		uint vi = meshlet_data[vertex_offset + i]; // vertex index

		vec3 position = vec3(vertices[vi].vx, vertices[vi].vy, vertices[vi].vz);
		vec3 normal = vec3(
			int(vertices[vi].nx),
			int(vertices[vi].ny),
			int(vertices[vi].nz)
		) / 127.0 - 1.0;
		vec2 texcoord = vec2(vertices[vi].tu, vertices[vi].tv);

		vec3 offset = vec3(0.25, -0.75, 0.5);
		// vec3 offset = vec3(0, 0, 0.5);
		vec3 scale = vec3(1.0, 1.0, 0.5);
		// vec3 scale = vec3(1, 1, 0.5);

		gl_MeshVerticesNV[i].gl_Position = vec4(position * scale + offset, 1.0);
#if DEBUG
		color[i] = vec4(mcolor, 1.0);
#else
		color[i] = vec4(normal * 0.5 + vec3(0.5), 1.0);
#endif
	}

	uint index_count = uint(meshlets[mi].triangle_count) * 3;
	uint index_group_count = (index_count + 3) / 4;
	for (uint i = ti; i < index_group_count; i += 32) {
		writePackedPrimitiveIndices4x8NV(i * 4, meshlet_data[index_offset + i]);
	}

	if (ti == 0) {
    	gl_PrimitiveCountNV = uint(meshlets[mi].triangle_count);
	}
}