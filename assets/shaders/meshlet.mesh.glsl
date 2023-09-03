#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_NV_mesh_shader: require
#extension GL_GOOGLE_include_directive: require

#include "mesh.h"

#define DEBUG 1
#define CULL 1

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 126) out;

layout(binding = 0) readonly buffer Vertices {
	Vertex vertices[];
};

layout(binding = 1) readonly buffer Meshlets {
	Meshlet meshlets[];
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

#if CULL
	vec4 cone = vec4(
		meshlets[mi].cone[0],
		meshlets[mi].cone[1],
		meshlets[mi].cone[2],
		meshlets[mi].cone[3]
	);
	if (cone_cull(cone, vec3(0, 0, 1))) {
		if (ti == 0) {
			gl_PrimitiveCountNV = 0;
		}
		return;
	}
#endif // CULL

#if DEBUG
	uint mhash = hash(mi);
	vec3 mcolor = vec3(
		float(mhash & 255),
		float((mhash >> 8) & 255),
		float((mhash >> 16) & 255)
	) / 255.0;
#endif

	uint vertexCount = uint(meshlets[mi].vertexCount);
	for (uint i = ti; i < vertexCount; i += 32)
	{
		uint vi = meshlets[mi].vertices[i]; // vertex index

		vec3 position = vec3(vertices[vi].vx, vertices[vi].vy, vertices[vi].vz);
		vec3 normal = vec3(
			int(vertices[vi].nx),
			int(vertices[vi].ny),
			int(vertices[vi].nz)
		) / 127.0 - 1.0;
		vec2 texcoord = vec2(vertices[vi].tu, vertices[vi].tv);

		vec3 offset = vec3(0.25, -0.75, 0.5);
		vec3 scale = vec3(1.0, 1.0, 0.5);

		gl_MeshVerticesNV[i].gl_Position = vec4(position * scale + offset, 1.0);
		color[i] = vec4(normal * 0.5 + vec3(0.5), 1.0);
#if DEBUG
		color[i] = vec4(mcolor, 1.0);
#endif
	}

	uint indexCount = uint(meshlets[mi].triangleCount) * 3;
	for (uint i = ti; i < indexCount; i += 32)
	{
		gl_PrimitiveIndicesNV[i] = uint(meshlets[mi].indices[i]);
	}

	if (ti == 0) {
    	gl_PrimitiveCountNV = uint(meshlets[mi].triangleCount);
	}
}