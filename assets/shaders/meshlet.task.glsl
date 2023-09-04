#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_NV_mesh_shader: require
#extension GL_GOOGLE_include_directive: require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_KHR_shader_subgroup_ballot: require

#include "mesh.h"

#define CULL 1

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 1) readonly buffer Meshlets {
	Meshlet meshlets[];
};

out taskNV block {
	uint meshlet_indices[32];
};

bool cone_cull(vec4 cone, vec3 view) {
	return dot(cone.xyz, view) > cone.w;
}

shared uint meshlet_count;

void main() {
	// meshlet group index
	uint mgi = gl_WorkGroupID.x;
	// task index
	uint ti = gl_LocalInvocationID.x;
	// meshlet index
	uint mi = mgi * 32 + ti;

#if CULL
	vec4 cone = vec4(
		meshlets[mi].cone[0],
		meshlets[mi].cone[1],
		meshlets[mi].cone[2],
		meshlets[mi].cone[3]
	);
	bool accept = !cone_cull(cone, vec3(0, 0, 1));
	uvec4 ballot = subgroupBallot(accept);
	uint index = subgroupBallotExclusiveBitCount(ballot);

	if (accept) {
		meshlet_indices[index] = mi;
	}

	uint count = subgroupBallotBitCount(ballot);
	if (ti == 0) {
		gl_TaskCountNV = count;
	}
#else
	meshlet_indices[ti] = mi;
	if (ti == 0) {
		gl_TaskCountNV = 32;
	}
#endif
}