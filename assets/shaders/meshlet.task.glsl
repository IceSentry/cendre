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

void main() {
	// meshlet group index
	uint mgi = gl_WorkGroupID.x;
	// task index
	uint ti = gl_LocalInvocationID.x;
	// meshlet index
	uint mi = mgi * 32 + ti;

	meshlet_indices[ti] = mi;

	if (ti == 0) {
		gl_TaskCountNV = 32;
	}
}