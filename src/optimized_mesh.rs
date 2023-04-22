use std::time::Instant;

use ash::vk;
use bevy::{
    prelude::*,
    render::mesh::{Indices, VertexAttributeValues},
};
use bytemuck::cast_slice;

use crate::{
    instance::{Buffer, CendreInstance},
    RTXEnabled,
};

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub norm: [u8; 4], // u32
    pub uv: [f32; 2],
}

#[derive(Component, Debug, Clone)]
pub struct OptimizedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub prepared: bool,
}

pub fn optimize_mesh(mesh: Mesh) -> OptimizedMesh {
    let Some(Indices::U32(indices)) = mesh.indices().as_ref() else {
        unimplemented!("Mesh require indices");
    };
    info!("Triangles: {}", indices.len() / 3);

    let vertices = {
        let pos = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .unwrap();
        let norms: Vec<_> = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .map(|n| {
                n.iter()
                    .map(|n| {
                        [
                            (n[0] * 127.0 + 127.0) as u8,
                            (n[1] * 127.0 + 127.0) as u8,
                            (n[2] * 127.0 + 127.0) as u8,
                            0,
                        ]
                    })
                    .collect()
            })
            .unwrap();

        let uvs = mesh
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(as_float2)
            .map(<[[f32; 2]]>::to_vec)
            .unwrap_or(vec![[0.0, 0.0]; pos.len()]);

        let mut vertices = vec![];
        for (pos, (norm, uv)) in pos.iter().zip(norms.iter().zip(uvs.iter())) {
            vertices.push(Vertex {
                pos: *pos,
                norm: *norm,
                uv: *uv,
            });
        }
        vertices
    };

    let (vertex_count, remap) = meshopt::generate_vertex_remap(&vertices, Some(indices));

    let vertices = meshopt::remap_vertex_buffer(&vertices, vertex_count, &remap);
    let indices = meshopt::remap_index_buffer(Some(indices), vertex_count, &remap);

    let mut indices = meshopt::optimize_vertex_cache(&indices, vertices.len());
    let vertices = meshopt::optimize_vertex_fetch(&mut indices, &vertices);

    OptimizedMesh {
        vertices,
        indices,
        prepared: false,
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Meshlet {
    pub vertices: [u32; 64],
    pub indices: [u8; 126], // up to 42 triangles
    pub index_count: u8,
    pub vertex_count: u8,
}

impl Default for Meshlet {
    fn default() -> Self {
        Self {
            vertices: [0; 64],
            indices: [0; 126],
            index_count: 0,
            vertex_count: 0,
        }
    }
}

impl Meshlet {
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut data = vec![];
        data.extend_from_slice(cast_slice(&self.vertices));
        data.extend_from_slice(cast_slice(&self.indices));
        data.push(self.index_count);
        data.push(self.vertex_count);
        data
    }
}

fn as_float2(val: &VertexAttributeValues) -> Option<&[[f32; 2]]> {
    match val {
        VertexAttributeValues::Float32x2(values) => Some(values),
        _ => None,
    }
}

#[allow(clippy::identity_op)]
fn build_meshlets(vertices: &[Vertex], indices: &[u32]) -> Vec<Meshlet> {
    let mut meshlets = vec![];
    let mut meshlet = Meshlet::default();
    let mut meshlet_vertices = vec![0xFF; vertices.len()];

    for chunk in indices.chunks(3) {
        let [a, b, c] = chunk else { unreachable!() };
        if meshlet.vertex_count
            + u8::from(meshlet_vertices[*a as usize] == 0xFF)
            + u8::from(meshlet_vertices[*b as usize] == 0xFF)
            + u8::from(meshlet_vertices[*c as usize] == 0xFF)
            > 64
            || meshlet.index_count + 3 > 126
        {
            meshlets.push(meshlet);
            for i in 0..meshlet.vertex_count {
                let v = meshlet.vertices[i as usize];
                meshlet_vertices[v as usize] = 0xFF;
            }
            meshlet = Meshlet::default();
        }

        if meshlet_vertices[*a as usize] == 0xFF {
            meshlet_vertices[*a as usize] = meshlet.vertex_count;
            meshlet.vertices[meshlet.vertex_count as usize] = *a;
            meshlet.vertex_count += 1;
        }
        if meshlet_vertices[*b as usize] == 0xFF {
            meshlet_vertices[*b as usize] = meshlet.vertex_count;
            meshlet.vertices[meshlet.vertex_count as usize] = *b;
            meshlet.vertex_count += 1;
        }
        if meshlet_vertices[*c as usize] == 0xFF {
            meshlet_vertices[*c as usize] = meshlet.vertex_count;
            meshlet.vertices[meshlet.vertex_count as usize] = *c;
            meshlet.vertex_count += 1;
        }

        meshlet.indices[meshlet.index_count as usize] = meshlet_vertices[*a as usize];
        meshlet.index_count += 1;
        meshlet.indices[meshlet.index_count as usize] = meshlet_vertices[*b as usize];
        meshlet.index_count += 1;
        meshlet.indices[meshlet.index_count as usize] = meshlet_vertices[*c as usize];
        meshlet.index_count += 1;
    }

    if meshlet.index_count > 1 {
        meshlets.push(meshlet);
    }

    meshlets
}

#[derive(Component, Deref)]
pub struct VertexBuffer(pub Buffer);
#[derive(Component, Deref)]
pub struct IndexBuffer(pub Buffer);
#[derive(Component, Deref)]
pub struct MeshletBuffer(pub Buffer);

#[derive(Component)]
pub struct MeshletsCount(pub u32);

// TODO async
pub fn prepare_mesh(
    mut commands: Commands,
    mut cendre: ResMut<CendreInstance>,
    mut meshes: Query<(Entity, &mut OptimizedMesh)>,
    rtx_enabled: Res<RTXEnabled>,
) {
    for (entity, mut mesh) in &mut meshes {
        if !mesh.prepared {
            info!("preparing mesh");
            let start = Instant::now();

            let vertex_buffer_data = cast_slice(&mesh.vertices);
            let index_buffer_data = cast_slice(&mesh.indices);

            let mut entity_cmd = commands.entity(entity);

            let mut scratch_buffer = cendre
                .create_buffer(
                    128 * 1024 * 1024,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )
                .unwrap();

            let vertex_buffer = cendre
                .create_buffer(
                    128 * 1024 * 1024,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::GpuOnly,
                )
                .unwrap();
            cendre.updload_buffer(
                cendre.command_buffers[0],
                &mut scratch_buffer,
                &vertex_buffer,
                vertex_buffer_data,
            );
            entity_cmd.insert(VertexBuffer(vertex_buffer));

            let index_buffer = cendre
                .create_buffer(
                    128 * 1024 * 1024,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                    gpu_allocator::MemoryLocation::GpuOnly,
                )
                .unwrap();
            cendre.updload_buffer(
                cendre.command_buffers[0],
                &mut scratch_buffer,
                &index_buffer,
                index_buffer_data,
            );
            entity_cmd.insert(IndexBuffer(index_buffer));

            if rtx_enabled.0 {
                let meshlets = build_meshlets(&mesh.vertices, &mesh.indices);
                info!("Meshlets: {}", meshlets.len());
                let data = meshlets.iter().flat_map(Meshlet::bytes).collect::<Vec<_>>();

                let meshlet_buffer = cendre
                    .create_buffer(
                        128 * 1024 * 1024,
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                        gpu_allocator::MemoryLocation::GpuOnly,
                    )
                    .unwrap();
                cendre.updload_buffer(
                    cendre.command_buffers[0],
                    &mut scratch_buffer,
                    &meshlet_buffer,
                    &data,
                );
                entity_cmd.insert((
                    MeshletBuffer(meshlet_buffer),
                    MeshletsCount(meshlets.len() as u32),
                ));
            }

            mesh.prepared = true;
            info!("mesh prepared in {}ms", start.elapsed().as_millis());
        }
    }
}
