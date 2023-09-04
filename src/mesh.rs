use std::time::Instant;

use ash::vk;
use bevy::prelude::*;
use bytemuck::cast_slice;
use meshopt::VertexDataAdapter;

use crate::instance::{Buffer, CendreInstance};

const MAX_TRIANGLE_COUNT: usize = 126;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub norm: [u8; 4], // u32
    pub uv: [u16; 2],
}

unsafe impl bytemuck::Pod for Vertex {}

#[derive(Component, Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub fn optimize_mesh(vertices: &[Vertex], indices: &[u32]) -> Mesh {
    info!("Triangles: {}", indices.len() / 3);

    let (vertex_count, remap) = meshopt::generate_vertex_remap(vertices, Some(indices));

    let vertices = meshopt::remap_vertex_buffer(vertices, vertex_count, &remap);
    let indices = meshopt::remap_index_buffer(Some(indices), vertex_count, &remap);

    let mut indices = meshopt::optimize_vertex_cache(&indices, vertices.len());
    let vertices = meshopt::optimize_vertex_fetch(&mut indices, &vertices);

    Mesh { vertices, indices }
}

#[derive(Copy, Clone)]
pub struct Meshlet {
    pub cone: [f32; 4],
    pub vertices: [u32; 64],
    pub indices: [[u8; 3]; MAX_TRIANGLE_COUNT],
    pub triangle_count: u8,
    pub vertex_count: u8,
}

impl Default for Meshlet {
    fn default() -> Self {
        Self {
            cone: [0.0, 0.0, 0.0, 0.0],
            vertices: [0; 64],
            indices: [[0, 0, 0]; MAX_TRIANGLE_COUNT],
            triangle_count: 0,
            vertex_count: 0,
        }
    }
}

impl Meshlet {
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut data = vec![];
        data.extend_from_slice(cast_slice(&self.cone));
        data.extend_from_slice(cast_slice(&self.vertices));
        data.extend_from_slice(cast_slice(&self.indices));
        data.push(self.triangle_count);
        data.push(self.vertex_count);
        data
    }
}

#[allow(clippy::identity_op)]
fn build_meshlets(vertices: &[Vertex], indices: &[u32]) -> Vec<Meshlet> {
    let max_vertices = 64;
    let max_triangles = 126;
    let mut meshopt_meshlets =
        meshopt::build_meshlets(indices, vertices.len(), max_vertices, max_triangles);

    // TODO this isn't necessary, but it makes it so we can assume to always
    // have 32 meshlets in the task shader
    while meshopt_meshlets.len() % 32 != 0 {
        meshopt_meshlets.push(meshopt::Meshlet {
            vertices: [0; 64],
            indices: [[0, 0, 0]; 126],
            triangle_count: 0,
            vertex_count: 0,
        });
    }

    let mut meshlets = Vec::with_capacity(meshopt_meshlets.len());
    for meshlet in &meshopt_meshlets {
        let vertices =
            VertexDataAdapter::new(cast_slice(vertices), std::mem::size_of::<Vertex>(), 0)
                .expect("Failed to create VertexDataAdapter from vertices");
        let bounds = meshopt::compute_meshlet_bounds(meshlet, &vertices);
        let cone = [
            bounds.cone_axis[0],
            bounds.cone_axis[1],
            bounds.cone_axis[2],
            bounds.cone_cutoff,
        ];
        meshlets.push(Meshlet {
            cone,
            vertices: meshlet.vertices,
            indices: meshlet.indices,
            triangle_count: meshlet.triangle_count,
            vertex_count: meshlet.vertex_count,
        });
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

#[derive(Component)]
pub struct PreparedMesh;

pub fn prepare_mesh(
    mut commands: Commands,
    mut cendre: ResMut<CendreInstance>,
    meshes: Query<(Entity, &Mesh), Without<PreparedMesh>>,
) {
    for (entity, mesh) in &meshes {
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

        if cendre.rtx_supported {
            // TODO build meshlets on load
            let meshlets = build_meshlets(&mesh.vertices, &mesh.indices);
            info!("Meshlets: {}", meshlets.len());

            let mut culled = 0;
            for meshlet in &meshlets {
                if meshlet.cone[2] > meshlet.cone[3] {
                    culled += 1;
                }
            }
            println!(
                "Cullled meshlets: {culled}/{} {:.2}%",
                meshlets.len(),
                (culled as f32 / meshlets.len() as f32) * 100.0
            );

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

        entity_cmd.insert(PreparedMesh);
        info!("mesh prepared in {}ms", start.elapsed().as_millis());
    }
}
