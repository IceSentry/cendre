use std::time::Instant;

use ash::vk;
use bevy::prelude::*;
use bytemuck::cast_slice;
use meshopt::VertexDataAdapter;

use crate::instance::{Buffer, CendreInstance};

const MAX_TRIANGLE_COUNT: usize = 124;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub norm: [u8; 4], // u32
    pub uv: [u16; 2],
}
unsafe impl bytemuck::Pod for Vertex {}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, Debug)]
pub struct MeshDraw {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
}
unsafe impl bytemuck::Pod for MeshDraw {}

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

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Meshlet {
    pub cone: [f32; 4],
    // data_offset..data_offset + vertex_count - 1 stores vertex indices,
    // we store indices packed un 4bytes units after that
    pub data_offset: u32,
    pub vertex_count: u8,
    pub triangle_count: u8,
}
unsafe impl bytemuck::Zeroable for Meshlet {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
unsafe impl bytemuck::Pod for Meshlet {}

#[allow(clippy::identity_op)]
fn build_meshlets(vertices: &[Vertex], indices: &[u32]) -> (Vec<Meshlet>, Vec<u32>) {
    let max_vertices = 64;
    let mut meshopt_meshlets =
        meshopt::build_meshlets(indices, vertices.len(), max_vertices, MAX_TRIANGLE_COUNT);

    // TODO this isn't necessary, but it makes it so we can assume to always
    // have 32 meshlets in the task shader
    let empty_meshlet = unsafe { std::mem::zeroed::<meshopt::Meshlet>() };
    while meshopt_meshlets.len() % 32 != 0 {
        meshopt_meshlets.push(empty_meshlet);
    }

    let mut meshlets = Vec::with_capacity(meshopt_meshlets.len());
    let mut meshlet_data = Vec::new();
    for meshlet in &meshopt_meshlets {
        let data_offset = meshlet_data.len() as u32;
        for i in 0..meshlet.vertex_count as usize {
            meshlet_data.push(meshlet.vertices[i]);
        }

        let index_group_count = (meshlet.triangle_count as usize * 3 + 3) / 4;
        let flat_indices = meshlet.indices.iter().flatten().collect::<Vec<_>>();
        for chunk in flat_indices.chunks(4).take(index_group_count) {
            let [a, b, c, d] = *chunk else {
                // If this actually happens, maybe consider padding
                panic!("invalid chunk for indices");
            };
            meshlet_data.push(u32::from_ne_bytes([*a, *b, *c, *d]));
        }

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
            data_offset,
            vertex_count: meshlet.vertex_count,
            triangle_count: meshlet.triangle_count,
        });
    }

    (meshlets, meshlet_data)
}

#[derive(Component, Deref)]
pub struct VertexBuffer(pub Buffer);
#[derive(Component, Deref)]
pub struct IndexBuffer(pub Buffer);
#[derive(Component, Deref)]
pub struct MeshletBuffer(pub Buffer);
#[derive(Component, Deref)]
pub struct MeshletDataBuffer(pub Buffer);

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
        cendre.upload_buffer(
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
        cendre.upload_buffer(
            cendre.command_buffers[0],
            &mut scratch_buffer,
            &index_buffer,
            index_buffer_data,
        );
        entity_cmd.insert(IndexBuffer(index_buffer));

        if cendre.mesh_shader_supported {
            // TODO build meshlets on load
            let (meshlets, meshlet_data) = build_meshlets(&mesh.vertices, &mesh.indices);
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

            let mut data = Vec::with_capacity(meshlets.len() * std::mem::size_of::<Meshlet>());
            for m in &meshlets {
                data.extend_from_slice(bytemuck::bytes_of(m));
            }

            let meshlet_buffer = cendre
                .create_buffer(
                    128 * 1024 * 1024,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::GpuOnly,
                )
                .unwrap();
            let meshlet_data_buffer = cendre
                .create_buffer(
                    128 * 1024 * 1024,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::GpuOnly,
                )
                .unwrap();

            cendre.upload_buffer(
                cendre.command_buffers[0],
                &mut scratch_buffer,
                &meshlet_buffer,
                &data,
            );
            cendre.upload_buffer(
                cendre.command_buffers[0],
                &mut scratch_buffer,
                &meshlet_data_buffer,
                &meshlet_data
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>(),
            );

            entity_cmd.insert((
                MeshletBuffer(meshlet_buffer),
                MeshletsCount(meshlets.len() as u32),
                MeshletDataBuffer(meshlet_data_buffer),
            ));
        }

        entity_cmd.insert(PreparedMesh);
        info!("mesh prepared in {}ms", start.elapsed().as_millis());
    }
}
