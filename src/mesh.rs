use std::time::Instant;

use ash::vk;
use bevy::prelude::*;
use bytemuck::cast_slice;

use crate::instance::{Buffer, CendreInstance};

const MAX_TRIANGLE_COUNT: usize = 126;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, Debug)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub norm: [u8; 4], // u32
    pub uv: [f32; 2],
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
    pub indices: [u8; MAX_TRIANGLE_COUNT * 3],
    pub triangle_count: u8,
    pub vertex_count: u8,
}

impl Default for Meshlet {
    fn default() -> Self {
        Self {
            cone: [0.0, 0.0, 0.0, 0.0],
            vertices: [0; 64],
            indices: [0; MAX_TRIANGLE_COUNT * 3],
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
            || meshlet.triangle_count >= MAX_TRIANGLE_COUNT as u8
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

        let index = meshlet.triangle_count as usize * 3;
        meshlet.indices[index + 0] = meshlet_vertices[*a as usize];
        meshlet.indices[index + 1] = meshlet_vertices[*b as usize];
        meshlet.indices[index + 2] = meshlet_vertices[*c as usize];
        meshlet.triangle_count += 1;
    }

    if meshlet.triangle_count > 1 {
        meshlets.push(meshlet);
    }

    meshlets
}

fn build_meshlet_cone(mesh: &Mesh, meshlets: &mut Vec<Meshlet>) {
    for meshlet in meshlets {
        let mut normals = vec![Vec3::ZERO; meshlet.triangle_count as usize];

        for i in 0..meshlet.triangle_count {
            let i = i as usize;
            let a = meshlet.indices[i * 3 + 0] as usize;
            let b = meshlet.indices[i * 3 + 1] as usize;
            let c = meshlet.indices[i * 3 + 2] as usize;

            let va = mesh.vertices[meshlet.vertices[a] as usize];
            let vb = mesh.vertices[meshlet.vertices[b] as usize];
            let vc = mesh.vertices[meshlet.vertices[c] as usize];

            let p0 = Vec3::new(va.pos[0], va.pos[1], va.pos[2]);
            let p1 = Vec3::new(vb.pos[0], vb.pos[1], vb.pos[2]);
            let p2 = Vec3::new(vc.pos[0], vc.pos[1], vc.pos[2]);

            let p10 = p1 - p0;
            let p20 = p2 - p0;

            let normal = p10.cross(p20);

            normals[i] = normal * normal.length_recip();
        }

        let mut avgnormal = Vec3::ZERO;
        for n in &normals {
            avgnormal += *n;
        }
        if avgnormal.length() == 0.0 {
            avgnormal.x = 1.0;
            avgnormal.y = 0.0;
            avgnormal.z = 0.0;
        } else {
            avgnormal /= avgnormal.length();
        }

        let mut mindp: f32 = 1.0;
        assert_eq!(normals.len(), meshlet.triangle_count as usize);
        for n in normals {
            let dp = n.dot(avgnormal);
            mindp = mindp.min(dp);
        }

        let conew = if mindp <= 0.0 {
            1.0
        } else {
            (1.0 - mindp * mindp).sqrt()
        };
        meshlet.cone[0] = avgnormal[0];
        meshlet.cone[1] = avgnormal[1];
        meshlet.cone[2] = avgnormal[2];
        meshlet.cone[3] = conew;
    }
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
            let mut meshlets = build_meshlets(&mesh.vertices, &mesh.indices);
            info!("Meshlets: {}", meshlets.len());

            build_meshlet_cone(mesh, &mut meshlets);

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
