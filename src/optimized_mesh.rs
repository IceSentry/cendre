use std::time::Instant;

use ash::vk;
use bevy::{
    prelude::*,
    render::mesh::{Indices, VertexAttributeValues},
};
use bytemuck::cast_slice;

use crate::{
    instance::{Buffer, CendreInstance},
    RTX,
};

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub norm: [u8; 4],
    pub uv: [f32; 2],
}

impl Vertex {
    fn bytes(&self) -> Vec<u8> {
        let mut data = vec![];
        data.extend_from_slice(cast_slice(&self.pos));
        data.extend_from_slice(&self.norm);
        data.extend_from_slice(cast_slice(&self.uv));
        data
    }
}

#[derive(Clone, Copy)]
pub struct Meshlet {
    pub vertices: [u32; 64],
    pub indices: [u8; 126], // up to 42 triangles
    pub triangle_count: u8,
    pub vertex_count: u8,
}

impl Default for Meshlet {
    fn default() -> Self {
        Self {
            vertices: [0; 64],
            indices: [0; 126],
            triangle_count: 0,
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
        data.push(self.triangle_count);
        data.push(self.vertex_count);
        data
    }
}

#[derive(Component)]
pub struct OptimizedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub prepared: bool,
}

impl OptimizedMesh {
    pub fn from_bevy_mesh(mesh: &Mesh) -> OptimizedMesh {
        // Meshopt version
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
                .unwrap();

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

        let indices = mesh.indices().map(|indices| match indices {
            Indices::U32(indices) => indices.clone(),
            Indices::U16(_) => panic!("only u32 indices are supported"),
        });

        // let (vertex_count, remap) = meshopt::generate_vertex_remap(&vertices, indices);

        // let vertex_buffer = meshopt::remap_vertex_buffer(&vertices, vertex_count, &remap)
        //     .iter()
        //     .flat_map(Vertex::bytes)
        //     .collect();
        // let index_buffer = meshopt::remap_index_buffer(indices, vertex_count, &remap)
        //     .iter()
        //     .flat_map(|x| x.to_ne_bytes())
        //     .collect();

        // Bevy version
        // let vertex_buffer = mesh.get_vertex_buffer_data();
        // let index_buffer = mesh.get_index_buffer_bytes().unwrap().to_vec();

        Self {
            vertices,
            indices,
            prepared: false,
        }
    }
}

fn as_float2(val: &VertexAttributeValues) -> Option<&[[f32; 2]]> {
    match val {
        VertexAttributeValues::Float32x2(values) => Some(values),
        _ => None,
    }
}

#[allow(clippy::identity_op)]
fn build_meshlets(mesh: &OptimizedMesh) -> Vec<Meshlet> {
    let mut meshlets = vec![];
    let mut meshlet = Meshlet::default();
    let meshlet_vertices = vec![0xFF; mesh.vertices.len()];

    let Some(indices) = mesh.indices.as_ref() else {
        panic!("Meshlets require indices");
    };

    for chunk in indices.chunks(3) {
        let [a, b, c] = chunk else { unreachable!() };
        let mut av = meshlet_vertices[*a as usize];
        let mut bv = meshlet_vertices[*b as usize];
        let mut cv = meshlet_vertices[*c as usize];

        if meshlet.vertex_count + u8::from(av == 0xFF) + u8::from(bv == 0xFF) + u8::from(cv == 0xFF)
            > 64
        {
            meshlets.push(meshlet);
            meshlet = Meshlet::default();
        }
        if meshlet.triangle_count + 1 > 126 / 3 {
            meshlets.push(meshlet);
            meshlet = Meshlet::default();
        }

        if av == 0xFF {
            av = meshlet.vertex_count;
            meshlet.vertex_count += 1;
            meshlet.vertices[meshlet.vertex_count as usize] = *a;
        }
        if bv == 0xFF {
            bv = meshlet.vertex_count;
            meshlet.vertex_count += 1;
            meshlet.vertices[meshlet.vertex_count as usize] = *b;
        }
        if cv == 0xFF {
            cv = meshlet.vertex_count;
            meshlet.vertex_count += 1;
            meshlet.vertices[meshlet.vertex_count as usize] = *c;
        }

        meshlet.indices[(meshlet.triangle_count * 3 + 0) as usize] = av;
        meshlet.indices[(meshlet.triangle_count * 3 + 1) as usize] = bv;
        meshlet.indices[(meshlet.triangle_count * 3 + 2) as usize] = cv;
        meshlet.triangle_count += 1;
    }
    if meshlet.triangle_count > 0 {
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

#[derive(Resource)]
pub struct MeshletsSize(pub u32);

pub(crate) fn prepare_mesh(
    mut commands: Commands,
    mut cendre: ResMut<CendreInstance>,
    mut meshes: Query<(Entity, &mut OptimizedMesh)>,
) {
    for (entity, mut mesh) in &mut meshes {
        if !mesh.prepared {
            info!("preparing mesh");
            let start = Instant::now();

            let (vertex_count, remap) = if let Some(indices) = mesh.indices.clone() {
                meshopt::generate_vertex_remap(&mesh.vertices, Some(&indices))
            } else {
                meshopt::generate_vertex_remap(&mesh.vertices, None)
            };

            let vertex_buffer_data =
                meshopt::remap_vertex_buffer(&mesh.vertices, vertex_count, &remap)
                    .iter()
                    .flat_map(Vertex::bytes)
                    .collect::<Vec<_>>();
            let index_buffer_data = if let Some(indices) = mesh.indices.clone() {
                meshopt::remap_index_buffer(Some(&indices), vertex_count, &remap)
            } else {
                meshopt::remap_index_buffer(None, vertex_count, &remap)
            }
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>();

            let mut entity_cmd = commands.entity(entity);

            let mut vertex_buffer = cendre
                .create_buffer(
                    128 * 1024 * 1024,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )
                .unwrap();
            vertex_buffer.write(&vertex_buffer_data);
            entity_cmd.insert(VertexBuffer(vertex_buffer));

            let mut index_buffer = cendre
                .create_buffer(
                    128 * 1024 * 1024,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )
                .unwrap();
            index_buffer.write(&index_buffer_data);
            entity_cmd.insert(IndexBuffer(index_buffer));

            if RTX {
                let meshlets = build_meshlets(&mesh);
                let data = meshlets.iter().flat_map(Meshlet::bytes).collect::<Vec<_>>();

                let mut meshlet_buffer = cendre
                    .create_buffer(
                        128 * 1024 * 1024,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        gpu_allocator::MemoryLocation::CpuToGpu,
                    )
                    .unwrap();
                meshlet_buffer.write(&data);
                entity_cmd.insert(MeshletBuffer(meshlet_buffer));

                commands.insert_resource(MeshletsSize(meshlets.len() as u32));
            }

            info!("mesh prepared in {}ms", start.elapsed().as_millis());
            mesh.prepared = true;
        }
    }
}
