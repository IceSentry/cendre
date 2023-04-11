use bevy::{
    prelude::*,
    render::mesh::{Indices, VertexAttributeValues},
};
use bytemuck::cast_slice;

#[repr(C)]
#[derive(Clone, Default)]
struct Vertex {
    pos: [f32; 3],
    norm: [u8; 4],
    uv: [f32; 2],
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

struct Meshlet {
    vertices: [u32; 64],
    indices: [u8; 126], // up to 42 triangles
    triangle_count: u8,
    vertex_count: u8,
}

#[derive(Component)]
pub struct OptimizedMesh {
    pub vertex_buffer: Vec<u8>,
    pub index_buffer: Vec<u8>,
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
            Indices::U32(indices) => indices.as_slice(),
            Indices::U16(_) => panic!("only u32 indices are supported"),
        });
        let (vertex_count, remap) = meshopt::generate_vertex_remap(&vertices, indices);

        let vertex_buffer = meshopt::remap_vertex_buffer(&vertices, vertex_count, &remap)
            .iter()
            .flat_map(Vertex::bytes)
            .collect();
        let index_buffer = meshopt::remap_index_buffer(indices, vertex_count, &remap)
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect();

        // Bevy version
        // let vertex_buffer = mesh.get_vertex_buffer_data();
        // let index_buffer = mesh.get_index_buffer_bytes().unwrap().to_vec();

        Self {
            vertex_buffer,
            index_buffer,
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
