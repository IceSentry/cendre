use anyhow::Context;
use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    prelude::*,
    reflect::TypeUuid,
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_resource::PrimitiveTopology,
    },
    utils::BoxedFuture,
};
use bytemuck::cast_slice;
use std::{
    io::{BufReader, Cursor},
    time::Instant,
};

#[derive(Debug, TypeUuid)]
#[uuid = "39cadc56-aa9c-4543-8640-a018b74b5052"]
pub struct LoadedObj {
    pub meshes: Vec<Mesh>,
}

pub struct ObjLoaderPlugin;
impl Plugin for ObjLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<LoadedObj>()
            .init_asset_loader::<ObjLoader>()
            .add_system(optimize_mesh);
    }
}

#[derive(Default)]
pub struct ObjLoader;
impl AssetLoader for ObjLoader {
    fn extensions(&self) -> &[&str] {
        &["obj"]
    }

    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let start = Instant::now();

            info!("Loading {:?}", load_context.path());

            let obj = load_obj(bytes, load_context).await?;
            load_context.set_default_asset(LoadedAsset::new(obj));

            info!(
                "Finished loading {:?} {}ms",
                load_context.path(),
                start.elapsed().as_millis(),
            );

            Ok(())
        })
    }
}

pub async fn load_obj<'a, 'b>(
    bytes: &'a [u8],
    load_context: &'a LoadContext<'b>,
) -> anyhow::Result<LoadedObj> {
    let (models, _materials) = tobj::load_obj_buf_async(
        &mut BufReader::new(Cursor::new(bytes)),
        &tobj::GPU_LOAD_OPTIONS,
        |mtl_path| async move {
            let path = load_context.path().parent().unwrap().join(mtl_path);
            let mtl_bytes = load_context.read_asset_bytes(&path).await.unwrap();
            let mtl = tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mtl_bytes)));
            info!("Finished loading {path:?}");
            mtl
        },
    )
    .await
    .with_context(|| format!("Failed to load obj {:?}", load_context.path()))?;

    let meshes = models.iter().map(generate_mesh).collect();

    Ok(LoadedObj { meshes })
}

fn generate_mesh(model: &tobj::Model) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    if !model.mesh.positions.is_empty() {
        println!("vertex attributes positons found");
        let mut positions = vec![];
        for verts in model.mesh.positions.chunks_exact(3) {
            let [v0, v1, v2] = verts else { unreachable!(); };
            positions.push([*v0, *v1, *v2]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    }

    if !model.mesh.normals.is_empty() {
        println!("vertex attributes normals found");
        let mut normals = vec![];
        for n in model.mesh.normals.chunks_exact(3) {
            let [n0, n1, n2] = n else { unreachable!(); };
            normals.push([*n0, *n1, *n2]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    }

    if !model.mesh.texcoords.is_empty() {
        println!("vertex attributes uvs found");
        let mut uvs = vec![];
        for uv in model.mesh.texcoords.chunks_exact(2) {
            let [u, v] = uv else { unreachable!(); };
            uvs.push([*u, *v]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }

    if !model.mesh.vertex_color.is_empty() {
        println!("vertex attributes colors found");
        let mut vertex_color = vec![];
        for color in model.mesh.vertex_color.chunks_exact(3) {
            let [r, g, b] = color else { unreachable!(); };
            vertex_color.push([*r, *g, *b]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_color);
    }

    if !model.mesh.indices.is_empty() {
        println!("vertex attributes indices found");
        let mut indices = vec![];
        for index in &model.mesh.indices {
            indices.push(*index);
        }
        mesh.set_indices(Some(Indices::U32(indices)));
    }

    mesh
}

#[derive(Default, Bundle)]
pub struct ObjBundle {
    pub obj: Handle<LoadedObj>,
}

#[derive(Component)]
pub struct OptimizedMesh {
    pub vertex_buffer: Vec<u8>,
    pub index_buffer: Vec<u8>,
    pub prepared: bool,
}

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

fn as_float2(val: &VertexAttributeValues) -> Option<&[[f32; 2]]> {
    match val {
        VertexAttributeValues::Float32x2(values) => Some(values),
        _ => None,
    }
}

// TODO: this could be done async or directly on load
fn optimize_mesh(
    mut commands: Commands,
    query: Query<(Entity, &Handle<LoadedObj>), Without<OptimizedMesh>>,
    obj_assets: Res<Assets<LoadedObj>>,
) {
    for (entity, obj_handle) in query.iter() {
        if let Some(obj) = obj_assets.get(obj_handle) {
            let start = Instant::now();

            let LoadedObj { meshes } = obj;
            // FIXME: this only uses the first mesh
            // complex models can have multiple meshes
            let mesh = &meshes[0];

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

            commands.entity(entity).insert(OptimizedMesh {
                vertex_buffer,
                index_buffer,
                prepared: false,
            });

            info!("obj mesh optimized {}ms", start.elapsed().as_millis(),);
        }
    }
}
