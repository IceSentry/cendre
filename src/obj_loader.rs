use anyhow::Context;
use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    prelude::*,
    reflect::TypeUuid,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    utils::BoxedFuture,
};
use std::{
    io::{BufReader, Cursor},
    time::Instant,
};

use crate::optimized_mesh::{optimize_mesh, OptimizedMesh};

#[derive(Default, Bundle)]
pub struct ObjBundle {
    pub obj: Handle<LoadedObj>,
}

#[derive(Debug, TypeUuid)]
#[uuid = "39cadc56-aa9c-4543-8640-a018b74b5052"]
pub struct LoadedObj {
    pub meshes: Vec<OptimizedMesh>,
}

pub struct ObjLoaderPlugin;
impl Plugin for ObjLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<LoadedObj>()
            .init_asset_loader::<ObjLoader>()
            .add_system(spawn_mesh);
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

    let start = Instant::now();

    let meshes = models
        .iter()
        .map(generate_mesh)
        .map(optimize_mesh)
        .collect();

    info!(
        "mesh generated and optimized {}ms",
        start.elapsed().as_millis()
    );

    Ok(LoadedObj { meshes })
}

fn generate_mesh(model: &tobj::Model) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    if !model.mesh.positions.is_empty() {
        let mut positions = vec![];
        for verts in model.mesh.positions.chunks_exact(3) {
            let [v0, v1, v2] = verts else {
                unreachable!();
            };
            positions.push([*v0, *v1, *v2]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    }

    if !model.mesh.texcoords.is_empty() {
        let mut uvs = vec![];
        for uv in model.mesh.texcoords.chunks_exact(2) {
            let [u, v] = uv else {
                unreachable!();
            };
            uvs.push([*u, *v]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }

    if !model.mesh.vertex_color.is_empty() {
        let mut vertex_color = vec![];
        for color in model.mesh.vertex_color.chunks_exact(3) {
            let [r, g, b] = color else {
                unreachable!();
            };
            vertex_color.push([*r, *g, *b]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_color);
    }

    if !model.mesh.indices.is_empty() {
        let mut indices = vec![];
        for index in &model.mesh.indices {
            indices.push(*index);
        }
        mesh.set_indices(Some(Indices::U32(indices)));
    }

    if model.mesh.normals.is_empty() {
        mesh.duplicate_vertices();
        mesh.compute_flat_normals();
    } else {
        let mut normals = vec![];
        for n in model.mesh.normals.chunks_exact(3) {
            let [n0, n1, n2] = n else {
                unreachable!();
            };
            normals.push([*n0, *n1, *n2]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    }

    mesh
}

fn spawn_mesh(
    mut commands: Commands,
    query: Query<(Entity, &Handle<LoadedObj>), Without<OptimizedMesh>>,
    obj_assets: Res<Assets<LoadedObj>>,
) {
    for (entity, obj_handle) in query.iter() {
        if let Some(obj) = obj_assets.get(obj_handle) {
            let LoadedObj { meshes } = obj;
            // FIXME: this only uses the first mesh
            // complex models can have multiple meshes
            // This should probably just loop on every mesh and spawn child entity with the optimized meshes.
            // Or even just spawn child entities with `Mesh`es and optimize them in a separate system.
            let mesh = &meshes[0];
            commands.entity(entity).insert(mesh.clone());
        }
    }
}
