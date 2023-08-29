use anyhow::Context;
use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    prelude::*,
    reflect::{TypePath, TypeUuid},
    utils::BoxedFuture,
};
use std::{
    io::{BufReader, Cursor},
    time::Instant,
};

use crate::mesh::{optimize_mesh, Mesh, Vertex};

#[derive(Default, Bundle)]
pub struct ObjBundle {
    pub obj: Handle<LoadedObj>,
}

#[derive(Debug, TypeUuid, TypePath)]
#[uuid = "39cadc56-aa9c-4543-8640-a018b74b5052"]
pub struct LoadedObj {
    pub meshes: Vec<Mesh>,
}

pub struct ObjLoaderPlugin;
impl Plugin for ObjLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<LoadedObj>()
            .init_asset_loader::<ObjLoader>()
            .add_systems(Update, spawn_mesh);
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

    let meshes = models.iter().map(generate_mesh).collect();

    info!(
        "mesh generated and optimized {}ms",
        start.elapsed().as_millis()
    );

    Ok(LoadedObj { meshes })
}

fn generate_mesh(model: &tobj::Model) -> Mesh {
    assert!(!model.mesh.positions.is_empty(), "Mesh requires positions");
    let mut positions = vec![];
    for verts in model.mesh.positions.chunks_exact(3) {
        let [v0, v1, v2] = verts else {
            unreachable!();
        };
        positions.push([*v0, *v1, *v2]);
    }

    let mut uvs = vec![];
    if model.mesh.texcoords.is_empty() {
        uvs = vec![[0.0, 0.0]; positions.len()];
    } else {
        for uv in model.mesh.texcoords.chunks_exact(2) {
            let [u, v] = uv else {
                unreachable!();
            };
            uvs.push([*u, *v]);
        }
    }

    assert!(!model.mesh.indices.is_empty(), "Mesh requires indices");
    let mut indices = vec![];
    for index in &model.mesh.indices {
        indices.push(*index);
    }

    assert!(!model.mesh.normals.is_empty(), "Mesh requires normals");
    let mut normals = vec![];
    for n in model.mesh.normals.chunks_exact(3) {
        let [n0, n1, n2] = n else {
            unreachable!();
        };
        let n = [
            (n0 * 127.0 + 127.0) as u8,
            (n1 * 127.0 + 127.0) as u8,
            (n2 * 127.0 + 127.0) as u8,
            0,
        ];
        normals.push(n);
    }

    let mut vertices = vec![];
    for (pos, (norm, uv)) in positions.iter().zip(normals.iter().zip(uvs.iter())) {
        vertices.push(Vertex {
            pos: [pos[0], pos[1], pos[2], 0.0],
            norm: *norm,
            uv: *uv,
        });
    }

    optimize_mesh(&vertices, &indices)
}

fn spawn_mesh(
    mut commands: Commands,
    query: Query<(Entity, &Handle<LoadedObj>), Without<Mesh>>,
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
