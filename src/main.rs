mod obj_loader;

use std::{ffi::OsStr, io::Write};

use anyhow::Context;
use bevy::{
    a11y::AccessibilityPlugin, app::AppExit, input::InputPlugin, log::LogPlugin, prelude::*,
    winit::WinitPlugin,
};
use naga::{valid::Capabilities, ShaderStage};
use obj_loader::{ObjBundle, ObjLoaderPlugin};

/// Compiles to spir-v any wgsl or glsl shaders found in ./assets/shaders
/// For glsl it uses .frag.glsl and .vert.glsl to detect the shader stage
// TODO use asset system for shaders
fn compile_shaders() {
    for entry in std::fs::read_dir("./assets/shaders/")
        .unwrap()
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
    {
        let extension = entry.extension().and_then(OsStr::to_str).unwrap();
        if extension == "spv" {
            // skip already compiled shader
            continue;
        }

        let source = std::fs::read_to_string(entry.clone()).unwrap();

        let module = match extension {
            "wgsl" => naga::front::wgsl::parse_str(&source).unwrap(),
            "glsl" => {
                let mut parser = naga::front::glsl::Parser::default();
                let file_name = entry.file_name().unwrap().to_string_lossy();
                let options = if file_name.contains(".vert") {
                    naga::front::glsl::Options::from(ShaderStage::Vertex)
                } else if file_name.contains(".frag") {
                    naga::front::glsl::Options::from(ShaderStage::Fragment)
                } else {
                    todo!()
                };
                parser.parse(&options, &source).unwrap()
            }
            _ => panic!("Unknown shader format"),
        };

        let module_info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::default(),
            Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let spv = naga::back::spv::write_vec(
            &module,
            &module_info,
            &naga::back::spv::Options {
                flags: naga::back::spv::WriterFlags::empty(),
                ..naga::back::spv::Options::default()
            },
            None,
        )
        .unwrap();
        let mut path = entry;
        path.set_extension("spv");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(path.clone())
            .context(format!("Failed to open {path:?}"))
            .unwrap();
        file.write_all(
            &spv.iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
    }
}

fn main() -> anyhow::Result<()> {
    compile_shaders();

    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugin(WindowPlugin {
            primary_window: Some(Window {
                title: "cendre".into(),
                ..default()
            }),
            ..default()
        })
        .add_plugin(AccessibilityPlugin)
        .add_plugin(WinitPlugin)
        .add_plugin(InputPlugin)
        .add_plugin(LogPlugin::default())
        .add_plugin(AssetPlugin {
            watch_for_changes: true,
            ..default()
        })
        .add_plugin(cendre::CendrePlugin)
        .add_plugin(ObjLoaderPlugin)
        .add_startup_system(load_mesh)
        .add_system(exit_on_esc)
        .run();

    Ok(())
}

fn exit_on_esc(key_input: Res<Input<KeyCode>>, mut exit_events: EventWriter<AppExit>) {
    if key_input.just_pressed(KeyCode::Escape) {
        exit_events.send_default();
    }
}

fn load_mesh(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(ObjBundle {
        obj: asset_server.load("bunny.obj"),
    });
}
