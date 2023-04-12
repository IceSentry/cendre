use std::{ffi::OsStr, io::Write, path::PathBuf};

use anyhow::Context;
use bevy::{
    a11y::AccessibilityPlugin, app::AppExit, input::InputPlugin, log::LogPlugin, prelude::*,
    winit::WinitPlugin,
};
use cendre::obj_loader::{ObjBundle, ObjLoaderPlugin};
use naga::valid::Capabilities;

fn write_binary_file(path: &PathBuf, data: &[u8]) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path.clone())
        .context(format!("Failed to open {path:?}"))
        .unwrap();
    file.write_all(data).unwrap();
}

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

        println!("Compiling shader: {}", entry.to_string_lossy());

        let source = std::fs::read_to_string(entry.clone()).unwrap();

        let module = match extension {
            "wgsl" => match naga::front::wgsl::parse_str(&source) {
                Ok(module) => module,
                Err(err) => {
                    println!("\x1b[91mERROR\x1b[0m {}: {}", entry.to_string_lossy(), err);
                    panic!("Invalid wgsl shader {}", entry.to_string_lossy())
                }
            },
            "glsl" => {
                let file_name = entry.file_name().unwrap().to_string_lossy();

                let compiler = shaderc::Compiler::new().unwrap();
                let options = shaderc::CompileOptions::new().unwrap();
                let shader_kind = if file_name.contains(".vert") {
                    shaderc::ShaderKind::Vertex
                } else if file_name.contains(".frag") {
                    shaderc::ShaderKind::Fragment
                } else if file_name.contains(".mesh") {
                    shaderc::ShaderKind::Mesh
                } else {
                    todo!()
                };

                match compiler.compile_into_spirv(
                    &source,
                    shader_kind,
                    &file_name,
                    "main",
                    Some(&options),
                ) {
                    Ok(result) => {
                        let mut path = entry;
                        path.set_extension("spv");
                        write_binary_file(&path, result.as_binary_u8());
                    }
                    Err(err) => {
                        println!("\x1b[91mERROR\x1b[0m: {}", err);
                        panic!("Invalid glsl shader {}", entry.to_string_lossy())
                    }
                }
                continue;

                // TODO naga doesn't currently support the required feature for glsl to use mesh shaders
                //
                // let mut parser = naga::front::glsl::Frontend::default();
                // let options = if file_name.contains(".vert") {
                //     naga::front::glsl::Options::from(ShaderStage::Vertex)
                // } else if file_name.contains(".frag") {
                //     naga::front::glsl::Options::from(ShaderStage::Fragment)
                // } else if file_name.contains(".mesh") {
                //     naga::front::glsl::Options::from(ShaderStage::Mesh)
                // } else {
                //     todo!()
                // };

                // match parser.parse(&options, &source) {
                //     Ok(module) => module,
                //     Err(errors) => {
                //         for err in errors {
                //             println!("\x1b[91mERROR\x1b[0m {}: {}", entry.to_string_lossy(), err);
                //         }
                //         panic!("Invalid glsl shader {}", entry.to_string_lossy())
                //     }
                // }
            }
            _ => panic!("Unknown shader format"),
        };

        let module_info = match naga::valid::Validator::new(
            naga::valid::ValidationFlags::default(),
            Capabilities::all(),
        )
        .validate(&module)
        {
            Ok(module_info) => module_info,
            Err(err) => {
                println!("{err}");
                panic!("Shader validation error for {}", entry.to_string_lossy());
            }
        };
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
        write_binary_file(
            &path,
            &spv.iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<u8>>(),
        );
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
        .add_plugin(cendre::plugin::CendrePlugin)
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
