use std::{ffi::OsStr, io::Write};

use anyhow::Context;
use bevy::{
    a11y::AccessibilityPlugin, app::AppExit, input::InputPlugin, log::LogPlugin, prelude::*,
    winit::WinitPlugin,
};
use naga::{valid::Capabilities, ShaderStage};

/// Compiles to spir-v any wgsl or glsl shaders found in ./assets/shaders
/// For glsl it uses .frag.glsl and .vert.glsl to detect the shader stage
fn compile_shaders() -> anyhow::Result<()> {
    for entry in std::fs::read_dir("./assets/shaders/")?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?
    {
        let module = match entry.extension().and_then(OsStr::to_str) {
            Some("spv") => {
                // skip already compiled spirv
                continue;
            }
            Some("wgsl") => {
                let source = std::fs::read_to_string(entry.clone())?;
                naga::front::wgsl::parse_str(&source)?
            }
            Some("glsl") => {
                let source = std::fs::read_to_string(entry.clone())?;
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
        .validate(&module)?;
        let spv = naga::back::spv::write_vec(
            &module,
            &module_info,
            &naga::back::spv::Options {
                flags: naga::back::spv::WriterFlags::empty(),
                ..naga::back::spv::Options::default()
            },
            None,
        )?;
        let mut path = entry;
        path.set_extension("spv");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(path.clone())
            .context(format!("Failed to open {path:?}"))?;
        file.write_all(
            &spv.iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<u8>>(),
        )?;
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    compile_shaders()?;

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
        .add_plugin(cendre::CendrePlugin)
        .add_system(exit_on_esc)
        .run();

    Ok(())
}

fn exit_on_esc(key_input: Res<Input<KeyCode>>, mut exit_events: EventWriter<AppExit>) {
    if key_input.just_pressed(KeyCode::Escape) {
        exit_events.send_default();
    }
}
