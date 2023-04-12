use bevy::{
    a11y::AccessibilityPlugin, app::AppExit, input::InputPlugin, log::LogPlugin, prelude::*,
    winit::WinitPlugin,
};
use cendre::{
    obj_loader::{ObjBundle, ObjLoaderPlugin},
    shaders::compile_shaders,
};

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
