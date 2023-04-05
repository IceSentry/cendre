use bevy::{
    a11y::AccessibilityPlugin, app::AppExit, input::InputPlugin, log::LogPlugin, prelude::*,
    winit::WinitPlugin,
};

fn main() {
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
}

fn exit_on_esc(key_input: Res<Input<KeyCode>>, mut exit_events: EventWriter<AppExit>) {
    if key_input.just_pressed(KeyCode::Escape) {
        exit_events.send_default();
    }
}
