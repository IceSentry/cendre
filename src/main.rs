#![warn(clippy::pedantic)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_lines)]

use std::time::{Duration, Instant};

use ash::vk::{self, DescriptorUpdateTemplateType};
use bevy::{
    a11y::AccessibilityPlugin,
    app::AppExit,
    asset::ChangeWatcher,
    input::InputPlugin,
    log::LogPlugin,
    prelude::*,
    window::WindowResized,
    winit::{WinitPlugin, WinitWindows},
};
use cendre::{
    instance::{CendreInstance, DescriptorUpdateTemplate, Pipeline},
    mesh::{prepare_mesh, IndexBuffer, Mesh, MeshletBuffer, MeshletsCount, VertexBuffer},
    obj_loader::{ObjBundle, ObjLoaderPlugin},
    RTXEnabled,
};

pub const OBJ_PATH: &str = "models/bunny.obj";

fn main() {
    App::new()
        .insert_resource(RTXEnabled(true))
        .add_plugins((
            MinimalPlugins,
            WindowPlugin {
                primary_window: Some(Window {
                    title: "cendre".into(),
                    // For some reason it seems like this is being ignored?
                    // present_mode: PresentMode::AutoVsync,
                    // present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            },
            // if you need to output the logs to a file just do:
            // cargo r 2>&1 > cendre.log
            LogPlugin::default(),
            AccessibilityPlugin,
            WinitPlugin,
            InputPlugin,
            AssetPlugin {
                watch_for_changes: ChangeWatcher::with_delay(Duration::from_millis(250)),
                ..default()
            },
            ObjLoaderPlugin,
        ))
        .add_systems(Startup, (init_cendre, load_mesh))
        .add_systems(Update, (resize, update).chain())
        .add_systems(Update, (prepare_mesh, toggle_rtx, exit_on_esc))
        .run();
}

fn exit_on_esc(key_input: Res<Input<KeyCode>>, mut exit_events: EventWriter<AppExit>) {
    if key_input.just_pressed(KeyCode::Escape) {
        exit_events.send_default();
    }
}

fn load_mesh(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(ObjBundle {
        obj: asset_server.load(OBJ_PATH),
    });
}

#[derive(Resource)]
pub struct CendrePipeline(pub Pipeline);
#[derive(Resource)]
pub struct CendrePipelineRTX(pub Pipeline);

#[derive(Resource, Deref)]
pub struct CendreMeshUpdateTemplateRTX(pub DescriptorUpdateTemplate);
#[derive(Resource, Deref)]
pub struct CendreMeshUpdateTemplate(pub DescriptorUpdateTemplate);

#[allow(clippy::too_many_lines)]
fn init_cendre(
    mut commands: Commands,
    windows: Query<Entity, With<Window>>,
    winit_windows: NonSendMut<WinitWindows>,
) {
    let winit_window = windows
        .get_single()
        .ok()
        .and_then(|window_id| winit_windows.get_window(window_id))
        .expect("Failed to get winit window");

    let mut cendre = CendreInstance::init(winit_window);
    info!("Instance created");

    let vertex_shader = cendre.load_shader("assets/shaders/mesh.vert.glsl");
    let fragment_shader = cendre.load_shader("assets/shaders/mesh.frag.glsl");

    let pipeline_rasterization_state_create_info =
        vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .front_face(vk::FrontFace::CLOCKWISE)
            .cull_mode(vk::CullModeFlags::FRONT)
            .line_width(1.0);

    if cendre.rtx_supported {
        let mesh_shader = cendre.load_shader("assets/shaders/meshlet.mesh.glsl");
        let task_shader = cendre.load_shader("assets/shaders/meshlet.task.glsl");

        let mesh_layout_rtx = cendre
            .create_pipeline_layout(&[&mesh_shader, &fragment_shader])
            .unwrap();

        let mesh_update_template_rtx = cendre
            .create_update_template(
                vk::PipelineBindPoint::GRAPHICS,
                DescriptorUpdateTemplateType::PUSH_DESCRIPTORS_KHR,
                &mesh_layout_rtx,
                &[&task_shader, &mesh_shader, &fragment_shader],
            )
            .unwrap();
        commands.insert_resource(CendreMeshUpdateTemplateRTX(mesh_update_template_rtx));
        let pipeline_rtx = cendre
            .create_graphics_pipeline(
                mesh_layout_rtx,
                cendre.render_pass,
                &[
                    task_shader.create_info(),
                    mesh_shader.create_info(),
                    fragment_shader.create_info(),
                ],
                vk::PrimitiveTopology::TRIANGLE_LIST,
                pipeline_rasterization_state_create_info,
            )
            .expect("Failed to create graphics pipeline RTX");
        commands.insert_resource(CendrePipelineRTX(pipeline_rtx));
    }

    let mesh_layout = cendre
        .create_pipeline_layout(&[&vertex_shader, &fragment_shader])
        .unwrap();
    let mesh_update_template = cendre
        .create_update_template(
            vk::PipelineBindPoint::GRAPHICS,
            DescriptorUpdateTemplateType::PUSH_DESCRIPTORS_KHR,
            &mesh_layout,
            &[&vertex_shader, &fragment_shader],
        )
        .unwrap();
    commands.insert_resource(CendreMeshUpdateTemplate(mesh_update_template));

    let pipeline = cendre
        .create_graphics_pipeline(
            mesh_layout,
            cendre.render_pass,
            &[vertex_shader.create_info(), fragment_shader.create_info()],
            vk::PrimitiveTopology::TRIANGLE_LIST,
            pipeline_rasterization_state_create_info,
        )
        .expect("Failed to create graphics pipeline");

    commands.insert_resource(CendrePipeline(pipeline));
    commands.insert_resource(cendre);

    info!("Pipeline created");
}

#[allow(clippy::too_many_lines)]
fn update(
    cendre: Res<CendreInstance>,
    cendre_pipeline: Res<CendrePipeline>,
    cendre_pipeline_rtx: Option<Res<CendrePipelineRTX>>,
    mut windows: Query<&mut Window>,
    meshes: Query<(
        &Mesh,
        &VertexBuffer,
        &IndexBuffer,
        Option<&MeshletBuffer>,
        Option<&MeshletsCount>,
    )>,
    mut frame_gpu_avg: Local<f64>,
    mut frame_cpu_avg: Local<f64>,
    rtx_enabled: Res<RTXEnabled>,
    mesh_update_template_rtx: Res<CendreMeshUpdateTemplateRTX>,
    mesh_update_template: Res<CendreMeshUpdateTemplate>,
) {
    let begin_frame = Instant::now();

    let rtx = cendre.rtx_supported && rtx_enabled.0;

    let mut window = windows.single_mut();

    // BEGIN

    let (image_index, command_buffer) = cendre.begin_frame();

    // BEGIN RENDER PASS

    #[cfg(feature = "trace")]
    let _span = bevy::utils::tracing::info_span!("begin render pass").entered();

    let clear_color = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.3, 0.3, 0.3, 1.0],
        },
    };
    let render_pass_begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(cendre.render_pass)
        .framebuffer(cendre.swapchain.framebuffers[image_index as usize])
        .render_area(vk::Rect2D::default().extent(vk::Extent2D {
            width: cendre.swapchain.width,
            height: cendre.swapchain.height,
        }))
        .clear_values(std::slice::from_ref(&clear_color));
    unsafe {
        cendre.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );
    }

    // DRAW
    {
        #[cfg(feature = "trace")]
        let _span = bevy::utils::tracing::info_span!("draw").entered();

        let width = window.physical_width();
        let height = window.physical_height();
        cendre.set_viewport(command_buffer, width, height);

        let pipeline = if rtx {
            &cendre_pipeline_rtx.as_ref().unwrap().0
        } else {
            &cendre_pipeline.0
        };
        cendre.bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        let draw_count = 2000;
        // let draw_count = 1;

        for (mesh, vb, ib, mb, meshlets_count) in &meshes {
            let indices = &mesh.indices;

            if rtx {
                let Some(mb) = mb else {
                    continue;
                };
                let Some(meshlets_count) = meshlets_count else {
                    continue;
                };

                let descriptors = [vb.descriptor_info(0), mb.descriptor_info(0)];
                cendre.push_descriptor_set_with_template(
                    command_buffer,
                    &mesh_update_template_rtx,
                    &pipeline.layout,
                    0,
                    &descriptors,
                );

                for _ in 0..draw_count {
                    cendre.draw_mesh_tasks(command_buffer, meshlets_count.0 / 32, 0);
                }
            } else {
                let descriptors = [vb.descriptor_info(0)];
                cendre.push_descriptor_set_with_template(
                    command_buffer,
                    &mesh_update_template,
                    &pipeline.layout,
                    0,
                    &descriptors,
                );
                cendre.bind_index_buffer(command_buffer, ib, 0, vk::IndexType::UINT32);
                for _ in 0..draw_count {
                    cendre.draw_indexed(command_buffer, indices.len() as u32, 1, 0, 0, 0);
                }
            }
        }
    }

    // END RENDER PASS

    unsafe {
        #[cfg(feature = "trace")]
        let _span = bevy::utils::tracing::info_span!("end render pass").entered();
        cendre.device.cmd_end_render_pass(command_buffer);
    }

    // END

    cendre.end_frame(image_index, command_buffer);

    {
        #[cfg(feature = "trace")]
        let _span = bevy::utils::tracing::info_span!("update frame time").entered();

        let (frame_gpu_begin, frame_gpu_end) = cendre.get_frame_time();
        *frame_gpu_avg = *frame_gpu_avg * 0.9 + (frame_gpu_end - frame_gpu_begin) * 0.1;
        *frame_cpu_avg =
            *frame_cpu_avg * 0.9 + (begin_frame.elapsed().as_secs_f64() * 1000.0) * 0.1;

        window.title = format!(
            "cpu: {:.2}ms gpu: {:.2}ms RTX: {}",
            *frame_cpu_avg,
            *frame_gpu_avg,
            if rtx_enabled.0 { "ON" } else { "OFF" }
        );
    }
}

fn resize(mut events: EventReader<WindowResized>, mut cendre: ResMut<CendreInstance>) {
    if events.is_empty() {
        return;
    }
    events.clear();
    let surface_capabilities = unsafe {
        cendre
            .surface_loader
            .get_physical_device_surface_capabilities(cendre.physical_device, cendre.surface)
            .unwrap()
    };

    let new_width = surface_capabilities.current_extent.width;
    let new_height = surface_capabilities.current_extent.height;
    if cendre.swapchain.width == new_width && cendre.swapchain.height == new_height {
        // FIXME: this will break with multiple windows
        return;
    }

    cendre.swapchain = cendre.swapchain.resize(
        &cendre.device,
        &cendre.swapchain_loader,
        &cendre.surface_loader,
        cendre.surface,
        cendre.surface_format,
        cendre.physical_device,
        new_width,
        new_height,
        cendre.render_pass,
    );
}

fn toggle_rtx(mut rtx_enabled: ResMut<RTXEnabled>, key_input: Res<Input<KeyCode>>) {
    if key_input.just_pressed(KeyCode::R) {
        rtx_enabled.0 = !rtx_enabled.0;
    }
}
