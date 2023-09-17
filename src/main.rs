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

use ash::vk::{self};
use bevy::{
    a11y::AccessibilityPlugin,
    app::AppExit,
    asset::ChangeWatcher,
    input::InputPlugin,
    log::LogPlugin,
    prelude::*,
    window::{PrimaryWindow, WindowResized},
    winit::{WinitPlugin, WinitWindows},
};
use cendre::{
    instance::{CendreInstance, Pipeline, Program},
    mesh::{
        prepare_mesh, IndexBuffer, Mesh, MeshDraw, MeshletBuffer, MeshletDataBuffer, MeshletsCount,
        VertexBuffer,
    },
    obj_loader::{ObjBundle, ObjLoaderPlugin},
    MeshShaderEnabled,
};

pub const OBJ_PATH: &str = "models/bunny.obj";
pub const VSYNC: bool = false;

fn main() {
    App::new()
        .insert_resource(MeshShaderEnabled(true))
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
        .add_systems(Update, (prepare_mesh, toggle_mesh_shader, exit_on_esc))
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
pub struct CendrePipelineMeshShader(pub Pipeline);
#[derive(Resource, Deref)]
pub struct MeshProgram(pub Program);
#[derive(Resource, Deref)]
pub struct MeshProgramMeshShader(pub Program);

#[allow(clippy::too_many_lines)]
fn init_cendre(
    mut commands: Commands,
    windows: Query<Entity, With<PrimaryWindow>>,
    winit_windows: NonSendMut<WinitWindows>,
) {
    let winit_window = windows
        .get_single()
        .ok()
        .and_then(|window_id| winit_windows.get_window(window_id))
        .expect("Failed to get winit window");

    let mut cendre = CendreInstance::init(
        winit_window,
        if VSYNC {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::IMMEDIATE
        },
    );
    info!("Instance created");

    // let color_target = cendre.create_image(cendre.swapchain.width, cendre.swapchain.height, cendre.swapchain, usage, memory_location);

    let vertex_shader = cendre.load_shader("assets/shaders/mesh.vert.glsl");
    let fragment_shader = cendre.load_shader("assets/shaders/mesh.frag.glsl");

    let pipeline_rasterization_state_create_info =
        vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .front_face(vk::FrontFace::CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .line_width(1.0);

    if cendre.mesh_shader_supported {
        let mesh_shader = cendre.load_shader("assets/shaders/meshlet.mesh.glsl");
        let task_shader = cendre.load_shader("assets/shaders/meshlet.task.glsl");

        let mesh_program_mesh_shader = cendre
            .create_program(
                vk::PipelineBindPoint::GRAPHICS,
                &[&task_shader, &mesh_shader, &fragment_shader],
                std::mem::size_of::<MeshDraw>() as u32,
            )
            .expect("Failed to create mesh_program_mesh_shader");
        let pipeline_mesh_shader = cendre
            .create_graphics_pipeline(
                &mesh_program_mesh_shader.layout,
                cendre.render_pass,
                &[
                    task_shader.create_info(),
                    mesh_shader.create_info(),
                    fragment_shader.create_info(),
                ],
                vk::PrimitiveTopology::TRIANGLE_LIST,
                pipeline_rasterization_state_create_info,
            )
            .expect("Failed to create graphics pipeline mesh shader");
        commands.insert_resource(CendrePipelineMeshShader(pipeline_mesh_shader));
        commands.insert_resource(MeshProgramMeshShader(mesh_program_mesh_shader));
    }

    let mesh_program = cendre
        .create_program(
            vk::PipelineBindPoint::GRAPHICS,
            &[&vertex_shader, &fragment_shader],
            std::mem::size_of::<MeshDraw>() as u32,
        )
        .expect("Failed to create mesh_program");
    let pipeline = cendre
        .create_graphics_pipeline(
            &mesh_program.layout,
            cendre.render_pass,
            &[vertex_shader.create_info(), fragment_shader.create_info()],
            vk::PrimitiveTopology::TRIANGLE_LIST,
            pipeline_rasterization_state_create_info,
        )
        .expect("Failed to create graphics pipeline");
    commands.insert_resource(CendrePipeline(pipeline));
    commands.insert_resource(MeshProgram(mesh_program));

    commands.insert_resource(cendre);

    info!("Pipeline created");
}

#[allow(clippy::too_many_lines)]
fn update(
    cendre: Res<CendreInstance>,
    cendre_pipeline: Res<CendrePipeline>,
    cendre_pipeline_mesh_shader: Option<Res<CendrePipelineMeshShader>>,
    mut windows: Query<&mut Window>,
    meshes: Query<(
        &Mesh,
        &VertexBuffer,
        &IndexBuffer,
        Option<&MeshletBuffer>,
        Option<&MeshletsCount>,
        Option<&MeshletDataBuffer>,
    )>,
    mut frame_gpu_avg: Local<f64>,
    mut frame_cpu_avg: Local<f64>,
    mesh_shader_enabled: Res<MeshShaderEnabled>,
    mesh_program_mesh_shader: Res<MeshProgramMeshShader>,
    mesh_program: Res<MeshProgram>,
) {
    let begin_frame = Instant::now();

    let mesh_shader = cendre.mesh_shader_supported && mesh_shader_enabled.0;

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
        .framebuffer(cendre.framebuffers[image_index as usize])
        .render_area(vk::Rect2D::default().extent(vk::Extent2D {
            width: cendre.swapchain_width,
            height: cendre.swapchain_height,
        }))
        .clear_values(std::slice::from_ref(&clear_color));
    unsafe {
        cendre.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );
    }

    let mut triangle_count = 0;

    // DRAW
    {
        #[cfg(feature = "trace")]
        let _span = bevy::utils::tracing::info_span!("draw").entered();

        let width = window.physical_width();
        let height = window.physical_height();
        cendre.set_viewport(command_buffer, width, height);

        let pipeline = if mesh_shader {
            &cendre_pipeline_mesh_shader.as_ref().unwrap().0
        } else {
            &cendre_pipeline.0
        };
        cendre.bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        let draw_count = 100;

        let mut draws = Vec::new();
        for i in 0..draw_count {
            #[rustfmt::skip]
            draws.push(MeshDraw {
                offset: [
                    (i % 10) as f32 / 10.0 + 0.5 / 10.0,
                    (i / 10) as f32 / 10.0 + 0.5 / 10.0
                ],
                scale: [
                    1.0 / 10.0,
                    1.0 / 10.0
                ],
            });
        }

        for (mesh, vb, ib, mb, meshlets_count, mdb) in &meshes {
            triangle_count += (mesh.indices.len() / 3) * draw_count;

            if mesh_shader {
                let Some(mb) = mb else {
                    continue;
                };
                let Some(meshlets_count) = meshlets_count else {
                    continue;
                };
                let Some(mdb) = mdb else {
                    continue;
                };

                let descriptors = [
                    vb.descriptor_info(0),
                    mb.descriptor_info(0),
                    mdb.descriptor_info(0),
                ];
                cendre.push_descriptor_set_with_template(
                    command_buffer,
                    &mesh_program_mesh_shader,
                    0,
                    &descriptors,
                );

                for draw in &draws {
                    cendre.push_constants(
                        command_buffer,
                        &mesh_program_mesh_shader,
                        0,
                        bytemuck::bytes_of(draw),
                    );
                    cendre.draw_mesh_tasks(command_buffer, meshlets_count.0 / 32, 0);
                }
            } else {
                let descriptors = [vb.descriptor_info(0)];
                cendre.push_descriptor_set_with_template(
                    command_buffer,
                    &mesh_program,
                    0,
                    &descriptors,
                );
                cendre.bind_index_buffer(command_buffer, ib, 0, vk::IndexType::UINT32);
                for draw in &draws {
                    cendre.push_constants(
                        command_buffer,
                        &mesh_program,
                        0,
                        bytemuck::bytes_of(draw),
                    );
                    cendre.draw_indexed(command_buffer, mesh.indices.len() as u32, 1, 0, 0, 0);
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
        *frame_gpu_avg = *frame_gpu_avg * 0.95 + (frame_gpu_end - frame_gpu_begin) * 0.05;
        let frame_cpu = begin_frame.elapsed().as_secs_f64() * 1000.0;
        *frame_cpu_avg = *frame_cpu_avg * 0.95 + frame_cpu * 0.05;

        let triangles_per_sec = triangle_count as f64 / (*frame_gpu_avg * 1e-3);

        window.title = format!(
            "cpu: {:.2} ms gpu: {:.2} ms mesh shading: {} {:.2}B tri/sec",
            *frame_cpu_avg,
            *frame_gpu_avg,
            if mesh_shader_enabled.0 { "ON" } else { "OFF" },
            triangles_per_sec / 1e9,
        );
    }
}

fn resize(mut events: EventReader<WindowResized>, mut cendre: ResMut<CendreInstance>) {
    if events.is_empty() {
        return;
    }
    for event in &mut events {
        if cendre.swapchain_width == event.width as u32
            && cendre.swapchain_height == event.height as u32
        {
            // FIXME: this will break with multiple windows
            return;
        }
        cendre.recreate_swapchain(event.width as u32, event.height as u32);
    }
}

fn toggle_mesh_shader(
    mut mesh_shader_enabled: ResMut<MeshShaderEnabled>,
    key_input: Res<Input<KeyCode>>,
) {
    if key_input.just_pressed(KeyCode::R) {
        mesh_shader_enabled.0 = !mesh_shader_enabled.0;
    }
}
