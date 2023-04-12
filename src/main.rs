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

use std::time::Instant;

use ash::vk;
use bevy::window::WindowResized;
use bevy::winit::WinitWindows;
use bevy::{
    a11y::AccessibilityPlugin, app::AppExit, input::InputPlugin, log::LogPlugin, prelude::*,
    winit::WinitPlugin,
};
use cendre::instance::{CendreInstance, Pipeline};
use cendre::obj_loader::{ObjBundle, ObjLoaderPlugin};
use cendre::optimized_mesh::{
    prepare_mesh, IndexBuffer, MeshletBuffer, MeshletsCount, OptimizedMesh, VertexBuffer,
};
use cendre::RTX;

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
        .add_plugin(AssetPlugin {
            watch_for_changes: true,
            ..default()
        })
        .add_plugin(ObjLoaderPlugin)
        .add_startup_system(load_mesh)
        .add_system(exit_on_esc)
        // renderer
        .add_startup_system(init_cendre)
        .add_system(resize.before(update))
        .add_system(prepare_mesh.before(update))
        .add_system(update)
        .run();
}

fn exit_on_esc(key_input: Res<Input<KeyCode>>, mut exit_events: EventWriter<AppExit>) {
    if key_input.just_pressed(KeyCode::Escape) {
        exit_events.send_default();
    }
}

fn load_mesh(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(ObjBundle {
        obj: asset_server.load("models/bunny.obj"),
    });
}

#[derive(Resource)]
pub struct CendrePipeline(pub Pipeline);

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

    let bindings = if RTX {
        vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::MESH_NV),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::MESH_NV),
        ]
    } else {
        vec![vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX)]
    };
    let pipeline_layout = cendre.create_pipeline_layout(&bindings).unwrap();

    let vertex_shader = if RTX {
        cendre.load_shader(
            "assets/shaders/meshlet.mesh.glsl",
            "main",
            vk::ShaderStageFlags::MESH_NV,
        )
    } else {
        cendre.load_shader(
            "assets/shaders/triangle.vert.wgsl",
            "vertex",
            vk::ShaderStageFlags::VERTEX,
        )
    };
    let fragment_shader = cendre.load_shader(
        "assets/shaders/triangle.frag.wgsl",
        "fragment",
        vk::ShaderStageFlags::FRAGMENT,
    );
    let pipeline = cendre
        .create_graphics_pipeline(
            pipeline_layout,
            cendre.render_pass,
            &[vertex_shader.create_info(), fragment_shader.create_info()],
            vk::PrimitiveTopology::TRIANGLE_LIST,
            vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .cull_mode(vk::CullModeFlags::BACK)
                .line_width(1.0),
        )
        .expect("Failed to create graphics pipeline");
    info!("Pipeline created");

    commands.insert_resource(CendrePipeline(pipeline));
    commands.insert_resource(cendre);
}

#[allow(clippy::too_many_lines)]
fn update(
    cendre: Res<CendreInstance>,
    cendre_pipeline: Res<CendrePipeline>,
    mut windows: Query<&mut Window>,
    meshes: Query<(
        &OptimizedMesh,
        &VertexBuffer,
        &IndexBuffer,
        Option<&MeshletBuffer>,
        Option<&MeshletsCount>,
    )>,
) {
    let begin_frame = Instant::now();

    let mut window = windows.single_mut();

    let device = &cendre.device;
    let pipeline = &cendre_pipeline.0;

    // BEGIN

    let (image_index, command_buffer) = cendre.begin_frame();

    // BEGIN RENDER PASS

    unsafe {
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
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );
    }

    // DRAW

    let width = window.physical_width();
    let height = window.physical_height();
    cendre.set_viewport(command_buffer, width, height);

    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.vk_pipeline(),
        );
    }

    for (mesh, vb, ib, mb, meshlets_count) in &meshes {
        let Some(indices) = &mesh.indices else { continue; };

        let vertex_buffer_info = vb.descriptor_info(0);
        if RTX {
            if let Some(mb) = mb {
                let Some(meshlets_count) = &meshlets_count else { continue; };

                let mesh_buffer_info = mb.descriptor_info(0);
                let descriptor_writes = [
                    vb.write_descriptor(0, vk::DescriptorType::STORAGE_BUFFER, &vertex_buffer_info),
                    mb.write_descriptor(1, vk::DescriptorType::STORAGE_BUFFER, &mesh_buffer_info),
                ];
                unsafe {
                    cendre.push_descriptor.cmd_push_descriptor_set(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout.vk_pipeline_layout(),
                        0,
                        &descriptor_writes,
                    );
                    cendre
                        .mesh_shader
                        .cmd_draw_mesh_tasks(command_buffer, meshlets_count.0, 0);
                }
            }
        } else {
            unsafe {
                cendre.push_descriptor.cmd_push_descriptor_set(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout.vk_pipeline_layout(),
                    0,
                    std::slice::from_ref(&vb.write_descriptor(
                        0,
                        vk::DescriptorType::STORAGE_BUFFER,
                        &vertex_buffer_info,
                    )),
                );
                device.cmd_bind_index_buffer(
                    command_buffer,
                    ib.vk_buffer(),
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(command_buffer, indices.len() as u32, 1, 0, 0, 0);
            }
        }
    }

    // END RENDER PASS

    unsafe {
        device.cmd_end_render_pass(command_buffer);
    }

    // END

    cendre.end_frame(image_index, command_buffer);

    window.title = format!(
        "cpu: {:.3}ms gpu: {}ms",
        begin_frame.elapsed().as_secs_f32() * 1000.0,
        0.0
    );
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
