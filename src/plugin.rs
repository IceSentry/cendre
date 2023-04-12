use std::ffi::CString;

use crate::instance::CendreInstance;
use crate::optimized_mesh::{
    prepare_mesh, IndexBuffer, MeshletBuffer, MeshletsSize, OptimizedMesh, VertexBuffer,
};
use crate::RTX;
use ash::vk;
use bevy::winit::WinitWindows;
use bevy::{prelude::*, window::WindowResized};

pub struct CendrePlugin;
impl Plugin for CendrePlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(init_cendre)
            .add_system(resize.before(update))
            .add_system(prepare_mesh.before(update))
            .add_system(update);
    }
}

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

    let vertex_name = CString::new("vertex".to_string()).unwrap();
    let mesh_name = CString::new("main".to_string()).unwrap();
    let fragment_name = CString::new("fragment".to_string()).unwrap();

    let triangle_fs = cendre.load_shader("assets/shaders/triangle.frag.wgsl");

    // TODO keep pipeline around and use it in update()
    let _pipeline = if RTX {
        let meshlet_ms = cendre.load_shader("assets/shaders/meshlet.mesh.glsl");
        let stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MESH_NV)
                .module(*meshlet_ms.vk_shader_module.lock().unwrap())
                .name(&mesh_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(*triangle_fs.vk_shader_module.lock().unwrap())
                .name(&fragment_name),
        ];
        cendre
            .create_graphics_pipeline(
                // TODO take the actual struct as param
                *pipeline_layout.pipeline_layout.lock().unwrap(),
                cendre.render_pass,
                &stages,
                vk::PrimitiveTopology::TRIANGLE_LIST,
            )
            .expect("Failed to create graphics pipeline")
    } else {
        let triangle_vs = cendre.load_shader("assets/shaders/triangle.vert.wgsl");
        let stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(*triangle_vs.vk_shader_module.lock().unwrap())
                .name(&vertex_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(*triangle_fs.vk_shader_module.lock().unwrap())
                .name(&fragment_name),
        ];
        cendre
            .create_graphics_pipeline(
                *pipeline_layout.pipeline_layout.lock().unwrap(),
                cendre.render_pass,
                &stages,
                vk::PrimitiveTopology::TRIANGLE_LIST,
            )
            .expect("Failed to create graphics pipeline")
    };
    info!("pipeline created");

    commands.insert_resource(cendre);
}

#[allow(clippy::too_many_lines)]
fn update(
    cendre: Res<CendreInstance>,
    windows: Query<&Window>,
    meshlets_size: Option<Res<MeshletsSize>>,
    meshes: Query<(
        &OptimizedMesh,
        &VertexBuffer,
        &IndexBuffer,
        Option<&MeshletBuffer>,
    )>,
) {
    let window = windows.single();

    let device = &cendre.device;

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
            // TODO don't assume there's only one pipeline
            *cendre.pipelines[0].lock().unwrap(),
        );
    }

    for (mesh, vb, ib, mb) in &meshes {
        let vertex_buffer_info = [vb.info(0)];

        if RTX {
            if let Some(mb) = mb {
                let mesh_buffer_info = [mb.info(0)];
                let descriptor_writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&vertex_buffer_info),
                    vk::WriteDescriptorSet::default()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&mesh_buffer_info),
                ];
                unsafe {
                    cendre.push_descriptor.cmd_push_descriptor_set(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        *cendre.pipeline_layouts[0].lock().unwrap(),
                        0,
                        &descriptor_writes,
                    );
                };

                // TODO add to entity
                if let Some(size) = &meshlets_size {
                    unsafe {
                        cendre
                            .mesh_shader
                            .cmd_draw_mesh_tasks(command_buffer, size.0, 0);
                    }
                }

                if let Some(indices) = &mesh.indices {
                    unsafe {
                        device.cmd_draw_indexed(command_buffer, indices.len() as u32, 1, 0, 0, 0);
                    }
                }
            }
        } else {
            let descriptor_writes = [vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&vertex_buffer_info)];

            unsafe {
                cendre.push_descriptor.cmd_push_descriptor_set(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    *cendre.pipeline_layouts[0].lock().unwrap(),
                    0,
                    &descriptor_writes,
                );
            };

            unsafe {
                device.cmd_bind_index_buffer(
                    command_buffer,
                    *ib.buffer_raw.lock().unwrap(),
                    0,
                    vk::IndexType::UINT32,
                );
            }

            if let Some(indices) = &mesh.indices {
                unsafe {
                    device.cmd_draw_indexed(command_buffer, indices.len() as u32, 1, 0, 0, 0);
                }
            }
        }
    }

    // END RENDER PASS

    unsafe {
        device.cmd_end_render_pass(command_buffer);
    }

    // END

    cendre.end_frame(image_index, command_buffer);
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
