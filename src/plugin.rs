use crate::instance::CendreInstance;
use crate::optimized_mesh::{
    prepare_mesh, IndexBuffer, MeshletBuffer, MeshletsSize, OptimizedMesh, VertexBuffer,
};
use crate::{image_barrier, RTX};
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

    commands.insert_resource(CendreInstance::init(winit_window));
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
    let acquire_semaphores = [cendre.acquire_semaphore];
    let release_semaphores = [cendre.release_semaphore];
    let device = &cendre.device;

    let (image_index, _) = unsafe {
        cendre
            .swapchain_loader
            .acquire_next_image(
                cendre.swapchain.swapchain,
                0,
                cendre.acquire_semaphore,
                vk::Fence::null(),
            )
            .expect("Failed to acquire next image")
    };

    unsafe {
        device
            .reset_command_pool(cendre.command_pool, vk::CommandPoolResetFlags::empty())
            .expect("Failed to reset command_pool");
    }

    let command_buffer = cendre.command_buffers[0];

    // BEGIN

    unsafe {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap();
    }

    unsafe {
        let render_begin_barrier = image_barrier(
            cendre.swapchain.images[image_index as usize],
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[render_begin_barrier],
        );
    }

    // CLEAR

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

    let viewport = vk::Viewport {
        x: 0.0,
        y: height as f32,
        width: width as f32,
        height: -(height as f32),
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D::default().extent(vk::Extent2D::default().width(width).height(height));

    unsafe { device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport)) };
    unsafe { device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor)) };

    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            cendre.pipeline,
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
                        cendre.pipeline_layout,
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
                    cendre.pipeline_layout,
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

    // END

    unsafe {
        device.cmd_end_render_pass(command_buffer);
    }

    unsafe {
        let render_end_barrier = image_barrier(
            cendre.swapchain.images[image_index as usize],
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::empty(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[render_end_barrier],
        );
    }

    unsafe {
        device.end_command_buffer(command_buffer).unwrap();
    }

    unsafe {
        let submits = [vk::SubmitInfo::default()
            .wait_semaphores(&acquire_semaphores)
            .wait_dst_stage_mask(std::slice::from_ref(
                &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ))
            .command_buffers(&cendre.command_buffers)
            .signal_semaphores(&release_semaphores)];
        device
            .queue_submit(cendre.present_queue, &submits, vk::Fence::null())
            .unwrap();
    }

    unsafe {
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(std::slice::from_ref(&cendre.swapchain.swapchain))
            .image_indices(std::slice::from_ref(&image_index))
            .wait_semaphores(&release_semaphores);
        cendre
            .swapchain_loader
            .queue_present(cendre.present_queue, &present_info)
            .expect("Failed to queue present");
    }

    unsafe {
        device.device_wait_idle().unwrap();
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
