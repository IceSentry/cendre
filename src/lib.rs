use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    os::raw::c_char,
};

use ash::{
    extensions::khr::{Surface, Swapchain},
    vk::SurfaceFormatKHR,
    Device,
};
use ash::{vk, Entry, Instance};
use bevy::{prelude::*, winit::WinitWindows};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

pub struct CendrePlugin;
impl Plugin for CendrePlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(hide_window)
            .add_startup_system(init_vulkan.after(hide_window))
            .add_startup_system(show_window.after(init_vulkan))
            .add_system(update);
    }
}

fn hide_window(windows: Query<Entity, With<Window>>, winit_windows: NonSendMut<WinitWindows>) {
    let winit_window = windows
        .get_single()
        .ok()
        .and_then(|window_id| winit_windows.get_window(window_id))
        .expect("Failed to get winit window");
    winit_window.set_visible(false);
}

fn show_window(windows: Query<Entity, With<Window>>, winit_windows: NonSendMut<WinitWindows>) {
    let winit_window = windows
        .get_single()
        .ok()
        .and_then(|window_id| winit_windows.get_window(window_id))
        .expect("Failed to get winit window");
    winit_window.set_visible(true);
}

#[derive(Resource)]
struct CendreRenderer {
    instance: Instance,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
    acquire_semaphore: vk::Semaphore,
    release_semaphore: vk::Semaphore,
    present_queue: vk::Queue,
    present_images: Vec<vk::Image>,
    present_image_views: Vec<vk::ImageView>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

#[derive(Resource, Deref)]
struct CendreDevice(pub Device);

unsafe fn create_render_pass(
    device: Device,
    surface_format: &SurfaceFormatKHR,
) -> anyhow::Result<vk::RenderPass> {
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpass = vk::SubpassDescription::default()
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

    let attachment = vk::AttachmentDescription::default()
        .format(surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass));
    Ok(device.create_render_pass(&create_info, None)?)
}

fn update(cendre: Res<CendreRenderer>, device: Res<CendreDevice>) {
    unsafe {
        let acquire_semaphores = [cendre.acquire_semaphore];
        let release_semaphores = [cendre.release_semaphore];

        let (image_index, _) = cendre
            .swapchain_loader
            .acquire_next_image(
                cendre.swapchain,
                0,
                cendre.acquire_semaphore,
                vk::Fence::null(),
            )
            .expect("Failed to acquire next image");

        device
            .reset_command_pool(cendre.command_pool, vk::CommandPoolResetFlags::empty())
            .expect("Failed to reset command_pool");

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cendre.command_buffers[0], &begin_info)
            .unwrap();

        let ranges = [vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1)];
        device.cmd_clear_color_image(
            cendre.command_buffers[0],
            cendre.present_images[0],
            vk::ImageLayout::GENERAL,
            &vk::ClearColorValue {
                float32: [1.0, 0.0, 1.0, 1.0],
            },
            &ranges,
        );

        device
            .end_command_buffer(cendre.command_buffers[0])
            .unwrap();

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

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(std::slice::from_ref(&cendre.swapchain))
            .image_indices(std::slice::from_ref(&image_index))
            .wait_semaphores(&release_semaphores);
        cendre
            .swapchain_loader
            .queue_present(cendre.present_queue, &present_info)
            .expect("Failed to queue present");

        device.device_wait_idle().unwrap();
    }
}

fn init_vulkan(
    mut commands: Commands,
    windows: Query<Entity, With<Window>>,
    winit_windows: NonSendMut<WinitWindows>,
) {
    let winit_window = windows
        .get_single()
        .ok()
        .and_then(|window_id| winit_windows.get_window(window_id))
        .expect("Failed to get winit window");

    unsafe {
        let entry = Entry::linked();
        let instance =
            create_instance(&entry, "Cendre", winit_window).expect("Failed to create instance");

        let physical_device =
            select_physical_device(&instance).expect("No physical device available");

        let surface = ash_window::create_surface(
            &entry,
            &instance,
            winit_window.raw_display_handle(),
            winit_window.raw_window_handle(),
            None,
        )
        .expect("Failed to create surface");
        let surface_loader = Surface::new(&entry, &instance);

        let queue_family_index =
            get_queue_family_index(&instance, &physical_device, &surface, &surface_loader)
                .expect("Failed to find queue family index") as u32;

        let device = create_device(&instance, &physical_device, queue_family_index)
            .expect("Failed to create device");

        let swapchain_loader = Swapchain::new(&instance, &device);
        let surface_format = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("failed to get surface formats")[0];

        let swapchain = create_swapchain(
            &physical_device,
            &surface_loader,
            &surface,
            &swapchain_loader,
            &surface_format,
            winit_window.inner_size().width,
            winit_window.inner_size().height,
        )
        .expect("Failed to create swapchain");

        let command_pool = create_command_pool(&device, queue_family_index);

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = device.allocate_command_buffers(&allocate_info).unwrap();

        let acquire_semaphore = create_semaphore(&device).expect("Failed to create semaphore");
        let release_semaphore = create_semaphore(&device).expect("Failed to create semaphore");
        let present_queue = device.get_device_queue(queue_family_index, 0);

        let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
        let present_image_views = present_images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                device.create_image_view(&create_view_info, None).unwrap()
            })
            .collect();

        commands.insert_resource(CendreRenderer {
            instance,
            swapchain_loader,
            swapchain,
            surface_format,
            acquire_semaphore,
            release_semaphore,
            present_queue,
            present_images,
            present_image_views,
            command_pool,
            command_buffers,
        });
        commands.insert_resource(CendreDevice(device));
    };
}

unsafe fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
    let pool_create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);
    device.create_command_pool(&pool_create_info, None).unwrap()
}

unsafe fn create_semaphore(device: &Device) -> anyhow::Result<vk::Semaphore> {
    let semaphore_create_info = vk::SemaphoreCreateInfo::default();
    Ok(device.create_semaphore(&semaphore_create_info, None)?)
}

unsafe fn create_swapchain(
    physical_device: &vk::PhysicalDevice,
    surface_loader: &Surface,
    surface: &vk::SurfaceKHR,
    swapchain_loader: &Swapchain,
    surface_format: &SurfaceFormatKHR,
    width: u32,
    height: u32,
) -> anyhow::Result<vk::SwapchainKHR> {
    let present_modes = surface_loader
        .get_physical_device_surface_present_modes(*physical_device, *surface)
        .unwrap();
    let present_mode = present_modes
        .iter()
        .cloned()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(*surface)
        .min_image_count(2)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(vk::Extent2D { width, height })
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .present_mode(present_mode);
    Ok(swapchain_loader.create_swapchain(&swapchain_create_info, None)?)
}

unsafe fn create_instance(
    entry: &Entry,
    app_name: &str,
    winit_window: &winit::window::Window,
) -> anyhow::Result<Instance> {
    let app_name = CString::new(app_name.to_string())?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 1, 0));

    let debug_layers_raw: Vec<*const c_char> = [
        #[cfg(debug_assertions)]
        {
            CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0")
        },
    ]
    .iter()
    .map(|raw_name: &&CStr| raw_name.as_ptr())
    .collect();

    let extension_names =
        ash_window::enumerate_required_extensions(winit_window.raw_display_handle())?.to_vec();

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(vk::InstanceCreateFlags::default())
        .enabled_layer_names(&debug_layers_raw)
        .enabled_extension_names(&extension_names);

    Ok(entry.create_instance(&create_info, None)?)
}

unsafe fn select_physical_device(instance: &Instance) -> Option<vk::PhysicalDevice> {
    let physical_devices = instance
        .enumerate_physical_devices()
        .expect("Failed to enumerate devices");

    for physical_device in &physical_devices {
        let props = instance.get_physical_device_properties(*physical_device);
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            let device_name = c_char_buf_to_string(props.device_name.as_ptr());
            info!("Using discrete GPU {}", device_name);
            return Some(*physical_device);
        }
    }

    // Discrete gpu not found
    if !physical_devices.is_empty() {
        let fallback_device = physical_devices[0];
        let props = instance.get_physical_device_properties(fallback_device);
        let device_name = c_char_buf_to_string(props.device_name.as_ptr());
        info!("Using fallback GPU {}", device_name);
        return Some(fallback_device);
    }
    None
}

unsafe fn create_device(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
    queue_family_index: u32,
) -> anyhow::Result<Device> {
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0]);

    let extension_names = [Swapchain::NAME.as_ptr()];
    let features = vk::PhysicalDeviceFeatures::default();

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&extension_names)
        .enabled_features(&features);

    Ok(instance.create_device(*physical_device, &device_create_info, None)?)
}

unsafe fn get_queue_family_index(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
    surface: &vk::SurfaceKHR,
    surface_loader: &Surface,
) -> Option<usize> {
    for (index, props) in instance
        .get_physical_device_queue_family_properties(*physical_device)
        .iter()
        .enumerate()
    {
        let supports_graphic_and_surface = props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && surface_loader
                .get_physical_device_surface_support(*physical_device, index as u32, *surface)
                .is_ok();
        if supports_graphic_and_surface {
            return Some(index);
        }
    }
    None
}

unsafe fn c_char_buf_to_string<'a>(buf: *const c_char) -> Cow<'a, str> {
    unsafe { CStr::from_ptr(buf) }.to_string_lossy()
}
