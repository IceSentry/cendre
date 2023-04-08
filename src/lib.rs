#![warn(clippy::pedantic)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]

use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    os::raw::c_char,
};

use anyhow::bail;
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    Device,
};
use ash::{vk, Entry, Instance};
use bevy::{prelude::*, winit::WinitWindows};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

pub struct CendrePlugin;
impl Plugin for CendrePlugin {
    fn build(&self, app: &mut App) {
        app
            // .add_startup_system(hide_window)
            // .add_startup_system(init_vulkan.after(hide_window))
            // .add_startup_system(show_window.after(init_vulkan))
            .add_startup_system(init_vulkan)
            .add_system(update);
    }
}

// fn hide_window(windows: Query<Entity, With<Window>>, winit_windows: NonSendMut<WinitWindows>) {
//     let winit_window = windows
//         .get_single()
//         .ok()
//         .and_then(|window_id| winit_windows.get_window(window_id))
//         .expect("Failed to get winit window");
//     winit_window.set_visible(false);
// }

// fn show_window(windows: Query<Entity, With<Window>>, winit_windows: NonSendMut<WinitWindows>) {
//     let winit_window = windows
//         .get_single()
//         .ok()
//         .and_then(|window_id| winit_windows.get_window(window_id))
//         .expect("Failed to get winit window");
//     winit_window.set_visible(true);
// }

unsafe fn create_render_pass(
    device: &Device,
    surface_format: vk::SurfaceFormatKHR,
) -> anyhow::Result<vk::RenderPass> {
    let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpass = vk::SubpassDescription::default()
        .color_attachments(&color_attachment_refs)
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
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);
    Ok(device.create_render_pass(&create_info, None)?)
}

unsafe fn create_frame_buffer(
    device: &Device,
    render_pass: vk::RenderPass,
    image_view: vk::ImageView,
    width: u32,
    height: u32,
) -> anyhow::Result<vk::Framebuffer> {
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(std::slice::from_ref(&image_view))
        .width(width)
        .height(height)
        .layers(1);
    Ok(device.create_framebuffer(&create_info, None)?)
}

unsafe fn create_image_view(
    device: &Device,
    surface_format: vk::SurfaceFormatKHR,
    image: vk::Image,
) -> anyhow::Result<vk::ImageView> {
    let create_view_info = vk::ImageViewCreateInfo::default()
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(surface_format.format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::R,
            g: vk::ComponentSwizzle::G,
            b: vk::ComponentSwizzle::B,
            a: vk::ComponentSwizzle::A,
        })
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1),
        )
        .image(image);
    Ok(device.create_image_view(&create_view_info, None)?)
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
    swapchain_loader: &Swapchain,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
) -> anyhow::Result<vk::SwapchainKHR> {
    let surface_capabilities =
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;

    let mut desired_image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0
        && desired_image_count > surface_capabilities.max_image_count
    {
        desired_image_count = surface_capabilities.max_image_count;
    }

    let surface_resolution = match surface_capabilities.current_extent.width {
        std::u32::MAX => vk::Extent2D { width, height },
        _ => surface_capabilities.current_extent,
    };

    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    let composite_alpha = match surface_capabilities.supported_composite_alpha {
        vk::CompositeAlphaFlagsKHR::OPAQUE
        | vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
        | vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED => {
            surface_capabilities.supported_composite_alpha
        }
        _ => vk::CompositeAlphaFlagsKHR::INHERIT,
    };

    let present_mode = surface_loader
        .get_physical_device_surface_present_modes(physical_device, surface)?
        .iter()
        .copied()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(desired_image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(surface_resolution)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(composite_alpha)
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

    let mut extension_names =
        ash_window::enumerate_required_extensions(winit_window.raw_display_handle())?.to_vec();
    #[cfg(debug_assertions)]
    {
        extension_names.push(DebugUtils::NAME.as_ptr());
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(vk::InstanceCreateFlags::default())
        .enabled_layer_names(&debug_layers_raw)
        .enabled_extension_names(&extension_names);

    Ok(entry.create_instance(&create_info, None)?)
}

unsafe fn select_physical_device(
    instance: &Instance,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
) -> Option<(vk::PhysicalDevice, usize)> {
    let physical_devices = instance
        .enumerate_physical_devices()
        .expect("Failed to enumerate devices");

    let mut fallback = None;
    for physical_device in &physical_devices {
        let props = instance.get_physical_device_properties(*physical_device);

        let Some(queue_family_index) =
            get_queue_family_index(instance, surface_loader, surface, *physical_device)
        else {
            continue;
        };

        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            let device_name = c_char_buf_to_string(props.device_name.as_ptr());
            info!("Using discrete GPU {:?}", device_name);
            return Some((*physical_device, queue_family_index));
        }

        if fallback.is_none() {
            fallback = Some((*physical_device, queue_family_index));
        }
    }
    if let Some((physical_device, _)) = fallback {
        let props = instance.get_physical_device_properties(physical_device);
        let device_name = c_char_buf_to_string(props.device_name.as_ptr());
        info!("Using fallback device {:?}", device_name);
    }
    fallback
}

unsafe fn get_queue_family_index(
    instance: &Instance,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Option<usize> {
    for (index, props) in instance
        .get_physical_device_queue_family_properties(physical_device)
        .iter()
        .enumerate()
    {
        let supports_graphic_and_surface = props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && surface_loader
                .get_physical_device_surface_support(physical_device, index as u32, surface)
                .is_ok();
        if supports_graphic_and_surface {
            return Some(index);
        }
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

unsafe fn load_shader(device: &Device, path: &str) -> anyhow::Result<vk::ShaderModule> {
    let mut shader_file = std::fs::File::open(path)?;
    let spv = ash::util::read_spv(&mut shader_file)?;

    let create_info = vk::ShaderModuleCreateInfo::default().code(&spv);
    Ok(device.create_shader_module(&create_info, None)?)
}

unsafe fn create_pipeline_layout(device: &Device) -> anyhow::Result<vk::PipelineLayout> {
    let create_info = vk::PipelineLayoutCreateInfo::default();
    Ok(device.create_pipeline_layout(&create_info, None)?)
}

unsafe fn create_graphics_pipeline(
    device: &Device,
    pipeline_cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
) -> anyhow::Result<vk::Pipeline> {
    let vertex_name = CString::new("vertex".to_string()).unwrap();
    let fragment_name = CString::new("fragment".to_string()).unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(&vertex_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(&fragment_name),
    ];

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

    let color_attachment_state = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&color_attachment_state));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(render_pass);
    let graphics_pipelines = match device.create_graphics_pipelines(
        pipeline_cache,
        std::slice::from_ref(&create_info),
        None,
    ) {
        Ok(pipelines) => pipelines,
        // For some reason the Err is a tuple and doesn't work with anyhow
        Err((_, result)) => bail!("Failed to create graphics pipelines.\n{result:?}"),
    };
    Ok(graphics_pipelines[0])
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let message_format =
        format!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n");
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => trace!("{message_format}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => info!("{message_format}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!("{message_format}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!("{message_format}");
            panic!("VALIDATION ERROR");
        }
        _ => panic!("Unknown message severity"),
    }

    vk::FALSE
}

unsafe fn init_debug_utils_messenger(
    debug_utils: &DebugUtils,
) -> anyhow::Result<vk::DebugUtilsMessengerEXT> {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    Ok(debug_utils.create_debug_utils_messenger(&debug_info, None)?)
}

unsafe fn get_surface_format(
    surface_loader: &Surface,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let formats = surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
    if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        Ok(vk::SurfaceFormatKHR::default().format(vk::Format::R8G8B8A8_UNORM))
    } else if let Some(format) = formats.iter().find(|format| {
        format.format == vk::Format::R8G8B8A8_UNORM || format.format == vk::Format::B8G8R8A8_UNORM
    }) {
        Ok(*format)
    } else {
        Ok(formats[0])
    }
}

fn image_barrier<'a>(
    image: vk::Image,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> vk::ImageMemoryBarrier<'a> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .layer_count(vk::REMAINING_ARRAY_LAYERS);

    vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access_mask)
        .old_layout(old_layout)
        .dst_access_mask(dst_access_mask)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
}

#[derive(Resource)]
struct CendreRenderer {
    instance: Instance,
    device: Device,
    debug_utils: DebugUtils,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    acquire_semaphore: vk::Semaphore,
    release_semaphore: vk::Semaphore,
    present_queue: vk::Queue,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_cache: vk::PipelineCache,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    triangle_vs: vk::ShaderModule,
    triangle_fs: vk::ShaderModule,
}

impl CendreRenderer {
    #[allow(clippy::too_many_lines)]
    unsafe fn init(winit_window: &winit::window::Window) -> Self {
        let entry = Entry::linked();
        let instance =
            create_instance(&entry, "Cendre", winit_window).expect("Failed to create instance");

        let debug_utils = DebugUtils::new(&entry, &instance);
        let debug_utils_messenger =
            init_debug_utils_messenger(&debug_utils).expect("Failed to init debug utils messenger");

        let surface = ash_window::create_surface(
            &entry,
            &instance,
            winit_window.raw_display_handle(),
            winit_window.raw_window_handle(),
            None,
        )
        .expect("Failed to create surface");
        let surface_loader = Surface::new(&entry, &instance);

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface_loader, surface).expect("No GPU found");

        let device = create_device(&instance, &physical_device, queue_family_index as u32)
            .expect("Failed to create device");

        let swapchain_loader = Swapchain::new(&instance, &device);
        let surface_format = get_surface_format(&surface_loader, physical_device, surface)
            .expect("Failed to get a surface format");

        let swapchain = create_swapchain(
            &swapchain_loader,
            &surface_loader,
            surface,
            surface_format,
            physical_device,
            winit_window.inner_size().width,
            winit_window.inner_size().height,
        )
        .expect("Failed to create swapchain");

        let command_pool = create_command_pool(&device, queue_family_index as u32);

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = device.allocate_command_buffers(&allocate_info).unwrap();

        let acquire_semaphore = create_semaphore(&device).expect("Failed to create semaphore");
        let release_semaphore = create_semaphore(&device).expect("Failed to create semaphore");
        let present_queue = device.get_device_queue(queue_family_index as u32, 0);

        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images
            .iter()
            .map(|&image| create_image_view(&device, surface_format, image).unwrap())
            .collect();

        let triangle_vs = load_shader(&device, "assets/shaders/triangle.vert.spv")
            .expect("Failed to load triangle vertex shader");
        let triangle_fs = load_shader(&device, "assets/shaders/triangle.frag.spv")
            .expect("Failed to load triangle fragment shader");

        let create_info = vk::PipelineCacheCreateInfo::default();
        let pipeline_cache = device.create_pipeline_cache(&create_info, None).unwrap();

        let render_pass = create_render_pass(&device, surface_format).unwrap();
        let pipeline_layout = create_pipeline_layout(&device).unwrap();
        let pipeline = create_graphics_pipeline(
            &device,
            pipeline_cache,
            pipeline_layout,
            render_pass,
            triangle_vs,
            triangle_fs,
        )
        .expect("Failed to create graphics pipeline");
        let framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                create_frame_buffer(
                    &device,
                    render_pass,
                    *image_view,
                    winit_window.inner_size().width,
                    winit_window.inner_size().height,
                )
                .expect("Failed to create frame buffer")
            })
            .collect();

        Self {
            instance,
            device,
            debug_utils,
            debug_utils_messenger,
            swapchain_loader,
            swapchain,
            surface_loader,
            surface,
            acquire_semaphore,
            release_semaphore,
            present_queue,
            swapchain_images,
            swapchain_image_views,
            command_pool,
            command_buffers,
            render_pass,
            framebuffers,
            pipeline_cache,
            pipeline,
            pipeline_layout,
            triangle_vs,
            triangle_fs,
        }
    }
}

impl Drop for CendreRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device.destroy_command_pool(self.command_pool, None);

            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }

            for image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }

            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);

            self.device.destroy_shader_module(self.triangle_vs, None);
            self.device.destroy_shader_module(self.triangle_fs, None);

            self.device.destroy_render_pass(self.render_pass, None);

            self.device.destroy_semaphore(self.acquire_semaphore, None);
            self.device.destroy_semaphore(self.release_semaphore, None);

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            self.device.destroy_device(None);

            self.surface_loader.destroy_surface(self.surface, None);

            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);

            self.instance.destroy_instance(None);
        }
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
        commands.insert_resource(CendreRenderer::init(winit_window));
    };
}

#[allow(clippy::too_many_lines)]
fn update(cendre: Res<CendreRenderer>, windows: Query<&Window>) {
    let window = windows.single();
    unsafe {
        let acquire_semaphores = [cendre.acquire_semaphore];
        let release_semaphores = [cendre.release_semaphore];
        let device = &cendre.device;

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

        let command_buffer = cendre.command_buffers[0];

        // BEGIN

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap();

        let render_begin_barrier = image_barrier(
            cendre.swapchain_images[image_index as usize],
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[render_begin_barrier],
        );

        // CLEAR

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.3, 0.3, 0.3, 1.0],
            },
        };
        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(cendre.render_pass)
            .framebuffer(cendre.framebuffers[image_index as usize])
            .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                width: window.physical_width(),
                height: window.physical_height(),
            }))
            .clear_values(std::slice::from_ref(&clear_color));
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );

        // DRAW

        let viewport = vk::Viewport::default()
            .width(window.physical_width() as f32)
            .height(window.physical_height() as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(
            vk::Extent2D::default()
                .width(window.physical_width())
                .height(window.physical_height()),
        );

        device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
        device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            cendre.pipeline,
        );
        device.cmd_draw(command_buffer, 3, 1, 0, 0);

        // END

        device.cmd_end_render_pass(command_buffer);

        let render_end_barrier = image_barrier(
            cendre.swapchain_images[image_index as usize],
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

        device.end_command_buffer(command_buffer).unwrap();

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

unsafe fn c_char_buf_to_string<'a>(buf: *const c_char) -> Cow<'a, str> {
    unsafe { CStr::from_ptr(buf) }.to_string_lossy()
}
