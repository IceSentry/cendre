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

pub mod obj_loader;
pub mod optimized_mesh;

use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    os::raw::c_char,
    sync::{Arc, Mutex},
};

use anyhow::bail;
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{PushDescriptor, Surface, Swapchain},
        nv::MeshShader,
    },
    Device,
};
use ash::{vk, Entry, Instance};
use bevy::{prelude::*, window::WindowResized, winit::WinitWindows};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
    MemoryLocation,
};
use optimized_mesh::{prepare_mesh, IndexBuffer, MeshletBuffer, OptimizedMesh, VertexBuffer};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

const RTX: bool = false;

pub struct CendrePlugin;
impl Plugin for CendrePlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(init_vulkan)
            .add_system(resize.before(update))
            .add_system(prepare_mesh.before(update))
            .add_system(update);
    }
}

fn create_render_pass(
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
    Ok(unsafe { device.create_render_pass(&create_info, None)? })
}

fn create_frame_buffer(
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
    Ok(unsafe { device.create_framebuffer(&create_info, None)? })
}

fn create_image_view(
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
    Ok(unsafe { device.create_image_view(&create_view_info, None)? })
}

fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
    let pool_create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);
    unsafe { device.create_command_pool(&pool_create_info, None).unwrap() }
}

fn create_semaphore(device: &Device) -> anyhow::Result<vk::Semaphore> {
    let semaphore_create_info = vk::SemaphoreCreateInfo::default();
    Ok(unsafe { device.create_semaphore(&semaphore_create_info, None)? })
}

struct CendreSwapchain {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    width: u32,
    height: u32,
}

impl CendreSwapchain {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &Device,
        swapchain_loader: &Swapchain,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        render_pass: vk::RenderPass,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Self {
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

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

        let present_mode = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
        }
        .unwrap()
        .iter()
        .copied()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
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
        if let Some(old_swapchain) = old_swapchain {
            swapchain_create_info.old_swapchain = old_swapchain;
        }
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };
        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| create_image_view(device, surface_format, image).unwrap())
            .collect();

        let framebuffers = image_views
            .iter()
            .map(|image_view| {
                create_frame_buffer(device, render_pass, *image_view, width, height)
                    .expect("Failed to create frame buffer")
            })
            .collect();

        Self {
            swapchain,
            images,
            image_views,
            framebuffers,
            width,
            height,
        }
    }

    #[must_use]
    fn resize(
        &self,
        device: &Device,
        swapchain_loader: &Swapchain,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        render_pass: vk::RenderPass,
    ) -> Self {
        let new_swapchain = CendreSwapchain::new(
            device,
            swapchain_loader,
            surface_loader,
            surface,
            surface_format,
            physical_device,
            width,
            height,
            render_pass,
            Some(self.swapchain),
        );
        unsafe { device.device_wait_idle().unwrap() };
        self.destroy(device, swapchain_loader);
        new_swapchain
    }

    fn destroy(&self, device: &Device, swapchain_loader: &Swapchain) {
        for framebuffer in &self.framebuffers {
            unsafe { device.destroy_framebuffer(*framebuffer, None) };
        }

        for image_view in &self.image_views {
            unsafe { device.destroy_image_view(*image_view, None) };
        }

        unsafe { swapchain_loader.destroy_swapchain(self.swapchain, None) };
    }
}

fn create_instance(
    entry: &Entry,
    app_name: &str,
    winit_window: &winit::window::Window,
) -> anyhow::Result<ash::Instance> {
    let app_name = CString::new(app_name.to_string())?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 1, 0));

    let debug_layers_raw: Vec<*const c_char> = [
        #[cfg(debug_assertions)]
        unsafe {
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

    Ok(unsafe { entry.create_instance(&create_info, None)? })
}

fn select_physical_device(
    instance: &Instance,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
) -> Option<(vk::PhysicalDevice, usize)> {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate devices")
    };

    let mut fallback = None;
    for physical_device in &physical_devices {
        let props = unsafe { instance.get_physical_device_properties(*physical_device) };

        if props.api_version < vk::API_VERSION_1_1 {
            continue;
        }

        let Some(queue_family_index) =
            get_queue_family_index(instance, surface_loader, surface, *physical_device)
        else {
            continue;
        };

        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            let device_name = unsafe { c_char_buf_to_string(props.device_name.as_ptr()) };
            info!("Using discrete GPU {:?}", device_name);
            return Some((*physical_device, queue_family_index));
        }

        if fallback.is_none() {
            fallback = Some((*physical_device, queue_family_index));
        }
    }
    if let Some((physical_device, _)) = fallback {
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { c_char_buf_to_string(props.device_name.as_ptr()) };
        info!("Using fallback device {:?}", device_name);
    }
    fallback
}

fn get_queue_family_index(
    instance: &Instance,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Option<usize> {
    for (index, props) in
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .iter()
            .enumerate()
    {
        let supports_graphic_and_surface = props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && unsafe {
                surface_loader
                    .get_physical_device_surface_support(physical_device, index as u32, surface)
                    .is_ok()
            };
        if supports_graphic_and_surface {
            return Some(index);
        }
    }
    None
}

fn load_shader(device: &Device, path: &str) -> anyhow::Result<vk::ShaderModule> {
    info!("loading {path}");
    let mut shader_file = std::fs::File::open(path)?;
    let spv = ash::util::read_spv(&mut shader_file)?;

    let create_info = vk::ShaderModuleCreateInfo::default().code(&spv);
    Ok(unsafe { device.create_shader_module(&create_info, None)? })
}

fn create_pipeline_layout(
    device: &Device,
) -> anyhow::Result<(vk::PipelineLayout, [vk::DescriptorSetLayout; 1])> {
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

    let create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        .bindings(&bindings);

    let layouts = [unsafe {
        device
            .create_descriptor_set_layout(&create_info, None)
            .unwrap()
    }];
    let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

    // unsafe {
    //     device.destroy_descriptor_set_layout(layouts[0], None);
    // }

    Ok((
        unsafe { device.create_pipeline_layout(&create_info, None)? },
        layouts,
    ))
}

fn create_graphics_pipeline(
    device: &Device,
    pipeline_cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
) -> anyhow::Result<vk::Pipeline> {
    // TODO use naga directly here
    let vertex_name = CString::new("vertex".to_string()).unwrap();
    let mesh_name = CString::new("main".to_string()).unwrap();
    let fragment_name = CString::new("fragment".to_string()).unwrap();

    let stages = if RTX {
        [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MESH_NV)
                .module(vertex_shader)
                .name(&mesh_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(&fragment_name),
        ]
    } else {
        [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(&vertex_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(&fragment_name),
        ]
    };

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
    let graphics_pipelines = match unsafe {
        device.create_graphics_pipelines(pipeline_cache, std::slice::from_ref(&create_info), None)
    } {
        Ok(pipelines) => pipelines,
        // For some reason the Err is a tuple and doesn't work with anyhow
        Err((_, result)) => {
            error!("{result:?}");
            bail!("Failed to create graphics pipelines")
        }
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
            panic!("VULKAN VALIDATION ERROR");
        }
        _ => panic!("Unknown message severity"),
    }

    vk::FALSE
}

fn init_debug_utils_messenger(
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
    Ok(unsafe { debug_utils.create_debug_utils_messenger(&debug_info, None)? })
}

fn get_surface_format(
    surface_loader: &Surface,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface)? };
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

pub struct Buffer {
    buffer_raw: Arc<Mutex<vk::Buffer>>,
    allocation_raw: Arc<Mutex<Option<Allocation>>>,
    size: u64,
}

impl Buffer {
    fn write(&mut self, data: &[u8]) {
        // let mut allocation = self.allocation_raw.lock().unwrap();
        if let Some(allocation) = self.allocation_raw.lock().unwrap().as_mut() {
            let slice = allocation.mapped_slice_mut().unwrap();
            slice[..data.len()].copy_from_slice(data);
        }
    }

    fn info(&self, offset: vk::DeviceSize) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(*self.buffer_raw.lock().unwrap())
            .offset(offset)
            .range(self.size)
    }
}

#[derive(Resource)]
struct CendreInstance {
    instance: Instance,
    device: Device,
    push_descriptor: PushDescriptor,
    physical_device: vk::PhysicalDevice,
    mesh_shader: MeshShader,
    debug_utils: DebugUtils,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    swapchain_loader: Swapchain,
    swapchain: CendreSwapchain,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    acquire_semaphore: vk::Semaphore,
    release_semaphore: vk::Semaphore,
    present_queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    render_pass: vk::RenderPass,
    pipeline_cache: vk::PipelineCache,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    triangle_vs: vk::ShaderModule,
    triangle_fs: vk::ShaderModule,
    allocator: Allocator,
    allocations: Vec<Arc<Mutex<Option<Allocation>>>>,
    buffers: Vec<Arc<Mutex<vk::Buffer>>>,
    desciptor_set_layouts: [vk::DescriptorSetLayout; 1],
}

impl CendreInstance {
    #[allow(clippy::too_many_lines)]
    fn init(winit_window: &winit::window::Window) -> Self {
        let entry = Entry::linked();
        let instance =
            create_instance(&entry, "Cendre", winit_window).expect("Failed to create instance");

        let debug_utils = DebugUtils::new(&entry, &instance);
        let debug_utils_messenger =
            init_debug_utils_messenger(&debug_utils).expect("Failed to init debug utils messenger");

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                winit_window.raw_display_handle(),
                winit_window.raw_window_handle(),
                None,
            )
            .expect("Failed to create surface")
        };
        let surface_loader = Surface::new(&entry, &instance);

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface_loader, surface).expect("No GPU found");

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index as u32)
            .queue_priorities(&[1.0]);

        let extension_names = [
            Swapchain::NAME.as_ptr(),
            PushDescriptor::NAME.as_ptr(),
            MeshShader::NAME.as_ptr(),
            unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_16bit_storage\0").as_ptr() },
            unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_8bit_storage\0").as_ptr() },
        ];

        let mut features2 = vk::PhysicalDeviceFeatures2::default();

        let mut physical_device_buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default();

        let mut features_16bit_storage =
            vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);

        let mut features_8bit_storage = vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true);

        let mut mesh_shader_features_nv =
            vk::PhysicalDeviceMeshShaderFeaturesNV::default().mesh_shader(RTX);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&extension_names)
            .push_next(&mut features2)
            .push_next(&mut physical_device_buffer_device_address_features)
            .push_next(&mut features_16bit_storage)
            .push_next(&mut features_8bit_storage)
            .push_next(&mut mesh_shader_features_nv);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };

        info!("device created");

        let push_descriptor = PushDescriptor::new(&instance, &device);
        let mesh_shader = MeshShader::new(&instance, &device);

        let surface_format = get_surface_format(&surface_loader, physical_device, surface)
            .expect("Failed to get a surface format");

        info!("surface format: {surface_format:?}");

        let render_pass = create_render_pass(&device, surface_format).unwrap();

        info!("render pass created");

        let swapchain_loader = Swapchain::new(&instance, &device);
        let swapchain = CendreSwapchain::new(
            &device,
            &swapchain_loader,
            &surface_loader,
            surface,
            surface_format,
            physical_device,
            winit_window.inner_size().width,
            winit_window.inner_size().height,
            render_pass,
            None,
        );

        info!("swapchain created");

        let command_pool = create_command_pool(&device, queue_family_index as u32);

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        let acquire_semaphore = create_semaphore(&device).expect("Failed to create semaphore");
        let release_semaphore = create_semaphore(&device).expect("Failed to create semaphore");
        let present_queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        let meshlet_ms = load_shader(&device, "assets/shaders/meshlet.mesh.spv")
            .expect("Failed to load meshlet mesh shader");
        let triangle_vs = load_shader(&device, "assets/shaders/triangle.vert.spv")
            .expect("Failed to load triangle vertex shader");
        let triangle_fs = load_shader(&device, "assets/shaders/triangle.frag.spv")
            .expect("Failed to load triangle fragment shader");

        let create_info = vk::PipelineCacheCreateInfo::default();
        let pipeline_cache = unsafe { device.create_pipeline_cache(&create_info, None).unwrap() };

        let (pipeline_layout, desciptor_set_layouts) = create_pipeline_layout(&device).unwrap();
        let pipeline = if RTX {
            create_graphics_pipeline(
                &device,
                pipeline_cache,
                pipeline_layout,
                render_pass,
                meshlet_ms,
                triangle_fs,
            )
            .expect("Failed to create graphics pipeline")
        } else {
            create_graphics_pipeline(
                &device,
                pipeline_cache,
                pipeline_layout,
                render_pass,
                triangle_vs,
                triangle_fs,
            )
            .expect("Failed to create graphics pipeline")
        };
        info!("pipeline created");

        let buffer_device_address =
            physical_device_buffer_device_address_features.buffer_device_address == 1;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: default(),
            buffer_device_address,
        })
        .unwrap();

        Self {
            instance,
            device,
            push_descriptor,
            mesh_shader,
            physical_device,
            debug_utils,
            debug_utils_messenger,
            swapchain_loader,
            swapchain,
            surface_loader,
            surface,
            surface_format,
            acquire_semaphore,
            release_semaphore,
            present_queue,
            command_pool,
            command_buffers,
            render_pass,
            pipeline_cache,
            pipeline,
            pipeline_layout,
            triangle_vs,
            triangle_fs,
            allocator,
            allocations: vec![],
            buffers: vec![],
            desciptor_set_layouts,
        }
    }

    fn create_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> anyhow::Result<Buffer> {
        let vk_info = vk::BufferCreateInfo::default().size(size).usage(usage);
        let buffer = unsafe { self.device.create_buffer(&vk_info, None) }.unwrap();
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let allocation = self.allocator.allocate(&AllocationCreateDesc {
            name: &format!("usage: {usage:?} size: {size} "),
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        // Bind memory to the buffer
        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        };

        let buffer_raw = Arc::new(Mutex::new(buffer));
        let allocation_raw = Arc::new(Mutex::new(Some(allocation)));

        self.buffers.push(buffer_raw.clone());
        self.allocations.push(allocation_raw.clone());

        Ok(Buffer {
            buffer_raw,
            allocation_raw,
            size,
        })
    }
}

impl Drop for CendreInstance {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device.destroy_command_pool(self.command_pool, None);

            for layout in self.desciptor_set_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }

            for buffer in &self.buffers {
                let buffer = buffer.lock().unwrap();
                self.device.destroy_buffer(*buffer, None);
            }
            for allocation in &self.allocations {
                let allocation = allocation.lock().unwrap().take().unwrap();
                self.allocator.free(allocation).unwrap();
            }

            self.swapchain.destroy(&self.device, &self.swapchain_loader);

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

    commands.insert_resource(CendreInstance::init(winit_window));
}

#[derive(Resource)]
struct MeshletsSize(pub u32);

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

unsafe fn c_char_buf_to_string<'a>(buf: *const c_char) -> Cow<'a, str> {
    unsafe { CStr::from_ptr(buf) }.to_string_lossy()
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
