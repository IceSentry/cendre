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
use bevy::prelude::*;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
    MemoryLocation,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use crate::{
    c_char_buf_to_string, image_barrier, shaders::load_vk_shader_module,
    swapchain::CendreSwapchain, RTX,
};

pub struct Buffer {
    vk_buffer: Arc<Mutex<vk::Buffer>>,
    allocation: Arc<Mutex<Option<Allocation>>>,
    pub size: u64,
}

impl Buffer {
    pub fn write(&mut self, data: &[u8]) {
        if let Some(allocation) = self.allocation.lock().unwrap().as_mut() {
            let slice = allocation.mapped_slice_mut().unwrap();
            slice[..data.len()].copy_from_slice(data);
        }
    }

    #[must_use]
    pub fn descriptor_info(&self, offset: vk::DeviceSize) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.vk_buffer())
            .offset(offset)
            .range(self.size)
    }

    #[must_use]
    pub fn write_descriptor<'a>(
        &'a self,
        dst_binding: u32,
        descriptor_type: vk::DescriptorType,
        buffer_info: &'a vk::DescriptorBufferInfo,
    ) -> vk::WriteDescriptorSet {
        vk::WriteDescriptorSet::default()
            .dst_binding(dst_binding)
            .descriptor_type(descriptor_type)
            .buffer_info(std::slice::from_ref(buffer_info))
    }

    #[must_use]
    pub fn vk_buffer(&self) -> vk::Buffer {
        *self.vk_buffer.lock().unwrap()
    }
}

pub struct PipelineLayout {
    vk_pipeline_layout: Arc<Mutex<vk::PipelineLayout>>,
    vk_descriptor_set_layout: Arc<Mutex<vk::DescriptorSetLayout>>,
}

impl PipelineLayout {
    #[must_use]
    pub fn vk_pipeline_layout(&self) -> vk::PipelineLayout {
        *self.vk_pipeline_layout.lock().unwrap()
    }

    #[must_use]
    pub fn vk_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        *self.vk_descriptor_set_layout.lock().unwrap()
    }
}

pub struct Pipeline {
    pub layout: PipelineLayout,
    vk_pipeline: Arc<Mutex<vk::Pipeline>>,
}

impl Pipeline {
    #[must_use]
    pub fn vk_pipeline(&self) -> vk::Pipeline {
        *self.vk_pipeline.lock().unwrap()
    }
}

pub struct Shader {
    vk_shader_module: Arc<Mutex<vk::ShaderModule>>,
    pub entry_point: CString,
    pub stage: vk::ShaderStageFlags,
}

impl Shader {
    #[must_use]
    pub fn create_info(&self) -> vk::PipelineShaderStageCreateInfo {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(self.stage)
            .module(*self.vk_shader_module.lock().unwrap())
            .name(&self.entry_point)
    }
}

#[derive(Resource)]
pub struct CendreInstance {
    pub instance: Instance,
    pub device: Device,
    pub push_descriptor: PushDescriptor,
    pub physical_device: vk::PhysicalDevice,
    pub mesh_shader: MeshShader,
    pub debug_utils: DebugUtils,
    pub debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    pub swapchain_loader: Swapchain,
    pub swapchain: CendreSwapchain,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub acquire_semaphore: vk::Semaphore,
    pub release_semaphore: vk::Semaphore,
    pub present_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub render_pass: vk::RenderPass,
    pub pipeline_cache: vk::PipelineCache,
    pipeline_layouts: Vec<Arc<Mutex<vk::PipelineLayout>>>,
    pipelines: Vec<Arc<Mutex<vk::Pipeline>>>,
    descriptor_set_layouts: Vec<Arc<Mutex<vk::DescriptorSetLayout>>>,
    shader_modules: Vec<Arc<Mutex<vk::ShaderModule>>>,
    allocator: Allocator,
    allocations: Vec<Arc<Mutex<Option<Allocation>>>>,
    buffers: Vec<Arc<Mutex<vk::Buffer>>>,
}

impl CendreInstance {
    #[allow(clippy::too_many_lines)]
    pub fn init(winit_window: &winit::window::Window) -> Self {
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

        let create_info = vk::PipelineCacheCreateInfo::default();
        let pipeline_cache = unsafe { device.create_pipeline_cache(&create_info, None).unwrap() };

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
            pipeline_layouts: vec![],
            pipelines: vec![],
            descriptor_set_layouts: vec![],
            shader_modules: vec![],
            allocator,
            allocations: vec![],
            buffers: vec![],
        }
    }

    pub fn create_buffer(
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
            vk_buffer: buffer_raw,
            allocation: allocation_raw,
            size,
        })
    }

    pub fn load_shader(
        &mut self,
        path: &str,
        entry_point: &str,
        stage: vk::ShaderStageFlags,
    ) -> Shader {
        let vk_shader_module = load_vk_shader_module(&self.device, path).unwrap();
        let vk_shader_module = Arc::new(Mutex::new(vk_shader_module));
        self.shader_modules.push(vk_shader_module.clone());
        let entry_point = CString::new(entry_point).unwrap();
        Shader {
            vk_shader_module,
            entry_point,
            stage,
        }
    }

    pub fn create_pipeline_layout(
        &mut self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> anyhow::Result<PipelineLayout> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(bindings);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        };

        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&create_info, None)? };

        let pipeline_layout = Arc::new(Mutex::new(pipeline_layout));
        self.pipeline_layouts.push(pipeline_layout.clone());

        let descriptor_set_layout = Arc::new(Mutex::new(descriptor_set_layout));
        self.descriptor_set_layouts
            .push(descriptor_set_layout.clone());

        Ok(PipelineLayout {
            vk_pipeline_layout: pipeline_layout,
            vk_descriptor_set_layout: descriptor_set_layout,
        })
    }

    pub fn create_graphics_pipeline(
        &mut self,
        pipeline_layout: PipelineLayout,
        render_pass: vk::RenderPass,
        stages: &[vk::PipelineShaderStageCreateInfo],
        primitive_topology: vk::PrimitiveTopology,
        rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    ) -> anyhow::Result<Pipeline> {
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly_state =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(primitive_topology);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

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
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(*pipeline_layout.vk_pipeline_layout.lock().unwrap())
            .render_pass(render_pass);
        let graphics_pipelines = match unsafe {
            self.device.create_graphics_pipelines(
                self.pipeline_cache,
                std::slice::from_ref(&create_info),
                None,
            )
        } {
            Ok(pipelines) => pipelines,
            // For some reason the Err is a tuple and doesn't work with anyhow
            Err((_, result)) => {
                error!("{result:?}");
                bail!("Failed to create graphics pipelines")
            }
        };
        let pipeline = Arc::new(Mutex::new(graphics_pipelines[0]));
        self.pipelines.push(pipeline.clone());
        Ok(Pipeline {
            vk_pipeline: pipeline,
            layout: pipeline_layout,
        })
    }

    #[must_use]
    pub fn begin_frame(&self) -> (u32, vk::CommandBuffer) {
        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain.swapchain,
                    0,
                    self.acquire_semaphore,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image")
        };

        unsafe {
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .expect("Failed to reset command_pool");
        }

        let command_buffer = self.command_buffers[0];

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        unsafe {
            let render_begin_barrier = image_barrier(
                self.swapchain.images[image_index as usize],
                vk::AccessFlags::empty(),
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[render_begin_barrier],
            );
        }
        (image_index, command_buffer)
    }

    pub fn end_frame(&self, image_index: u32, command_buffer: vk::CommandBuffer) {
        unsafe {
            let render_end_barrier = image_barrier(
                self.swapchain.images[image_index as usize],
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::empty(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            self.device.cmd_pipeline_barrier(
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
            self.device.end_command_buffer(command_buffer).unwrap();
        }

        self.submit();
        self.present(image_index);

        unsafe {
            self.device.device_wait_idle().unwrap();
        }
    }

    pub fn submit(&self) {
        let acquire_semaphores = [self.acquire_semaphore];
        let release_semaphores = [self.release_semaphore];
        unsafe {
            let submits = [vk::SubmitInfo::default()
                .wait_semaphores(&acquire_semaphores)
                .wait_dst_stage_mask(std::slice::from_ref(
                    &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ))
                .command_buffers(&self.command_buffers)
                .signal_semaphores(&release_semaphores)];
            self.device
                .queue_submit(self.present_queue, &submits, vk::Fence::null())
                .unwrap();
        }
    }

    pub fn present(&self, image_index: u32) {
        let release_semaphores = [self.release_semaphore];
        unsafe {
            let present_info = vk::PresentInfoKHR::default()
                .swapchains(std::slice::from_ref(&self.swapchain.swapchain))
                .image_indices(std::slice::from_ref(&image_index))
                .wait_semaphores(&release_semaphores);
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
                .expect("Failed to queue present");
        }
    }

    pub fn set_viewport(&self, command_buffer: vk::CommandBuffer, width: u32, height: u32) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: height as f32,
            width: width as f32,
            height: -(height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor =
            vk::Rect2D::default().extent(vk::Extent2D::default().width(width).height(height));

        unsafe {
            self.device
                .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
        };
        unsafe {
            self.device
                .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));
        };
    }
}

impl Drop for CendreInstance {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device.destroy_command_pool(self.command_pool, None);

            for buffer in &self.buffers {
                let buffer = buffer.lock().unwrap();
                self.device.destroy_buffer(*buffer, None);
            }
            for allocation in &self.allocations {
                let allocation = allocation.lock().unwrap().take().unwrap();
                self.allocator.free(allocation).unwrap();
            }

            self.swapchain.destroy(&self.device, &self.swapchain_loader);

            for pipeline in &self.pipelines {
                self.device
                    .destroy_pipeline(*pipeline.lock().unwrap(), None);
            }

            for pipeline_layout in &self.pipeline_layouts {
                self.device
                    .destroy_pipeline_layout(*pipeline_layout.lock().unwrap(), None);
            }
            for descriptor_set_layout in &self.descriptor_set_layouts {
                self.device
                    .destroy_descriptor_set_layout(*descriptor_set_layout.lock().unwrap(), None);
            }

            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);

            for module in &self.shader_modules {
                self.device
                    .destroy_shader_module(*module.lock().unwrap(), None);
            }

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
