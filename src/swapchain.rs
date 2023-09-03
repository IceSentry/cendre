use ash::vk;
use ash::{
    extensions::khr::{Surface, Swapchain},
    Device,
};

// TODO could also hold present_queue and swapchain_loader
// This would make it possible to use it for submit/present
pub struct CendreSwapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub width: u32,
    pub height: u32,
    pub present_mode: vk::PresentModeKHR,
}

impl CendreSwapchain {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
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
        present_mode: vk::PresentModeKHR,
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

        // TODO use present mode from Window
        let present_mode = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
        }
        .unwrap()
        .iter()
        .copied()
        .find(|&mode| mode == present_mode)
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
            present_mode,
        }
    }

    #[must_use]
    pub fn resize(
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
        present_mode: vk::PresentModeKHR,
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
            present_mode,
        );
        unsafe { device.device_wait_idle().unwrap() };
        self.destroy(device, swapchain_loader);
        new_swapchain
    }

    // TODO consider using Arc<Mutex<T>> for stuff that needs to be destroyed
    // This would make it possible to have the swapchain on it's own
    pub fn destroy(&self, device: &Device, swapchain_loader: &Swapchain) {
        for framebuffer in &self.framebuffers {
            unsafe { device.destroy_framebuffer(*framebuffer, None) };
        }

        for image_view in &self.image_views {
            unsafe { device.destroy_image_view(*image_view, None) };
        }

        unsafe { swapchain_loader.destroy_swapchain(self.swapchain, None) };
    }
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
