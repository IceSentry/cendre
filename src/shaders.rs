use std::{ffi::OsStr, io::Write, path::PathBuf, time::Instant};

use anyhow::Context;
use ash::{vk, Device};
use bevy::prelude::*;
use naga::valid::{Capabilities, ValidationFlags, Validator};

pub fn load_vk_shader_module(device: &Device, path: &str) -> anyhow::Result<vk::ShaderModule> {
    info!("Loading {path:?}");

    let path_buf = PathBuf::from(path);
    let source = std::fs::read_to_string(path_buf.clone())
        .unwrap_or_else(|err| panic!("{err}\nFailed to read file at {path_buf:?}"));
    let file_name = path_buf.file_name().unwrap().to_string_lossy();

    let start = Instant::now();
    let spv = match path_buf.extension().and_then(OsStr::to_str).unwrap() {
        "wgsl" => compile_wgsl(&file_name, &source),
        "glsl" => compile_glsl(&file_name, &source),
        _ => unimplemented!(),
    };
    info!("Compiling {path:?} took {}ms", start.elapsed().as_millis());

    let create_info = vk::ShaderModuleCreateInfo::default().code(&spv);
    let module = unsafe { device.create_shader_module(&create_info, None)? };
    Ok(module)
}

fn compile_wgsl(file_name: &str, source: &str) -> Vec<u32> {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(err) => {
            error!("{file_name}: {err}");
            panic!("Invalid wgsl shader {file_name}");
        }
    };

    let module_info =
        match Validator::new(ValidationFlags::default(), Capabilities::all()).validate(&module) {
            Ok(module_info) => module_info,
            Err(err) => {
                error!("{file_name}: {err}");
                panic!("Shader validation error for {file_name}");
            }
        };

    naga::back::spv::write_vec(
        &module,
        &module_info,
        &naga::back::spv::Options {
            flags: naga::back::spv::WriterFlags::empty(),
            ..naga::back::spv::Options::default()
        },
        None,
    )
    .unwrap()
}

// This doesn't work because naga doesn't support uint8_t
fn _compile_glsl_naga(file_name: &str, source: &str) -> Vec<u32> {
    let mut frontend = naga::front::glsl::Frontend::default();
    let module = match frontend.parse(
        &naga::front::glsl::Options {
            stage: if file_name.contains(".vert") {
                naga::ShaderStage::Vertex
            } else if file_name.contains(".frag") {
                naga::ShaderStage::Fragment
            } else if file_name.contains(".mesh") {
                naga::ShaderStage::Mesh
            } else {
                todo!()
            },
            defines: std::collections::HashMap::default(),
        },
        source,
    ) {
        Ok(module) => module,
        Err(err) => {
            error!("{file_name}: {err:?}");
            panic!("Invalid glsl shader {file_name}");
        }
    };

    let module_info =
        match Validator::new(ValidationFlags::default(), Capabilities::all()).validate(&module) {
            Ok(module_info) => module_info,
            Err(err) => {
                error!("{file_name}: {err}");
                panic!("Shader validation error for {file_name}");
            }
        };

    naga::back::spv::write_vec(
        &module,
        &module_info,
        &naga::back::spv::Options {
            flags: naga::back::spv::WriterFlags::empty(),
            ..naga::back::spv::Options::default()
        },
        None,
    )
    .unwrap()
}

fn compile_glsl(file_name: &str, source: &str) -> Vec<u32> {
    let compiler = shaderc::Compiler::new().unwrap();
    let options = shaderc::CompileOptions::new().unwrap();
    let shader_kind = if file_name.contains(".vert") {
        shaderc::ShaderKind::Vertex
    } else if file_name.contains(".frag") {
        shaderc::ShaderKind::Fragment
    } else if file_name.contains(".mesh") {
        shaderc::ShaderKind::Mesh
    } else {
        todo!()
    };

    match compiler.compile_into_spirv(source, shader_kind, file_name, "main", Some(&options)) {
        Ok(result) => result.as_binary().to_vec(),
        Err(err) => {
            error!("{err}");
            panic!("Invalid glsl shader {file_name}");
        }
    }
}

fn _write_binary_file(path: &PathBuf, data: &[u8]) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path.clone())
        .context(format!("Failed to open {path:?}"))
        .unwrap();
    file.write_all(data).unwrap();
}
