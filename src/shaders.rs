use std::{
    ffi::{CString, OsStr},
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::Context;
use ash::{
    vk::{self, ShaderStageFlags},
    Device,
};
use bevy::prelude::*;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use shaderc::ResolvedInclude;

pub struct Shader {
    pub vk_shader_module: Arc<Mutex<vk::ShaderModule>>,
    pub entry_point: CString,
    pub stage: vk::ShaderStageFlags,
    pub storage_buffer_mask: u32,
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

pub fn compile_shader(path: &str) -> anyhow::Result<Vec<u32>> {
    let path_buf = PathBuf::from(path);
    if !path_buf.exists() {
        anyhow::bail!("Invalid shader path {path_buf:?}");
    }

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
    Ok(spv)
}

pub fn create_shader_module(device: &Device, spv: Vec<u32>) -> anyhow::Result<vk::ShaderModule> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(&spv);
    Ok(unsafe { device.create_shader_module(&create_info, None)? })
}

#[derive(Clone, Copy, Debug)]
struct Id {
    kind: IdKind,
    id_type: u32,
    storage_class: rspirv::spirv::StorageClass,
    binding: u32,
    set: u32,
}
impl Default for Id {
    fn default() -> Self {
        Self {
            kind: IdKind::Unknown,
            id_type: 0,
            storage_class: rspirv::spirv::StorageClass::Uniform,
            binding: 0,
            set: 0,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IdKind {
    Unknown,
    Variable,
}

#[derive(Clone)]
pub struct ParseInfo {
    pub stage: ShaderStageFlags,
    pub entry_point: String,
    pub storage_buffer_mask: u32,
}

impl std::fmt::Debug for ParseInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParseInfo")
            .field("stage", &self.stage)
            .field("entry_point", &self.entry_point)
            .field(
                "storage_buffer_mask",
                &format!("{:032b}", self.storage_buffer_mask),
            )
            .finish()
    }
}

#[allow(clippy::wildcard_imports)]
#[allow(clippy::match_on_vec_items)]
pub fn parse_spirv(code: &[u32]) -> anyhow::Result<ParseInfo> {
    use rspirv::{dr::Operand, spirv::*};

    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(code, &mut loader).unwrap();
    let module = loader.module();

    let mut parse_info = ParseInfo {
        stage: vk::ShaderStageFlags::empty(),
        entry_point: String::new(),
        storage_buffer_mask: 0,
    };

    let mut ids: Vec<Id> = vec![Id::default(); module.header.as_ref().unwrap().bound as usize];

    for entry_point in &module.entry_points {
        parse_info.stage = match entry_point.operands[0].unwrap_execution_model() {
            ExecutionModel::Vertex => ShaderStageFlags::VERTEX,
            ExecutionModel::Fragment => ShaderStageFlags::FRAGMENT,
            ExecutionModel::MeshNV => ShaderStageFlags::MESH_NV,
            x => panic!("Unsupported execution model {x:?}"),
        };
        parse_info.entry_point = entry_point.operands[2].unwrap_literal_string().to_string();
    }

    for annotation in &module.annotations {
        let Op::Decorate = annotation.class.opcode else {
            continue;
        };
        let id = annotation.operands[0].unwrap_id_ref();

        match annotation.operands[1] {
            Operand::Decoration(Decoration::DescriptorSet) => {
                ids[id as usize].set = annotation.operands[2].unwrap_literal_int32();
            }
            Operand::Decoration(Decoration::Binding) => {
                ids[id as usize].binding = annotation.operands[2].unwrap_literal_int32();
            }
            _ => {}
        }
    }

    for inst in module.all_inst_iter() {
        let Op::Variable = inst.class.opcode else {
            continue;
        };
        let id = inst.result_id.unwrap();
        ids[id as usize].kind = IdKind::Variable;
        ids[id as usize].id_type = inst.result_type.unwrap();
        ids[id as usize].storage_class = inst.operands[0].unwrap_storage_class();
    }

    for id in ids {
        if id.kind == IdKind::Variable
            && matches!(
                id.storage_class,
                StorageClass::Uniform | StorageClass::StorageBuffer
            )
        {
            // WARN right now we assume any buffers are StorageBuffer

            assert!(id.set == 0);
            assert!(id.binding < 32);
            assert!(parse_info.storage_buffer_mask & (1 << id.binding) == 0);

            parse_info.storage_buffer_mask |= 1 << id.binding;
        }
    }

    println!("{parse_info:?}");

    Ok(parse_info)
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
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_include_callback(|a, _, _, _| {
        // WARN this assumes flat imports and that imports are always inside assets/shaders/
        let path = format!("assets/shaders/{a}");
        let content = std::fs::read_to_string(path.clone())
            .expect("Tried to include a file that isn't in assets/shaders/");
        Ok(ResolvedInclude {
            resolved_name: path,
            content,
        })
    });
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
