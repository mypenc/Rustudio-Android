//! What the GPU can do and what rustudio asks of it: the device limits
//! (only what the adapter offers), whether the renderer builds its reduced
//! binding set (adapters under 24 sampled textures per stage, and OpenGL),
//! weak-GPU mode (software adapters, OpenGL, old Intel graphics, or
//! downlevel limits: start at the lowest Automatic quality level) and which
//! adapter to pick.

use egui_wgpu::wgpu;

/// Sampled textures per stage the full binding set needs (the part pass's
/// globals, texture and glass groups: 13 + 3 + 2, rounded to wgpu's tier).
pub const FULL_SAMPLED_TEXTURES: u32 = 24;

#[derive(Clone, Debug)]
pub struct GpuCaps {
    pub name: String,
    pub vendor: u32,
    pub backend: wgpu::Backend,
    pub driver: String,
    pub driver_info: String,
    pub device_type: wgpu::DeviceType,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
    pub downlevel: wgpu::DownlevelFlags,
    /// R8Unorm storage textures (HBAO's visibility output); else it writes
    /// Rgba8Unorm.
    pub r8_storage: bool,
}

impl GpuCaps {
    /// The device made from `adapter`.
    pub fn of_adapter(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Self {
        let mut caps = Self::of_device(device);
        caps.downlevel = adapter.get_downlevel_capabilities().flags;
        caps.r8_storage = device.features().contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
            && adapter
                .get_texture_format_features(wgpu::TextureFormat::R8Unorm)
                .allowed_usages
                .contains(wgpu::TextureUsages::STORAGE_BINDING);
        caps
    }

    /// The device's limits and features with its adapter's info, where the
    /// adapter is not at hand: compute from its limits, R8 storage from the
    /// adapter-specific format feature.
    pub fn of_device(device: &wgpu::Device) -> Self {
        let info = device.adapter_info();
        let limits = device.limits();
        let mut downlevel = wgpu::DownlevelFlags::compliant();
        downlevel.set(wgpu::DownlevelFlags::COMPUTE_SHADERS, limits.max_compute_workgroups_per_dimension > 0);
        Self {
            name: info.name,
            vendor: info.vendor,
            backend: info.backend,
            driver: info.driver,
            driver_info: info.driver_info,
            device_type: info.device_type,
            r8_storage: info.backend != wgpu::Backend::Gl
                && device.features().contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES),
            limits,
            features: device.features(),
            downlevel,
        }
    }

    /// The renderer's main pass binds 13 + 3 + 2 textures in one stage; under
    /// 24 (OpenGL's 16) it drops the local light shadow atlas, the scene
    /// probe's blend target and the BRDF table (render.rs REDUCED_BINDINGS).
    /// OpenGL also numbers texture units across every group of a pipeline
    /// layout, whatever the stage, with 16 in all.
    pub fn reduced_bindings(&self) -> bool {
        self.backend == wgpu::Backend::Gl || self.limits.max_sampled_textures_per_shader_stage < FULL_SAMPLED_TEXTURES
    }

    /// Multisampled textures the shaders read (the 3D view's MSAA colour,
    /// depth and see-through targets). wgpu-hal's OpenGL backend makes a
    /// sampled multisampled texture a GL_TEXTURE_2D (it has no
    /// GL_TEXTURE_2D_MULTISAMPLE path), so every pass drawing into one is an
    /// incomplete framebuffer: on OpenGL the view draws one sample a pixel
    /// (as Studio does through quality level 16), which also leaves out the
    /// 4x-only HBAO.
    pub fn sampled_msaa(&self) -> bool {
        self.backend != wgpu::Backend::Gl
    }

    /// Compute shaders (skinning, HBAO, light grid builds).
    pub fn compute(&self) -> bool {
        self.downlevel.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            && self.limits.max_compute_workgroups_per_dimension > 0
            && self.limits.max_compute_invocations_per_workgroup >= 64
    }


    /// Why this adapter runs in weak-GPU mode, if it does (before the
    /// RUSTUDIO_LOWEND override).
    pub fn weak_reason(&self) -> Option<String> {
        if self.device_type == wgpu::DeviceType::Cpu {
            return Some("software adapter".into());
        }
        if self.backend == wgpu::Backend::Gl {
            return Some("OpenGL backend".into());
        }
        if old_intel(self.vendor, &self.name) {
            return Some("pre-Skylake Intel graphics".into());
        }
        if self.limits.max_sampled_textures_per_shader_stage < FULL_SAMPLED_TEXTURES
            || self.limits.max_texture_dimension_2d < 8192
        {
            return Some(format!(
                "downlevel limits ({} sampled textures per stage, {} px textures)",
                self.limits.max_sampled_textures_per_shader_stage, self.limits.max_texture_dimension_2d
            ));
        }
        None
    }
}

/// Intel's Gen7/Gen7.5/Gen8 graphics (HD 2500/4000, HD 4200-5200 Haswell,
/// HD 5300-6300 and Iris 5100-6200 Broadwell, Bay Trail's unnumbered "HD
/// Graphics"): no Vulkan on Windows, DX12 only on old Haswell drivers.
/// Skylake's HD 5xx and every UHD / Iris Xe / Arc are not old.
pub fn old_intel(vendor: u32, name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if vendor != 0x8086 && !lower.contains("intel") {
        return false;
    }
    if lower.contains("uhd") || lower.contains("xe") || lower.contains("arc") {
        return false;
    }
    // Linux names the chip: "Intel(R) HD Graphics 4600 (HSW GT2)", "Mesa
    // Intel(R) HD Graphics 4600 (HSW GT2)", "... (BDW GT2)", "(IVB GT2)".
    for gen in ["(hsw", "(ivb", "(bdw", "(byt", "(chv", "(snb", "(ilk", "hsw gt", "ivb gt", "bdw gt"] {
        if lower.contains(gen) {
            return true;
        }
    }
    if !(lower.contains("hd graphics") || lower.contains("iris")) {
        return false;
    }
    let number: Option<u32> = lower
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .find(|n| *n >= 100);
    match number {
        // "Intel(R) HD Graphics" alone: Bay Trail / Cherry Trail / Ivy Bridge.
        None => lower.contains("hd graphics"),
        Some(n) => (2000..=6300).contains(&n),
    }
}

/// The limits rustudio asks of `adapter`: wgpu's defaults with 24 sampled
/// textures per stage and 8192-pixel textures where the adapter has them
/// (every Vulkan / DX12 / Metal GPU of the last decade: unchanged), else
/// the same with each limit lowered to what the adapter offers, so the
/// device never fails for a limit (OpenGL's 16 textures, WARP's, ...).
pub fn required_limits(adapter_limits: &wgpu::Limits) -> wgpu::Limits {
    let full = wgpu::Limits {
        max_sampled_textures_per_shader_stage: FULL_SAMPLED_TEXTURES,
        max_texture_dimension_2d: 8192,
        ..wgpu::Limits::default()
    };
    if full.check_limits(adapter_limits) {
        full
    } else {
        full.or_worse_values_from(adapter_limits)
    }
}

/// RUSTUDIO_LOWEND=1 forces weak-GPU mode, =0 turns it off.
pub fn lowend_override() -> Option<bool> {
    match std::env::var("RUSTUDIO_LOWEND").ok()?.trim() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// Decide weak-GPU mode for this device and log it under `rustudio::gpu`.
/// Returns whether it is on.
pub fn decide_lowend(caps: &GpuCaps) -> bool {
    let reason = caps.weak_reason();
    let forced = lowend_override();
    let on = forced.unwrap_or(reason.is_some());
    let why = match (forced, &reason) {
        (Some(true), _) => "RUSTUDIO_LOWEND=1".to_string(),
        (Some(false), Some(r)) => format!("RUSTUDIO_LOWEND=0 (would be on: {r})"),
        (Some(false), None) => "RUSTUDIO_LOWEND=0".to_string(),
        (None, Some(r)) => r.clone(),
        (None, None) => "capable GPU".to_string(),
    };
    if on {
        log::info!(
            target: "rustudio::gpu",
            "weak-GPU mode on ({why}): Automatic quality from level 1 (no HBAO, sun or light shadows, clouds, grass, depth of field, bloom or scene probe until frame times allow); RUSTUDIO_LOWEND=0 turns it off"
        );
    } else {
        log::info!(target: "rustudio::gpu", "weak-GPU mode off ({why})");
    }
    if !caps.sampled_msaa() {
        log::info!(target: "rustudio::gpu", "no sampled multisample textures ({:?}): 3D view without MSAA", caps.backend);
    }
    if !caps.compute() {
        log::info!(target: "rustudio::gpu", "no compute shaders: skinned meshes drawn at rest");
    }
    if caps.reduced_bindings() {
        log::info!(
            target: "rustudio::gpu",
            "reduced shader bindings ({:?}, {} sampled textures per stage): no local light shadows, scene probe blending or BRDF table (analytic fit)",
            caps.backend, caps.limits.max_sampled_textures_per_shader_stage
        );
    }
    on
}

/// Rank for picking an adapter: a real GPU before a software one, a native
/// API before OpenGL, discrete before integrated. With
/// WGPU_FORCE_FALLBACK_ADAPTER=1, software adapters rank first.
pub fn adapter_rank(info: &wgpu::AdapterInfo, force_fallback: bool) -> u32 {
    let cpu = info.device_type == wgpu::DeviceType::Cpu
        || info.name.contains("Microsoft Basic Render Driver")
        || info.name.to_ascii_lowercase().contains("llvmpipe");
    let device = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    let api = if info.backend == wgpu::Backend::Gl { 1 } else { 0 };
    let soft = if cpu != force_fallback { 1 } else { 0 };
    soft * 100 + api * 10 + device
}

/// WGPU_FORCE_FALLBACK_ADAPTER=1: pick the software adapter (WARP on
/// Windows DX12, llvmpipe on Linux) to test it.
pub fn force_fallback() -> bool {
    matches!(std::env::var("WGPU_FORCE_FALLBACK_ADAPTER").ok().as_deref(), Some("1" | "true"))
}

/// The adapters in the order to try them: surface-compatible ones ranked
/// by [`adapter_rank`] (so on Windows a real GPU through OpenGL beats WARP,
/// the only DX12 adapter of an HD 4600 without a DX12 driver).
pub fn rank_adapters(adapters: &[wgpu::Adapter], surface: Option<&wgpu::Surface<'_>>) -> Vec<wgpu::Adapter> {
    let fallback = force_fallback();
    let mut ranked: Vec<(u32, usize, wgpu::Adapter)> = adapters
        .iter()
        .enumerate()
        .filter(|(_, a)| surface.is_none_or(|s| a.is_surface_supported(s)))
        .map(|(i, a)| (adapter_rank(&a.get_info(), fallback), i, a.clone()))
        .collect();
    ranked.sort_by_key(|(rank, i, _)| (*rank, *i));
    ranked.into_iter().map(|(_, _, a)| a).collect()
}

/// OpenGL: GLSL has no `textureLoad` from depth textures (naga's GLSL
/// writer refuses it), so there depth textures are bound as unfilterable
/// float textures (WebGPU allows both for a depth format) and the shaders
/// read `.x` (see [`depth_as_float`]).
pub fn depth_loads_as_float(device: &wgpu::Device) -> bool {
    device.adapter_info().backend == wgpu::Backend::Gl
}

/// OpenGL: wgpu-hal picks a texture's GL target from its layer count, so a
/// one-layer texture meant to be viewed as `texture_2d_array` becomes a
/// GL_TEXTURE_2D and the array view of it reads nothing. Such textures get
/// a second (unused) layer there.
pub fn single_layer_arrays_break(device: &wgpu::Device) -> bool {
    device.adapter_info().backend == wgpu::Backend::Gl
}

/// The sample type for a depth texture binding read with `textureLoad`.
pub fn depth_sample_type(device: &wgpu::Device) -> wgpu::TextureSampleType {
    if depth_loads_as_float(device) {
        wgpu::TextureSampleType::Float { filterable: false }
    } else {
        wgpu::TextureSampleType::Depth
    }
}

/// `src` for this device: unchanged except on OpenGL, where its depth
/// textures become float ones ([`depth_as_float`]) and each texture keeps
/// one sampler (GLSL combines them; naga refuses a texture sampled with two
/// in one entry point): the materials' low-quality sampler (anisotropy 2)
/// becomes the studs sampler.
pub fn shader_for(device: &wgpu::Device, src: &str) -> String {
    let src = if depth_loads_as_float(device) { gl_shader(src) } else { src.to_owned() };
    let src = if uniform_arrays(device) { num_levels_as_size(&uniform_array_shader(device, &src)) } else { src };
    chunk_mat_params(device, &src)
}

/// WebGL2 (GLES 3.0) has no storage buffers: `max_storage_buffers_per_shader_stage`
/// is 0. The read-only light arrays then become fixed-size uniform arrays.
pub fn uniform_arrays(device: &wgpu::Device) -> bool {
    device.limits().max_storage_buffers_per_shader_stage == 0
}

/// Size of one `Light` (6 x vec4) and one `ShadowSlot` (6 x mat4x4 + vec4), in bytes.
const LIGHT_BYTES: usize = 96;
const SLOT_BYTES: usize = 6 * 64 + 16;

/// Local lights drawn per frame: the full set, or what one uniform block holds.
pub fn max_lights(device: &wgpu::Device) -> usize {
    let full = rustudio_core::mech::MAX_LIGHTS;
    if uniform_arrays(device) { full.min(device.limits().max_uniform_buffer_binding_size as usize / LIGHT_BYTES) } else { full }
}

/// Shadow-casting local lights at once: likewise.
pub fn max_lshadow(device: &wgpu::Device) -> usize {
    let full = rustudio_core::mech::LSHADOW_MAX_LIGHTS;
    if uniform_arrays(device) { full.min(device.limits().max_uniform_buffer_binding_size as usize / SLOT_BYTES) } else { full }
}

/// Binding type for the light arrays: storage where it exists, else uniform.
pub fn array_binding(device: &wgpu::Device) -> wgpu::BufferBindingType {
    if uniform_arrays(device) { wgpu::BufferBindingType::Uniform } else { wgpu::BufferBindingType::Storage { read_only: true } }
}

/// Buffer usage matching [`array_binding`].
pub fn array_usage(device: &wgpu::Device) -> wgpu::BufferUsages {
    (if uniform_arrays(device) { wgpu::BufferUsages::UNIFORM } else { wgpu::BufferUsages::STORAGE }) | wgpu::BufferUsages::COPY_DST
}

/// GLSL ES 3.00 (WebGL2) has no `textureQueryLevels`: `textureNumLevels(t)` becomes
/// the length of a full mip chain for the texture's width (the prefiltered cubes
/// that call it carry one).
fn num_levels_as_size(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("textureNumLevels(") {
        out.push_str(&rest[..i]);
        let after = &rest[i + "textureNumLevels(".len()..];
        let end = after.find(')').unwrap_or(0);
        out.push_str(&format!("(u32(log2(f32(textureDimensions({}).x))) + 1u)", &after[..end]));
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

/// shader.wgsl's `mat_params` table: 2560 vec4s.
const MAT_PARAMS_BYTES: u64 = 2560 * 16;

/// Bytes of the material table bound per uniform block (a multiple of 256, the
/// offset alignment, and at most the backend's uniform binding limit).
pub fn mat_chunk_bytes(device: &wgpu::Device) -> u64 {
    (device.limits().max_uniform_buffer_binding_size as u64 & !255).clamp(256, MAT_PARAMS_BYTES)
}

/// How many uniform blocks the material table needs (1 except on WebGL2).
pub fn mat_chunks(device: &wgpu::Device) -> u32 {
    MAT_PARAMS_BYTES.div_ceil(mat_chunk_bytes(device)) as u32
}

/// Binding `chunk` of the material table: the whole buffer, or one sub-range of it.
pub fn mat_chunk_binding<'a>(device: &wgpu::Device, buf: &'a wgpu::Buffer, chunk: u32) -> wgpu::BindingResource<'a> {
    if mat_chunks(device) == 1 {
        return buf.as_entire_binding();
    }
    let bytes = mat_chunk_bytes(device);
    let offset = chunk as u64 * bytes;
    let size = bytes.min(MAT_PARAMS_BYTES.saturating_sub(offset)).max(16);
    wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buf, offset, size: std::num::NonZeroU64::new(size) })
}

/// Splits `mat_params` into one uniform array per chunk (bindings 18, 30, 31, ...)
/// behind `mat_param(i)`.
fn chunk_mat_params(device: &wgpu::Device, src: &str) -> String {
    let chunks = mat_chunks(device);
    if chunks <= 1 {
        return src.to_owned();
    }
    let per = (mat_chunk_bytes(device) / 16) as u32;
    let total = (MAT_PARAMS_BYTES / 16) as u32;
    let decl_old = "@group(0) @binding(18) var<uniform> mat_params: array<vec4<f32>, 2560>;";
    let mut decl = String::new();
    let mut body = String::new();
    for c in 0..chunks {
        let binding = if c == 0 { 18 } else { 29 + c };
        let len = per.min(total - c * per);
        decl.push_str(&format!("@group(0) @binding({binding}) var<uniform> mat_params{c}: array<vec4<f32>, {len}>;\n"));
        body.push_str(&format!("    if i < {}u {{ return mat_params{c}[i - {}u]; }}\n", (c + 1) * per, c * per));
    }
    decl.push_str(&format!("fn mat_param(i: u32) -> vec4<f32> {{\n{body}    return mat_params{}[0];\n}}", chunks - 1));
    let re_index = |s: &str| -> String {
        let mut out = String::with_capacity(s.len());
        let mut rest = s;
        while let Some(i) = rest.find("mat_params[") {
            out.push_str(&rest[..i]);
            let after = &rest[i + "mat_params[".len()..];
            let end = after.find(']').unwrap_or(0);
            out.push_str("mat_param(");
            out.push_str(&after[..end]);
            out.push(')');
            rest = &after[end + 1..];
        }
        out.push_str(rest);
        out
    };
    re_index(&src.replace(decl_old, &decl))
}

/// The WGSL for [`uniform_arrays`]: `var<storage, read> x: array<T>` becomes
/// `var<uniform> x: array<T, N>` for the light and shadow-slot arrays.
fn uniform_array_shader(device: &wgpu::Device, src: &str) -> String {
    src.replace("var<storage, read> lights: array<Light>;", &format!("var<uniform> lights: array<Light, {}>;", max_lights(device).max(1)))
        .replace("var<storage, read> lshadow_slots: array<ShadowSlot>;", &format!("var<uniform> lshadow_slots: array<ShadowSlot, {}>;", max_lshadow(device).max(1)))
}

/// [`shader_for`]'s OpenGL rewrite.
pub fn gl_shader(src: &str) -> String {
    depth_as_float(src).replace(", low_material_sampler,", ", studs_sampler,")
}

/// Rewrite WGSL's depth textures as float textures: `texture_depth_2d` ->
/// `texture_2d<f32>`, `texture_depth_multisampled_2d` ->
/// `texture_multisampled_2d<f32>`, and each `textureLoad(name, ..)` of one
/// gets `.x` (the depth). Only loads are rewritten: these shaders never
/// sample or compare their depth textures.
pub fn depth_as_float(src: &str) -> String {
    let mut names: Vec<String> = Vec::new();
    for line in src.lines() {
        let line = line.trim_start();
        if !(line.contains(": texture_depth_2d") || line.contains(": texture_depth_multisampled_2d")) {
            continue;
        }
        let Some(var) = line.find("var") else { continue };
        let rest = &line[var + 3..];
        let rest = match rest.strip_prefix('<') {
            Some(r) => &r[r.find('>').map_or(0, |i| i + 1)..],
            None => rest,
        };
        let name = rest.trim_start().split(|c: char| !(c.is_alphanumeric() || c == '_')).next().unwrap_or("");
        if !name.is_empty() {
            names.push(name.to_owned());
        }
    }
    let mut out = src
        .replace("texture_depth_multisampled_2d", "texture_multisampled_2d<f32>")
        .replace("texture_depth_2d", "texture_2d<f32>");
    for name in &names {
        let mut at = 0;
        while let Some(i) = out[at..].find("textureLoad(") {
            let open = at + i + "textureLoad(".len();
            let arg = out[open..].trim_start();
            let ident_end = arg.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(arg.len());
            if &arg[..ident_end] != name.as_str() {
                at = open;
                continue;
            }
            let mut depth = 1;
            let mut close = open;
            for (j, c) in out[open..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            close = open + j;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            out.insert_str(close + 1, ".x");
            at = close + 1;
        }
    }
    out
}

/// Poll a wgpu future to completion: on native backends adapter, device
/// and error-scope futures are ready when first polled.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        rt_thread::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, backend: wgpu::Backend, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.into(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: None,
            limit_bucket: None,
        }
    }

    #[test]
    fn depth_loads_become_float_loads() {
        let src = "@group(0) @binding(0) var depth_tex: texture_depth_2d;\n@group(0) @binding(1) var ms: texture_depth_multisampled_2d;\nfn f(p: vec2<i32>) -> f32 { return textureLoad(depth_tex, p, 0) + textureLoad( ms, max(p, vec2<i32>(0)), i32(1)) + textureLoad(other, p, 0).x; }";
        let out = depth_as_float(src);
        assert!(out.contains("var depth_tex: texture_2d<f32>;"));
        assert!(out.contains("var ms: texture_multisampled_2d<f32>;"));
        assert!(out.contains("textureLoad(depth_tex, p, 0).x + textureLoad( ms, max(p, vec2<i32>(0)), i32(1)).x + textureLoad(other, p, 0).x;"), "{out}");
    }

    #[test]
    fn old_intel_names() {
        assert!(old_intel(0x8086, "Intel(R) HD Graphics 4600"));
        assert!(old_intel(0x8086, "Mesa Intel(R) HD Graphics 4600 (HSW GT2)"));
        assert!(old_intel(0x8086, "Intel(R) HD Graphics 5500"));
        assert!(old_intel(0x8086, "Intel(R) Iris(TM) Pro Graphics 5200"));
        assert!(old_intel(0x8086, "Intel(R) HD Graphics"));
        assert!(!old_intel(0x8086, "Intel(R) UHD Graphics 620"));
        assert!(!old_intel(0x8086, "Intel(R) HD Graphics 520"));
        assert!(!old_intel(0x8086, "Intel(R) Iris(R) Xe Graphics"));
        assert!(!old_intel(0x8086, "Intel(R) Arc(TM) A770 Graphics"));
        assert!(!old_intel(0x1002, "AMD Radeon RX 9070"));
    }

    #[test]
    fn a_real_gpu_through_gl_beats_warp() {
        use wgpu::{Backend as B, DeviceType as D};
        let warp = info("Microsoft Basic Render Driver", B::Dx12, D::Cpu);
        let gl = info("Intel(R) HD Graphics 4600", B::Gl, D::IntegratedGpu);
        let dx = info("Intel(R) HD Graphics 4600", B::Dx12, D::IntegratedGpu);
        assert!(adapter_rank(&gl, false) < adapter_rank(&warp, false));
        assert!(adapter_rank(&dx, false) < adapter_rank(&gl, false));
        assert!(adapter_rank(&warp, true) < adapter_rank(&gl, true));
    }

    #[test]
    fn limits_never_exceed_the_adapter() {
        let mut gl = wgpu::Limits::downlevel_defaults();
        gl.max_sampled_textures_per_shader_stage = 16;
        let asked = required_limits(&gl);
        assert!(asked.check_limits(&gl));
        assert_eq!(asked.max_sampled_textures_per_shader_stage, 16);
        // A capable adapter gets exactly what it always got.
        let big = wgpu::Limits { max_sampled_textures_per_shader_stage: 1_000_000, max_texture_dimension_2d: 32768, ..wgpu::Limits::default() }
            .or_better_values_from(&wgpu::Limits::default());
        let asked = required_limits(&big);
        assert_eq!(asked.max_sampled_textures_per_shader_stage, 24);
        assert_eq!(asked.max_texture_dimension_2d, 8192);
    }

    #[test]
    fn mali_g52_vulkan_is_not_forced_weak_and_limits_fit() {
        // Typical ARM Mali-G52 MC2 Vulkan 1.1 limits: 8192 2D textures, >= 24 sampled textures.
        let mali_limits = wgpu::Limits {
            max_sampled_textures_per_shader_stage: 128,
            max_texture_dimension_2d: 8192,
            ..wgpu::Limits::default()
        };
        let asked = required_limits(&mali_limits);
        assert!(asked.check_limits(&mali_limits));
        assert_eq!(asked.max_sampled_textures_per_shader_stage, FULL_SAMPLED_TEXTURES);
        assert_eq!(asked.max_texture_dimension_2d, 8192);

        let caps = GpuCaps {
            name: "Mali-G52 MC2".into(),
            vendor: 0x13b5,
            backend: wgpu::Backend::Vulkan,
            driver: "ARM".into(),
            driver_info: "Vulkan 1.1.191".into(),
            device_type: wgpu::DeviceType::IntegratedGpu,
            limits: asked,
            features: wgpu::Features::empty(),
            downlevel: wgpu::DownlevelFlags::compliant(),
            r8_storage: false,
        };
        assert_eq!(caps.driver, "ARM");
        assert_eq!(caps.driver_info, "Vulkan 1.1.191");
        assert_eq!(caps.weak_reason(), None);
        assert!(!caps.reduced_bindings());
        assert!(caps.sampled_msaa());
    }

    #[test]
    fn gles3_fallback_software_and_downlevel_limits_detected() {
        let gles3_limits = wgpu::Limits::downlevel_webgl2_defaults();
        let asked_gl = required_limits(&gles3_limits);
        assert!(asked_gl.check_limits(&gles3_limits));

        let gl_caps = GpuCaps {
            name: "Mali-G52 MC2".into(),
            vendor: 0x13b5,
            backend: wgpu::Backend::Gl,
            driver: "ARM".into(),
            driver_info: "OpenGL ES 3.2".into(),
            device_type: wgpu::DeviceType::IntegratedGpu,
            limits: asked_gl,
            features: wgpu::Features::empty(),
            downlevel: wgpu::DownlevelFlags::empty(),
            r8_storage: false,
        };
        assert_eq!(gl_caps.weak_reason().as_deref(), Some("OpenGL backend"));
        assert!(gl_caps.reduced_bindings());
        assert!(!gl_caps.sampled_msaa());

        let cpu_caps = GpuCaps {
            name: "llvmpipe".into(),
            vendor: 0,
            backend: wgpu::Backend::Vulkan,
            driver: "Mesa".into(),
            driver_info: String::new(),
            device_type: wgpu::DeviceType::Cpu,
            limits: wgpu::Limits::default(),
            features: wgpu::Features::empty(),
            downlevel: wgpu::DownlevelFlags::compliant(),
            r8_storage: false,
        };
        assert_eq!(cpu_caps.weak_reason().as_deref(), Some("software adapter"));

        let downlevel_vk = GpuCaps {
            name: "Mali-G52 MC2".into(),
            vendor: 0x13b5,
            backend: wgpu::Backend::Vulkan,
            driver: "ARM".into(),
            driver_info: String::new(),
            device_type: wgpu::DeviceType::IntegratedGpu,
            limits: wgpu::Limits {
                max_sampled_textures_per_shader_stage: 16,
                max_texture_dimension_2d: 4096,
                ..wgpu::Limits::downlevel_defaults()
            },
            features: wgpu::Features::empty(),
            downlevel: wgpu::DownlevelFlags::compliant(),
            r8_storage: false,
        };
        assert!(downlevel_vk.weak_reason().is_some_and(|r| r.contains("downlevel limits")));
    }
}
