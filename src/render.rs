//! wgpu viewport renderer. Draws the scene into a float (HDR, MSAA-resolved)
//! target, then a post pass tone-maps it into the texture egui shows as an
//! image, so the 3D view docks like any other panel. The sun's shadow map is
//! drawn first.

#[path = "env_probe_gpu.rs"]
mod env_probe_gpu;
#[path = "vf_render.rs"]
pub mod vf;

use std::collections::{HashMap, HashSet};
use web_time::Instant;

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;
use glam::{Mat4, Vec3};
use rbx_dom_weak::types::Ref;
use wgpu::util::DeviceExt;

use crate::math::Xform;

/// The scene is lit in linear light into a target of the quality
/// settings' format (`Renderer::scene_format`: Rgba16Float, or the 32-bit
/// Rg11b10Ufloat without HDR, quality.rs)...
/// ...and the post pass writes display values (gamma 2.0) here.
const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Depth of field's targets keep the circle of confusion in alpha
/// (shader.wgsl fs_dof_h / fs_dof_v), whatever the scene's format.
const DOF_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The see-through encoding's copy of the scene (SeeThrough): its alpha
/// marks the pixels a layer covered, which the 32-bit scene format
/// (Rg11b10Ufloat, quality.rs) has no channel for; so always this one.
const SEE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// EVSM moments of the local light shadow maps (Studio's warps of 5.54 keep
/// e^(2·5.54) under half float's 65504).
const LSHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// One face's uniforms for the light shadow pass, at a dynamic offset.
const LSHADOW_FACE_STRIDE: u64 = 256;

/// The local light shadow atlas (Studio's Future path: a cube map per
/// shadowed light, Studio measurements "Local light shadows").
pub struct LightShadows {
    moments_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// ShadowSlot per light (6 view-projections + position / 1 over Range).
    slots_buf: wgpu::Buffer,
    /// LFace per face, read at a dynamic offset.
    face_buf: wgpu::Buffer,
    face_layout: wgpu::BindGroupLayout,
    face_bg: wgpu::BindGroup,
    /// What the atlas holds: static scene generation + the lights; redrawn when it changes.
    key: Option<u64>,
}

impl LightShadows {
    /// `enabled` false (quality settings `light_shadows`): a 1 × 1 atlas
    /// that is bound but never drawn or read (no light gets a slot).
    fn new(device: &wgpu::Device, enabled: bool) -> Self {
        let side = if enabled { crate::mech::LSHADOW_ATLAS } else { 1 };
        let tex = |label: &str, format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let moments_view = tex(
            "light shadow moments",
            LSHADOW_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let depth_view = tex("light shadow depth", DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("light shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let n = crate::mech::LSHADOW_MAX_LIGHTS.max(1);
        let slots_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light shadow slots"),
            size: (n * (6 * 64 + 16)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let face_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light shadow faces"),
            size: n as u64 * 6 * LSHADOW_FACE_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let face_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light shadow face"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(96),
                },
                count: None,
            }],
        });
        let face_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light shadow face"),
            layout: &face_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &face_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(96),
                }),
            }],
        });
        Self { moments_view, depth_view, sampler, slots_buf, face_buf, face_layout, face_bg, key: None }
    }

    /// The six face view-projections of a light: 93° frusta (Studio's
    /// 1/0.94896), near LSHADOW_NEAR, far its Range.
    fn faces(pos: Vec3, range: f32) -> [(Mat4, Vec3); 6] {
        let axes = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        let fov = 2.0 * crate::mech::LSHADOW_TAN_HALF_FOV.atan();
        let proj = Mat4::perspective_rh(fov, 1.0, crate::mech::LSHADOW_NEAR, range.max(crate::mech::LSHADOW_NEAR * 2.0));
        axes.map(|a| {
            let up = if a.y.abs() > 0.5 { Vec3::Z } else { Vec3::Y };
            (proj * Mat4::look_to_rh(pos, a, up), a)
        })
    }
}
/// DepthOfFieldEffect intensity from which Studio runs the reduced near chain
/// instead of the full-resolution cheap pass (shader.wgsl DOF_CHEAP_BELOW).
const DOF_CHEAP_BELOW: f32 = 0.22;

/// The depth-of-field near field's passes (shader.wgsl fs_near_*), in order.
const NEAR_PASSES: usize = 11;
const NEAR_ENTRIES: [&str; NEAR_PASSES] = ["fs_near_h", "fs_near_v", "fs_near_mask0", "fs_near_mask1", "fs_near_mask2", "fs_near_mask3", "fs_near_mask4", "fs_near_mask5", "fs_near_edge", "fs_near_bokeh", "fs_near_tent"];
/// The bloom pyramid's levels at most (half resolution down to 1/64).
pub const BLOOM_LEVELS: usize = 5;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub ambient: [f32; 4],
    pub outdoor: [f32; 4],
    pub sky_top: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub params: [f32; 4],
    pub fog: [f32; 4],
    pub fog_color: [f32; 4],
    pub cc: [f32; 4],
    pub cc_tint: [f32; 4],
    pub bloom: [f32; 4],
    /// The sun's shadow cascades (sun_shadow.rs): world to each tile's
    /// (u, v, depth), each sphere (centre, radius²), each tile in the atlas.
    pub sun_casc: [[[f32; 4]; 4]; 4],
    pub sun_sphere: [[f32; 4]; 4],
    pub sun_rect: [[f32; 4]; 4],
    pub sky_rot: [f32; 4],
    /// Skybox irradiance facing +X, −X, +Y, −Y, +Z, −Z, times EnvironmentDiffuseScale.
    pub env: [[f32; 4]; 6],
    /// The sky's brightness (and its light's) by time of day.
    pub sky_tint: [f32; 4],
    /// Light grid (lightgrid.rs): xyz its first voxel, w studs out along the normal (0 = none yet).
    pub lightgrid: [f32; 4],
    /// Atmosphere (round 26): rgb the haze band colour (linear), w the
    /// band strength at the horizon (0 = no Atmosphere).
    pub atmos: [f32; 4],
    /// x EnvironmentSpecularScale (0 = no sky reflection), y roughness (round 26).
    pub env_spec: [f32; 4],
    /// Parts blocking local lights (round 26, lightocc.rs): xyz the grid's
    /// first voxel, w studs per voxel (0 = no grid).
    pub lightocc: [f32; 4],
    /// Atmosphere haze toward +X, −X, +Y, −Y, +Z, −Z (AdvSkyFS irradiance × FogColor).
    pub atmos_cube: [[f32; 4]; 6],
    /// x fogOffset.x, y fogOffset.y, z Glare, w Offset.
    pub atmos_sky: [f32; 4],
    /// xyz the glare's exponents per channel (from Decay).
    pub atmos_glare: [f32; 4],
    /// Studio's indoor environment light (AmbientCube[6..11]): the
    /// irradiance of textures/sky/indoor512 × EnvironmentDiffuseScale,
    /// +X, −X, +Y, −Y, +Z, −Z (Studio measurements "Environment light and AO").
    pub env_indoor: [[f32; 4]; 6],
    /// SkyGradientTop / Bottom (CB0[30] / [31]): the time-of-day sky
    /// levels c0², c1²; w EnvironmentDiffuse / SpecularScale.
    pub sky_grad: [[f32; 4]; 2],
    /// Lamp1: the back light (faces turned from Lamp0), rgb.
    pub lamp1: [f32; 4],
    /// Screen-space AO: x, y half the projection diagonal, z near plane, w on.
    pub ssao: [f32; 4],
    pub dof: [f32; 4],
}

/// One local light as the shader reads it (see `Light` in shader.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct LightRaw {
    pub pos_range: [f32; 4],
    pub color: [f32; 4],
    pub dir: [f32; 4],
    pub extra: [f32; 4],
    pub surface_u: [f32; 4],
    pub surface_v: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub projection: [f32; 4],
    /// Per-vertex colour (sRGB 0..1); alpha 0 = use the part's Color.
    pub color: [f32; 4],
}

/// Box-projected UVs for built-in shapes. Each face is seen from outside
/// with its image upright: the front face (−Z) has image-left at +X (the
/// part's right), matching the classic clothing-template layout.
fn box_uv(p: Vec3, n: Vec3) -> [f32; 2] {
    let a = n.abs();
    if a.x >= a.y && a.x >= a.z {
        if n.x > 0.0 {
            [0.5 - p.z, 0.5 - p.y]
        } else {
            [p.z + 0.5, 0.5 - p.y]
        }
    } else if a.y >= a.z {
        if n.y > 0.0 {
            [0.5 - p.x, 0.5 - p.z]
        } else {
            [0.5 - p.x, p.z + 0.5]
        }
    } else if n.z < 0.0 {
        [0.5 - p.x, 0.5 - p.y]
    } else {
        [p.x + 0.5, 0.5 - p.y]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    r0: [f32; 4],
    r1: [f32; 4],
    r2: [f32; 4],
    size: [f32; 4],
    color: [f32; 4],
    extra: [f32; 4],
    uvt: [f32; 4],
    /// SurfaceAppearance emission: EmissiveTint² × EmissiveStrength.
    emissive: [f32; 4],
    /// SurfaceAppearance.Color (rgb, clamped 0..1) and w 1 for a part drawn
    /// with a SurfaceAppearance (Studio's per-instance constant beside the
    /// instance colour: a script tinting it only rewrites instance data).
    tint: [f32; 4],
}

/// How a texture combines with the part (shader `extra.w`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexMode {
    None,
    /// Over the part colour (clothing, SurfaceAppearance AlphaMode Overlay):
    /// transparent texels show Color.
    Over,
    /// A decal: the texture's own alpha, tinted by the colour.
    Decal,
    /// The texture alone, alpha ignored (MeshPart TextureID, SpecialMesh
    /// TextureId: Studio shows neither Color nor alpha through them).
    Opaque,
    /// SpecialMesh TextureId multiplied by its VertexColor.
    OpaqueTinted,
    /// SurfaceAppearance AlphaMode Transparency: blended by texel alpha
    /// (drawn with the see-through parts even at Transparency 0).
    Cutout,
    /// SurfaceAppearance AlphaMode TintMask: texel alpha says where Color
    /// tints the texture; the part stays opaque.
    TintMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MeshKind {
    Block = 0,
    Ball = 1,
    Cylinder = 2,
    Wedge = 3,
    CornerWedge = 4,
}

/// One part as the renderer sees it.
#[derive(Debug, Clone, Copy)]
pub struct PartDraw {
    pub xf: Xform,
    pub size: Vec3,
    /// Linear RGB.
    pub color: [f32; 3],
    pub alpha: f32,
    pub mesh: MeshKind,
    /// 0 plastic, 1 neon, 2 smooth, 3 glass.
    pub material: f32,
    pub placeholder: bool,
    pub hover: bool,
    /// Reflectance (0..1): the sky's reflection mixed into the colour.
    pub reflectance: f32,
    /// A loaded mesh (index from [`Renderer::add_mesh`]) drawn instead of `mesh`.
    pub custom_mesh: Option<u32>,
    /// A texture from [`Renderer::add_texture`].
    pub texture: Option<u32>,
    pub tex_mode: TexMode,
    /// UV scale (x, y) and offset (z, w), e.g. Texture tiling.
    pub uvt: [f32; 4],
    /// Emission (linear): the texture times its emissive mask times this.
    pub emissive: [f32; 3],
    /// The Material's darkening of textures (the part colour already has it).
    pub tex_albedo: f32,
    /// Legacy surfaces, 3 bits per face in Enum.NormalId order (Right, Top,
    /// Back, Left, Bottom, Front): 0 smooth, 1 studs, 2 weld / glue,
    /// 3 inlet, 4 universal (bands of Studio's studs.dds).
    pub surfaces: u32,
    /// Enum.Material value.
    pub material_id: u32,
    /// Material texture array layer (-1: none, draw flat).
    pub mat_layer: i32,
    /// Drawn with a SurfaceAppearance: its Color (sRGB, clamped 0..1), which
    /// multiplies the ColorMap texel before squaring (Studio's
    /// SurfaceAppearanceFS); None for every other part.
    pub surface_tint: Option<[f32; 3]>,
    /// BasePart.CastShadow: false keeps it out of the sun's shadow map.
    pub cast_shadow: bool,
}

impl PartDraw {
    pub const NO_UVT: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
}

pub struct Frame<'a> {
    pub globals: Globals,
    /// Ordinary (not reverse-Z) view-projection, for culling.
    pub cull_view_proj: Mat4,
    pub fov_y: f32,
    /// Local lights (at most MAX_LIGHTS).
    pub lights: &'a [LightRaw],
    /// Draw the sun's shadow cascades (`sun_cascades`).
    pub shadows: bool,
    pub sun_cascades: [crate::sun_shadow::Cascade; crate::sun_shadow::CASCADES],
    /// Drawn on top of what's already there (hover highlight).
    pub overlay: &'a [PartDraw],
    pub lines: &'a [LineVertex],
    /// Draw the sky (an orthographic game camera has none: Studio leaves
    /// black where nothing is).
    pub sky: bool,
    /// Terrain water's waves, animation and clearness (water.rs).
    pub water: crate::water::WaterLook,
    /// Terrain.Clouds (clouds.rs), drawn with the sky.
    pub clouds: Option<crate::clouds::CloudsLook>,
    /// The scene environment probe's work this frame (env_probe.rs).
    pub probe: Option<crate::env_probe::ProbeFrame>,
    /// Screen-space ambient occlusion (five passes a phone's GPU can do
    /// without; View > Ambient Occlusion).
    pub ssao: bool,
}

/// What a frame draws, hashed: the viewport draws again only when this,
/// [`Renderer::generation`] or the size changed (or time has to show: water,
/// a game running). The animated material clock is left out (it only
/// shows on the few materials that move).
pub fn frame_key(size: [u32; 2], out: [u32; 2], frame: &Frame) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    size.hash(&mut h);
    out.hash(&mut h);
    h.write(bytemuck::bytes_of(&frame.globals));
    for v in frame.cull_view_proj.to_cols_array() {
        v.to_bits().hash(&mut h);
    }
    frame.fov_y.to_bits().hash(&mut h);
    h.write(bytemuck::cast_slice(frame.lights));
    frame.shadows.hash(&mut h);
    for c in &frame.sun_cascades {
        for v in [c.centre.to_array(), c.right.to_array(), c.up.to_array(), c.travel.to_array()].concat() {
            v.to_bits().hash(&mut h);
        }
        c.radius.to_bits().hash(&mut h);
        c.test_radius.to_bits().hash(&mut h);
        (c.level, c.tile, c.origin).hash(&mut h);
    }
    for d in frame.overlay {
        h.write(bytemuck::bytes_of(&to_raw(d, 0.0)));
        d.cast_shadow.hash(&mut h);
    }
    h.write(bytemuck::cast_slice(frame.lines));
    frame.sky.hash(&mut h);
    frame.ssao.hash(&mut h);
    // The probe's blend toward its next slot shows in the shading (its
    // frames of work keep the viewport live: viewport.rs).
    if let Some(p) = frame.probe.as_ref() {
        p.blend.to_bits().hash(&mut h);
        p.centre.to_array().map(f32::to_bits).hash(&mut h);
    }
    let w = frame.water;
    for v in [w.wave_size, w.wave_speed, w.transparency] {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Studs per side of the cells static parts are grouped into.
const CELL: f32 = crate::mech::RENDER_CELL;

/// Which pass a draw belongs to; None = blended part, sorted every frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Pass {
    Opaque,
    Decal,
}

fn pass_of(p: &PartDraw) -> Option<Pass> {
    if p.tex_mode == TexMode::Decal && p.texture.is_some() {
        // Texture batching changes coplanar decal layer order. Keep decals
        // with the sorted translucent draws so their source order survives
        // equal distances, and translucent hosts draw behind their decals.
        None
    } else if p.tex_mode == TexMode::Cutout && p.texture.is_some() {
        // SurfaceAppearance AlphaMode Transparency: blended by texel alpha
        // on an opaque part too, never cut out (Studio 0.747: alpha 0.5 shows
        // half, 0.31 faintly; checks/sa-color).
        None
    } else if p.alpha >= 0.999 && p.material_id != 1584 {
        Some(Pass::Opaque)
    } else {
        None
    }
}

fn transparent_order(a_kind: f32, a_distance: f32, b_kind: f32, b_distance: f32) -> std::cmp::Ordering {
    let a_glass = (a_kind as u32 & 7) == 3;
    let b_glass = (b_kind as u32 & 7) == 3;
    b_glass.cmp(&a_glass).then_with(|| b_distance.total_cmp(&a_distance))
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    #[test]
    fn primitive_parity_geometry() {
        let meshes = build_meshes();
        let points: Vec<_> = ball_outline_points().collect();
        for vertex in &meshes[MeshKind::Ball as usize] {
            assert!(points.iter().any(|p| p.distance(Vec3::from_array(vertex.pos)) < 1e-6));
        }
        for index in [MeshKind::Wedge as usize, MeshKind::CornerWedge as usize] {
            assert!(meshes[index].iter().any(|v| v.projection[0] == -99.0));
        }
        for index in [0usize, MeshKind::Ball as usize, MeshKind::Cylinder as usize] {
            assert!(meshes[index].iter().all(|v| v.projection[0] == 0.0));
        }
    }

    #[test]
    fn native_cell_distance_selection_counts() {
        // Actual quality21 native draws, camera (16,20,-20), 128 domains.
        for (lo,hi,near,allocated,expected) in [
            ([-4.,-4.,-4.],[68.,68.,68.],false,1664,1489),
            ([-4.,-68.,-4.],[68.,4.,68.],false,1664,1423),
            ([-4.,-4.,-4.],[68.,68.,68.],true,2048,1830),
            ([-4.,-68.,-4.],[68.,4.,68.],true,2048,1749),
        ] {
            let range = GrassRange {start:0,ordinals:(0..allocated).collect(),allocated,near,lo:Vec3::from_array(lo),hi:Vec3::from_array(hi)};
            assert_eq!(range.count(Vec3::new(16.,20.,-20.))/6,expected);
        }
    }

    #[test]
    fn native_sun_grass_near_mid_selection_counts() {
        // Actual sun writer + receiver GPU ranges, native frames 377 / 8833.
        // Both X regions have identical distance at camera X=2 (margin 4).
        for x in [-64.0, 0.0] {
            for (z, near, allocated, camera_z, vertices) in [
                (-64.0, true, 8192, -34.0, 24570),
                (-64.0, false, 6656, -34.0, 19968),
                (0.0, true, 8192, -34.0, 19962),
                (0.0, false, 6656, -34.0, 16224),
                (-64.0, false, 6656, -150.0, 11163),
                (0.0, false, 6656, -150.0, 6594),
            ] {
                let range = GrassRange { start: 0, ordinals: (0..allocated).collect(), allocated, near,
                    lo: Vec3::new(x-4.0, -4.0, z-4.0),
                    hi: Vec3::new(x+68.0, 68.0, z+68.0) };
                // Rust uploads both sides; native submits one three-vertex triangle.
                assert_eq!(range.count(Vec3::new(2.0,28.0,camera_z)), vertices*2);
            }
        }
    }

    #[test]
    #[ignore = "writes integrated full-domain renderer selection evidence"]
    fn export_integrated_grass_selection() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        let mut limits = wgpu::Limits::default();
        limits.max_sampled_textures_per_shader_stage = 24;
        limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() })).unwrap();
        let mut renderer = Renderer::new(&device, &queue, None, Renderer::msaa_support_of(&adapter), crate::quality::RenderSettings::level(crate::quality::LEVELS));
        let mut vox = crate::terrain::Voxels::empty();
        // Same FillBlock: x,z in [0,32), y in [-16,0), full Grass voxels.
        for x in 0..8 { for y in -4..0 { for z in 0..8 { vox.set([x,y,z],2,1.0); } } }
        let mut rows = Vec::new();
        for length in [0.7,0.1] {
            renderer.clear_terrain();
            for y in -1..=0 { for x in -1..=0 { for z in -1..=0 {
                let key = [x,y,z];
                let geometry = crate::terrain::grass_geometry_density(&vox,key,length,1.0);
                let positions: Vec<_> = geometry.cells.iter().map(|cell| cell.batch.blades.iter().map(|&(_,first)| geometry.vertices[first..first+3].iter().map(|v|v.pos).collect::<Vec<_>>()).collect::<Vec<_>>()).collect();
                renderer.set_terrain_grass_geometry(&device,key,geometry);
                if let Some(chunk) = renderer.terrain_grass.get(&key) {
                    for (range, blades) in chunk.grass_ranges.iter().zip(positions) {
                        let count = range.count(Vec3::new(16.,20.,-20.)) as usize / 6;
                        rows.push(serde_json::json!({"length":length,"key":key,"near":range.near,"lo":range.lo.to_array(),"hi":range.hi.to_array(),"allocated":range.allocated,"ordinals":range.ordinals,"selected_ordinals":range.ordinals[..count],"selected_triangles":blades[..count],"resident_triangles":blades}));
                    }
                }
            } } }
        }
        std::fs::write("checks/grass-full-re/integrated-selection/renderer-selection.json",serde_json::to_vec(&rows).unwrap()).unwrap();
    }

    /// Every pipeline of the 3D view, built on this machine's GPU with the
    /// limits the app asks for (lib.rs `wgpu_options`): the WGSL through the platform's
    /// shader compiler (Metal on macOS, Vulkan/SPIR-V on Linux), with wgpu's
    /// validation errors caught. Skipped with a note where there is no GPU.
    #[test]
    fn renderer_pipelines_build_on_this_gpu() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("no GPU adapter here: skipped");
            return;
        };
        let info = adapter.get_info();
        let mut limits = wgpu::Limits::default();
        limits.max_sampled_textures_per_shader_stage = 24;
        limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
        let desc = wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc))
            .unwrap_or_else(|e| panic!("{} ({:?}) refused rustudio's limits: {e}", info.name, info.backend));
        // Every quality level's build (quality.rs): the multisampled
        // shaders and the plain one, the HDR scene format and (where the
        // device has it) the 32-bit one, the light shadow atlas on and off,
        // the lite shader of the low levels and (RUSTUDIO_LITE=31) the
        // litest one, the samplers' anisotropy.
        let mut builds: Vec<(String, crate::quality::RenderSettings)> =
            (1..=crate::quality::LEVELS).map(|level| (format!("level {level}"), crate::quality::RenderSettings::level(level))).collect();
        let mut litest = crate::quality::RenderSettings::level(1);
        litest.lite = crate::quality::LITE_ALL;
        builds.push(("level 1, RUSTUDIO_LITE=31".to_string(), litest.normalized()));
        let mut built = std::collections::HashSet::new();
        for (name, settings) in builds {
            if !built.insert(settings.build_key()) {
                continue;
            }
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let r = Renderer::new(&device, &queue, None, Renderer::msaa_support_of(&adapter), settings);
            r.build_all_pipelines();
            drop(r);
            let err = pollster::block_on(scope.pop());
            assert!(err.is_none(), "{} ({:?}) at {name}: {err:?}", info.name, info.backend);
            eprintln!("renderer built on {} ({:?}) at {name}", info.name, info.backend);
        }
    }

    /// Material packs fill the texture arrays one layer at a time, the
    /// arrays growing on the GPU as they go (`material_layer`,
    /// `remake_materials`), and are copied across when the texture quality
    /// changes: every copy and write within the arrays' layers, as wgpu
    /// checks (a copy past them is a validation error, fatal by default:
    /// the fourth material of a place closed the Android editor). Skipped
    /// with a note where there is no GPU.
    #[test]
    fn material_arrays_grow_within_bounds_on_this_gpu() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("no GPU adapter here: skipped");
            return;
        };
        let info = adapter.get_info();
        let mut limits = wgpu::Limits::default();
        limits.max_sampled_textures_per_shader_stage = 24;
        limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
        let desc = wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc))
            .unwrap_or_else(|e| panic!("{} ({:?}) refused rustudio's limits: {e}", info.name, info.backend));
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        // A phone's starting level (512² arrays), then more materials than
        // the arrays' first capacity, one per pack.
        let settings = crate::quality::RenderSettings::level(crate::mech::QUALITY_DEFAULT_PHONE).normalized();
        let mut r = Renderer::new(&device, &queue, None, Renderer::msaa_support_of(&adapter), settings);
        let packs: Vec<_> = (1..=9u64).map(|p| crate::matpack::test_maps(p, [p as u8 * 20, 90, 60])).collect();
        for (i, m) in packs[..6].iter().enumerate() {
            assert_eq!(r.material_layer(&device, &queue, m), i as i32, "pack {}", m.pack);
            assert_eq!(r.material_layer(&device, &queue, m), i as i32, "pack {} again", m.pack);
            r.materials_tick(&device, &queue);
        }
        assert_eq!((r.materials.size, r.materials.cpu.len()), (512, 6));
        assert!(r.materials.capacity >= 6);
        // Larger arrays (the level went up): made aside, filled as the
        // maps come back; new materials go to both.
        let up = crate::quality::RenderSettings::level(crate::quality::LEVELS);
        r.set_settings(&device, &queue, &up);
        assert_eq!(r.texture_size(), 1024);
        assert!(r.materials_next.is_some() && r.materials_next_pending.len() == 6);
        assert_eq!(r.material_layer(&device, &queue, &packs[6]), 6);
        assert_eq!(r.materials_next.as_ref().map(|n| n.cpu.len()), Some(7));
        r.materials_tick(&device, &queue);
        // Smaller arrays (the level went down): the layers copied from the
        // matching level, then more materials.
        let down = crate::quality::RenderSettings::level(1);
        r.set_settings(&device, &queue, &down);
        assert_eq!((r.materials.size, r.materials.cpu.len()), (256, 7));
        assert!(r.materials_next.is_none());
        for (i, m) in packs[7..].iter().enumerate() {
            assert_eq!(r.material_layer(&device, &queue, m), 7 + i as i32);
            r.materials_tick(&device, &queue);
        }
        assert_eq!(r.materials.cpu.len(), 9);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "{} ({:?}): {err:?}", info.name, info.backend);
        eprintln!("material arrays grew within bounds on {} ({:?})", info.name, info.backend);
    }
}

#[cfg(test)]
mod see_bounds_tests {
    use super::SeeBounds;
    use glam::{Mat4, Vec3};

    /// A camera at the origin looking down -Z (90° vertical field, square view).
    fn bounds() -> SeeBounds {
        let vp = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 1000.0);
        SeeBounds::new(vp, [320, 320])
    }

    fn covers(rects: &[[u32; 4]], x: u32, y: u32) -> bool {
        rects.iter().any(|r| x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3])
    }

    #[test]
    fn nothing_marks_nothing() {
        assert!(bounds().rects().is_empty());
    }

    #[test]
    fn a_small_part_marks_only_the_tiles_under_it() {
        let mut b = bounds();
        b.add_sphere(Vec3::new(0.0, 0.0, -10.0), 1.0);
        let rects = b.rects();
        // centre covered, merged into one rectangle, far corners not
        assert_eq!(rects.len(), 1, "{rects:?}");
        assert!(covers(&rects, 160, 160));
        assert!(!covers(&rects, 5, 5) && !covers(&rects, 315, 315));
        let area: u32 = rects.iter().map(|r| r[2] * r[3]).sum();
        assert!(area < 320 * 320 / 8, "{rects:?}");
    }

    #[test]
    fn a_part_reaching_behind_the_camera_marks_everything() {
        let mut b = bounds();
        b.add_sphere(Vec3::new(0.0, 0.0, -0.5), 2.0);
        assert_eq!(b.rects(), vec![[0, 0, 320, 320]]);
    }

    #[test]
    fn a_part_outside_the_view_marks_nothing() {
        let mut b = bounds();
        b.add_sphere(Vec3::new(500.0, 0.0, -10.0), 1.0);
        assert!(b.rects().is_empty());
    }

    #[test]
    fn two_parts_mark_two_regions() {
        let mut b = bounds();
        b.add_sphere(Vec3::new(-6.0, 6.0, -10.0), 0.5);
        b.add_sphere(Vec3::new(6.0, -6.0, -10.0), 0.5);
        let rects = b.rects();
        assert!(covers(&rects, 64, 64) && covers(&rects, 256, 256), "{rects:?}");
        assert!(!covers(&rects, 160, 160) && !covers(&rects, 256, 64), "{rects:?}");
    }
}

#[cfg(test)]
mod transparent_order_tests {
    use super::transparent_order;
    use std::cmp::Ordering;

    #[test]
    fn glass_precedes_other_transparents_regardless_of_distance() {
        assert_eq!(transparent_order(3.0, 1.0, 0.0, 100.0), Ordering::Less);
        assert_eq!(transparent_order(6.0, 100.0, 3.0, 1.0), Ordering::Greater);
    }

    #[test]
    fn each_group_remains_back_to_front() {
        for kind in [0.0, 3.0, 6.0] {
            assert_eq!(transparent_order(kind, 100.0, kind, 1.0), Ordering::Less);
            assert_eq!(transparent_order(kind, 1.0, kind, 1.0), Ordering::Equal);
        }
    }
}

fn shows(p: &PartDraw) -> bool {
    p.alpha > 0.001
}

fn radius_of(p: &PartDraw) -> f32 {
    p.size.length() * 0.5
}

/// The camera is inside the part's box.
fn camera_inside(p: &PartDraw, cam: Vec3) -> bool {
    let local = p.xf.rot.transpose() * (cam - p.xf.pos);
    local.abs().cmple(p.size * 0.5).all()
}

#[derive(Debug, Clone, PartialEq)]
enum ItemState {
    /// New: drawn per frame until the next static rebuild.
    Pending,
    /// In the static buffer: slot per draw (u32::MAX = not drawn).
    Static(Vec<u32>),
    /// Moves (or is see-through): drawn per frame.
    Dynamic,
}

/// One part's draws (the part itself plus its decals).
struct Item {
    draws: Vec<PartDraw>,
    bias: f32,
    state: ItemState,
    /// Last time the part changed place or look (Dynamic parts quiet for
    /// DYNAMIC_SETTLE_S go back to the static batches).
    moved: Instant,
}

/// Same place within float noise: a CFrame write of the same value comes
/// back through the scene's transform maths with its low bits changed.
fn near_xf(a: &crate::math::Xform, b: &crate::math::Xform) -> bool {
    a.pos.abs_diff_eq(b.pos, 1e-4) && a.rot.abs_diff_eq(b.rot, 1e-5)
}

struct Batch {
    pass: Pass,
    key: (usize, usize),
    lo: Vec3,
    hi: Vec3,
    start: u32,
    count: u32,
}

/// Anchored, opaque parts: uploaded once, sorted by (pass, mesh/texture,
/// cell, size) so each cell is one instanced draw that can be culled and
/// trimmed of its smallest parts with distance.
#[derive(Default)]
struct StaticLayout {
    raw: Vec<InstanceRaw>,
    /// Bounding radius per instance (descending within a batch).
    radii: Vec<f32>,
    batches: Vec<Batch>,
    hidden: usize,
    /// Slots changed in place, uploaded next frame.
    writes: Vec<u32>,
}

/// Counters for the last frame (shown in the bench report).
#[derive(Default, Debug, Clone, Copy)]
pub struct RenderStats {
    pub items: usize,
    pub static_instances: usize,
    pub batches: usize,
    pub per_frame_items: usize,
    pub drawn_instances: usize,
    pub draw_calls: usize,
    pub culled_batches: usize,
    pub lod_skipped: usize,
    pub rebuilds: u64,
}

/// A small, stable depth nudge per part so exactly coplanar faces of two
/// parts always resolve the same way instead of flickering. From the part's
/// own size and placement, not its Ref (Refs are random per run: which of
/// two coplanar floors showed changed from run to run).
fn bias_of(draws: &[PartDraw]) -> f32 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    if let Some(d) = draws.first() {
        for v in d.size.to_array().into_iter().chain(d.xf.pos.to_array()) {
            v.to_bits().hash(&mut h);
        }
    }
    (h.finish() % 16) as f32 * crate::mech::DEPTH_BIAS_STEP
}

fn spread(v: u64) -> u64 {
    let mut x = v & 0x1f_ffff;
    x = (x | x << 32) & 0x1f_0000_0000_ffff;
    x = (x | x << 16) & 0x1f_0000_ff00_00ff;
    x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
    x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
    (x | x << 2) & 0x1249_2492_4924_9249
}

/// Z-order of a cell, so neighbouring cells sit next to each other.
fn morton(c: [i32; 3]) -> u64 {
    let o = |v: i32| (v + (1 << 20)) as u64;
    spread(o(c[0])) | spread(o(c[1])) << 1 | spread(o(c[2])) << 2
}

/// World-space bounding box of a draw.
fn aabb_of(p: &PartDraw) -> (Vec3, Vec3) {
    let r = p.xf.rot;
    let h = p.size * 0.5;
    let ext = r.x_axis.abs() * h.x + r.y_axis.abs() * h.y + r.z_axis.abs() * h.z;
    (p.xf.pos - ext, p.xf.pos + ext)
}

/// Frustum planes (normal, d) from a view-projection matrix (depth 0..1).
pub fn frustum(m: Mat4) -> [glam::Vec4; 5] {
    let (r0, r1, r2, r3) = (m.row(0), m.row(1), m.row(2), m.row(3));
    let n = |p: glam::Vec4| p / p.truncate().length().max(1e-6);
    [n(r3 + r0), n(r3 - r0), n(r3 + r1), n(r3 - r1), n(r2)]
}

fn box_visible(planes: &[glam::Vec4; 5], lo: Vec3, hi: Vec3) -> bool {
    planes.iter().all(|p| {
        let v = Vec3::new(
            if p.x > 0.0 { hi.x } else { lo.x },
            if p.y > 0.0 { hi.y } else { lo.y },
            if p.z > 0.0 { hi.z } else { lo.z },
        );
        p.truncate().dot(v) + p.w >= 0.0
    })
}

pub fn sphere_visible(planes: &[glam::Vec4; 5], c: Vec3, r: f32) -> bool {
    planes.iter().all(|p| p.truncate().dot(c) + p.w >= -r)
}

/// The pixels a frame's decals and see-through parts can cover: the
/// see-through encode and decode passes (SeeThrough) only run there, since
/// outside every layer they change nothing (encode writes alpha 0, decode
/// keeps alpha-0 samples as they are). Kept as a TILES x TILES mask of the
/// view: each item marks the tiles under its screen rectangle. Conservative:
/// an item's bounding sphere (its box's, as frustum culling uses), plus a
/// margin; a box behind or across the camera plane, or a skinned mesh,
/// marks the whole view.
/// RUSTUDIO_SEE_FULL=1: always the whole view.
struct SeeBounds {
    view_proj: Mat4,
    size: [u32; 2],
    rows: [u32; SeeBounds::TILES],
}

impl SeeBounds {
    const TILES: usize = 32;
    /// Pixels added on each side (MSAA samples and rasterisation at the edge).
    const MARGIN: f32 = 4.0;

    fn new(view_proj: Mat4, size: [u32; 2]) -> Self {
        static FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mut b = SeeBounds { view_proj, size, rows: [0; Self::TILES] };
        if *FULL.get_or_init(|| std::env::var_os("RUSTUDIO_SEE_FULL").is_some()) {
            b.set_full();
        }
        b
    }

    fn set_full(&mut self) {
        self.rows = [u32::MAX; Self::TILES];
    }

    /// The tile column / row holding pixel coordinate `v` of `extent`.
    fn tile(v: f32, extent: u32) -> usize {
        ((v / extent as f32 * Self::TILES as f32).floor().max(0.0) as usize).min(Self::TILES - 1)
    }

    fn add_box(&mut self, lo: Vec3, hi: Vec3) {
        if self.rows[0] == u32::MAX && self.rows.iter().all(|r| *r == u32::MAX) {
            return;
        }
        if !(lo.is_finite() && hi.is_finite()) {
            self.set_full();
            return;
        }
        let (w, h) = (self.size[0] as f32, self.size[1] as f32);
        let (mut a, mut b) = (glam::Vec2::splat(f32::MAX), glam::Vec2::splat(f32::MIN));
        for i in 0..8 {
            let c = Vec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            let clip = self.view_proj * c.extend(1.0);
            if !(clip.w > 1e-3) {
                self.set_full();
                return;
            }
            let p = glam::Vec2::new((clip.x / clip.w * 0.5 + 0.5) * w, (0.5 - clip.y / clip.w * 0.5) * h);
            a = a.min(p);
            b = b.max(p);
        }
        a -= glam::Vec2::splat(Self::MARGIN);
        b += glam::Vec2::splat(Self::MARGIN);
        if b.x < 0.0 || b.y < 0.0 || a.x >= w || a.y >= h {
            return;
        }
        let (c0, c1) = (Self::tile(a.x, self.size[0]), Self::tile(b.x, self.size[0]));
        let (r0, r1) = (Self::tile(a.y, self.size[1]), Self::tile(b.y, self.size[1]));
        let mask = ((1u64 << (c1 + 1)) - (1u64 << c0)) as u32;
        for row in &mut self.rows[r0..=r1] {
            *row |= mask;
        }
    }

    fn add_sphere(&mut self, c: Vec3, r: f32) {
        let r = Vec3::splat(r);
        self.add_box(c - r, c + r);
    }

    /// The marked tiles as scissor rectangles (x, y, width, height): runs of
    /// tiles along a row, merged with the same run on the rows below.
    fn rects(&self) -> Vec<[u32; 4]> {
        let edge = |i: usize, extent: u32| (i as u64 * extent as u64 / Self::TILES as u64) as u32;
        let mut out: Vec<([u32; 4], usize)> = Vec::new();
        for (r, &bits) in self.rows.iter().enumerate() {
            let mut c = 0;
            while c < Self::TILES {
                if bits & (1 << c) == 0 {
                    c += 1;
                    continue;
                }
                let c0 = c;
                while c < Self::TILES && bits & (1 << c) != 0 {
                    c += 1;
                }
                let (x0, x1) = (edge(c0, self.size[0]), edge(c, self.size[0]));
                let (y0, y1) = (edge(r, self.size[1]), edge(r + 1, self.size[1]));
                match out.iter_mut().find(|(o, last)| *last + 1 == r && o[0] == x0 && o[2] == x1 - x0) {
                    Some((o, last)) => {
                        o[3] = y1 - o[1];
                        *last = r;
                    }
                    None => out.push(([x0, y0, x1 - x0, y1 - y0], r)),
                }
            }
        }
        out.into_iter().map(|(o, _)| o).filter(|o| o[2] > 0 && o[3] > 0).collect()
    }
}

fn box_distance(p: Vec3, lo: Vec3, hi: Vec3) -> f32 {
    (p.clamp(lo, hi) - p).length()
}

const HIDDEN: InstanceRaw = InstanceRaw {
    r0: [0.0; 4],
    r1: [0.0; 4],
    r2: [0.0; 4],
    size: [0.0; 4],
    color: [0.0; 4],
    extra: [0.0; 4],
    uvt: [0.0; 4],
    emissive: [0.0; 4],
    tint: [0.0; 4],
};

/// Terrain palette entries: 23 SmoothGrid materials × top / side / bottom.
pub const TERRAIN_PALETTE: usize = 69;

/// One terrain chunk's triangles (round 21) and their bounds.
struct GrassRange {
    start: u32,
    ordinals: Vec<usize>,
    allocated: usize,
    near: bool,
    lo: Vec3,
    hi: Vec3,
}
impl GrassRange {
    fn count(&self, camera: Vec3) -> u32 {
        let distance = camera.distance(camera.clamp(self.lo, self.hi));
        let cap = crate::mech::GRASS_RADIUS;
        let factor = (2.0 / (1.0 + distance.min(cap) / cap) - 1.0).clamp(0.0, 1.0);
        let group = if self.near {3} else {1};
        let limit = ((self.allocated / group) as f32 * factor).trunc() as usize * group;
        (self.ordinals.partition_point(|&o| o < limit) * 6) as u32
    }
}
struct TerrainChunk {
    grass_ranges: Vec<GrassRange>,
    diagnostic_positions: Vec<[f32; 3]>,
    vbuf: wgpu::Buffer,
    count: u32,
    lo: Vec3,
    hi: Vec3,
}

/// Layers with their own Studio material constants in `MaterialArrays::params`.
const MAT_PARAM_LAYERS: usize = 256;

/// Two RGBA8 texture arrays of material maps (layer = pack), mipmapped, and
/// the part shader's per-layer material constants (shader `mat_params`):
/// 5 vec4 per layer: detail/tiling, noise/type, far tiling/emissive R,
/// cutoffs/emissive G, cutoff scales/emissive B; then Studio's 32²
/// noise and 128² Wang index, packed as exact integers in f32 lanes.
struct MaterialArrays {
    albedo: wgpu::Texture,
    surface: wgpu::Texture,
    emission: wgpu::Texture,
    albedo_view: wgpu::TextureView,
    surface_view: wgpu::TextureView,
    emission_view: wgpu::TextureView,
    size: u32,
    capacity: u32,
    layers: std::collections::HashMap<u64, u32>,
    cpu: Vec<std::sync::Arc<crate::matpack::Maps>>,
    params: Vec<[f32; 4]>,
    params_buf: wgpu::Buffer,
}

impl MaterialArrays {
    /// The default constants: no detail map, textureTiling 1, tiled; noise.
    fn default_params() -> Vec<[f32; 4]> {
        let mut p = Vec::with_capacity(MAT_PARAM_LAYERS * 5 + 256 + 1024);
        for _ in 0..MAT_PARAM_LAYERS {
            p.push([-1.0, 0.0, 0.0, 1.0]);
            p.push([0.0, 0.0, 4.0, 1.0]);
            p.push([1.0, 1.0, 1.0, 0.0]);
            p.push([0.0; 4]);
            p.push([1.0, 1.0, 1.0, 0.0]);
        }
        p.extend(crate::matpack::detile_noise().chunks(4).map(|c| [c[0], c[1], c[2], c[3]]));
        p.extend(crate::matpack::wang_index());
        p
    }

    fn new(device: &wgpu::Device, size: u32, capacity: u32) -> Self {
        Self::with_params(device, size, capacity, Self::default_params())
    }

    fn with_params(device: &wgpu::Device, size: u32, capacity: u32, params: Vec<[f32; 4]>) -> Self {
        let make = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: capacity,
                },
                mip_level_count: mip_levels(size, size),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                // (COPY_SRC: growing the arrays copies the layers across)
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let view = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let albedo = make("material albedo", wgpu::TextureFormat::Rgba8Unorm);
        let surface = make("material surface", wgpu::TextureFormat::Rgba8Unorm);
        let emission = make("material emission", wgpu::TextureFormat::R8Unorm);
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        MaterialArrays {
            albedo_view: view(&albedo),
            surface_view: view(&surface),
            emission_view: view(&emission),
            albedo,
            surface,
            emission,
            size,
            capacity,
            layers: Default::default(),
            cpu: Vec::new(),
            params,
            params_buf,
        }
    }

    /// Mip levels of a pack's chain (matpack::LAYER_SIZE² first) left out:
    /// an array smaller than the packs (the texture quality setting) takes
    /// the chain from the level of its size.
    fn skipped_mips(&self) -> usize {
        skipped_mips(self.size)
    }

    /// Copies `from`'s first `layers` layers into these arrays, every mip
    /// these arrays have: from the same size whole, from larger arrays
    /// (the texture quality setting went down) from the level of this size
    /// on. Nothing from smaller arrays: their top levels aren't there.
    fn copy_layers(&self, device: &wgpu::Device, queue: &wgpu::Queue, from: &MaterialArrays, layers: u32) {
        // (a copy past either array's layers would be a fatal validation error)
        let layers = layers.min(from.capacity).min(self.capacity);
        if layers == 0 || from.size < self.size {
            return;
        }
        let skip = skipped_mips(self.size) - skipped_mips(from.size);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("material arrays grow") });
        for (src, dst) in [(&from.albedo, &self.albedo), (&from.surface, &self.surface), (&from.emission, &self.emission)] {
            for mip in 0..mip_levels(self.size, self.size) {
                let side = (self.size >> mip).max(1);
                let at = |texture, mip_level| wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                };
                enc.copy_texture_to_texture(at(src, mip + skip as u32), at(dst, mip), wgpu::Extent3d { width: side, height: side, depth_or_array_layers: layers });
            }
        }
        queue.submit(Some(enc.finish()));
    }

    fn upload(&self, queue: &wgpu::Queue, layer: u32, maps: &crate::matpack::Pixels) {
        if layer >= self.capacity {
            // (a write past the arrays' layers would be a fatal validation error)
            eprintln!("rustudio: material layer {layer} past the arrays' {} layers: not uploaded", self.capacity);
            return;
        }
        let skip = self.skipped_mips();
        for (tex, data, channels) in [(&self.albedo, &maps.albedo, 4), (&self.surface, &maps.surface, 4), (&self.emission, &maps.emissive, 1)] {
            for (mip, (at, size)) in crate::matpack::mip_levels().into_iter().enumerate().skip(skip) {
                let at = at / (4 / channels) as usize;
                let bytes = (size * size * channels) as usize;
                let Some(level) = data.get(at..at + bytes) else {
                    break;
                };
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: (mip - skip) as u32,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    level,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size * channels),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
}

/// Mip levels of a pack's chain (matpack::LAYER_SIZE² first) that arrays
/// `size` across leave out.
fn skipped_mips(size: u32) -> usize {
    (crate::matpack::LAYER_SIZE / size.max(1)).max(1).trailing_zeros() as usize
}

struct Mesh {
    vbuf: wgpu::Buffer,
    count: u32,
    /// Back faces culled (Some(true)), as Studio draws a MeshPart with
    /// DoubleSided off (an inverted-hull outline shell then shows only as its
    /// rim); Some(false) two-sided by choice (DoubleSided); None two-sided
    /// because the winding isn't known (CSG).
    cull: Option<bool>,
    /// Made for the skinning pass (STORAGE usage).
    storage: bool,
}

/// Studio blends see-through surfaces in its scene buffer's encoding,
/// e = sqrt(clamp(L x 0.25 x exposure)) (shader.wgsl see_enc), not in linear
/// light. Frames with decals or see-through parts copy the scene into that
/// encoding (per sample, with MSAA), draw them there, and decode the pixels
/// they covered back (the rest keep their linear value, highlights above
/// the encoding's ceiling too). The copy and the decode run only over the
/// screen tiles those layers can cover (SeeBounds), not the whole view.
/// The pipelines are the view's (`Pipe::SeeEncode`, `Pipe::SeeDecode`, the
/// `*Enc` part pipelines), made for the scene's sample count and format.
/// RUSTUDIO_LINEAR_BLEND=1 blends in linear light as before (for comparisons).
struct SeeThrough {
    /// The scene (or its encoded copy), at the scene's sample count.
    layout: wgpu::BindGroupLayout,
    linear: bool,
}

impl SeeThrough {
    fn new(device: &wgpu::Device, samples: u32) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("see-through scene"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: samples > 1,
                },
                count: None,
            }],
        });
        SeeThrough { layout, linear: std::env::var_os("RUSTUDIO_LINEAR_BLEND").is_some() }
    }

    fn bind(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("see-through scene"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) }],
        })
    }
}

struct Target {
    size: [u32; 2],
    /// The multisampled scene, resolved into `hdr`; None without MSAA
    /// (the passes draw into `hdr` directly).
    msaa: Option<wgpu::TextureView>,
    /// The scene in the see-through encoding (SeeThrough; the scene's
    /// sample count), and the bind groups reading the scene (encode) and
    /// it (decode).
    see: wgpu::TextureView,
    see_encode_bg: wgpu::BindGroup,
    see_decode_bg: wgpu::BindGroup,
    depth: wgpu::TextureView,
    /// The resolved scene in linear light (`Renderer::scene_format`).
    hdr: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    /// The opaque scene's colour, copied for the water pass to look through.
    scene_copy: wgpu::Texture,
    scene_copy_view: wgpu::TextureView,
    water_bg: wgpu::BindGroup,
    water_depth: wgpu::TextureView,
    glass_bg: wgpu::BindGroup,
    glass_depth: wgpu::TextureView,
    /// The post pass's inputs: the scene, the bloom (or black) and the far
    /// field (or black).
    hdr_bg: wgpu::BindGroup,
    /// Group 2 of the post pass: the depth and the near field (or black).
    post_depth_bg: wgpu::BindGroup,
    /// Group 2 with the depth alone (black input): the glass depth
    /// snapshot, SSAO's depth down and the far field's CoC.
    depth_bg: wgpu::BindGroup,
    /// Depth of field's targets, when the settings have it.
    dof: Option<DofTargets>,
    /// The bloom pyramid, when the settings have it.
    bloom: Option<BloomTargets>,
    /// SSAO's half-resolution targets, when the settings have it.
    ssao: Option<SsaoTargets>,
    /// The tone-mapped image, `size` pixels; what egui shows unless the
    /// frame is scaled up.
    display: wgpu::Texture,
    display_view: wgpu::TextureView,
    /// The panel's size in pixels: the display image is scaled up to it
    /// when it differs from `size` (quality.rs `render_scale`,
    /// `motion_scale`).
    out_size: [u32; 2],
    /// EASU then RCAS from `display` to the panel's size, when the frame
    /// is drawn under it and the settings have the upscale.
    upscale: Option<UpscaleTargets>,
}

/// The frame scaled up (shader.wgsl fs_easu, fs_rcas): EASU's output and
/// the bind groups reading the display image and it; RCAS's output is
/// what egui shows.
struct UpscaleTargets {
    easu_bg: wgpu::BindGroup,
    easu_view: wgpu::TextureView,
    rcas_bg: wgpu::BindGroup,
    view: wgpu::TextureView,
}

/// Depth of field: the far field (two full-resolution passes) and the
/// near field (NEAR_PASSES quarter-resolution passes), DOF_FORMAT.
struct DofTargets {
    bgs: [wgpu::BindGroup; 2],
    views: [wgpu::TextureView; 2],
    near_bgs: Vec<wgpu::BindGroup>,
    near_views: Vec<wgpu::TextureView>,
}

/// The linear bloom pyramid, from half resolution down a level each
/// (`RenderSettings::bloom_levels` of them at most).
struct BloomTargets {
    /// Level 0 (half resolution): what the post pass adds.
    view: wgpu::TextureView,
    bright_capture: Option<wgpu::Texture>,
    textures: Vec<wgpu::Texture>,
    levels: Vec<wgpu::TextureView>,
    down_bgs: Vec<wgpu::BindGroup>,
    up_bgs: Vec<wgpu::BindGroup>,
}

/// SSAO at half resolution: the AO depth, then (occlusion, depth) ping-
/// ponged through the blurs; a bind group per pass (depth + its input):
/// occlusion (reads the AO depth), blur across (reads a; the apply too),
/// blur down (reads b).
struct SsaoTargets {
    depth_view: wgpu::TextureView,
    views: [wgpu::TextureView; 2],
    bgs: [wgpu::BindGroup; 3],
}

/// `RUSTUDIO_GPU_BISECT=1` (with the perf log): the 3D frame drawn with
/// one thing left out, a different one with every perf line (5 s each,
/// perf_log::periods), the line naming it — to find what a frame's GPU
/// time goes to on a device without a GPU profiler (docs/ANDROID.md
/// "Graphics quality"). The steps, in order:
///
/// 0. everything (the reference);
/// 1. no lines (the grid, selection boxes, handles);
/// 2. no sky;
/// 3. no parts (nor decals, terrain or see-through parts);
/// 4. the scene pass empty (its clear and store alone);
/// 5. no scene pass at all (post shows the last scene);
/// 6. no post passes (the 3D view keeps its last image);
/// 7. nothing encoded but the timestamps (the submit alone);
/// 8. everything, submitted by the renderer itself, then the CPU waits
///    for the GPU to finish the frame (`device.poll`, timed: section
///    `vpr_gpu_sync`) — whether the timestamps' span is wall time;
/// 9. everything, the frame submitted the other way (`one_submit_wanted`:
///    with egui's where the renderer submits its own, its own where one
///    submit a frame is the default — a phone);
/// 10. no 3D frame at all (the view shows its last image): the editor's
///     frame interval without the 3D work.
///
/// Off, every step reads as 0.
pub mod bisect {
    pub const STEPS: [&str; 11] = [
        "everything",
        "no lines",
        "no sky",
        "no parts",
        "empty scene pass",
        "no scene pass",
        "no post",
        "nothing encoded",
        "GPU sync after submit",
        "the other submit mode",
        "no 3D frame",
    ];

    /// Whether the bisection is on (RUSTUDIO_GPU_BISECT set, not 0).
    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("RUSTUDIO_GPU_BISECT").is_some_and(|v| v != "0"))
    }

    /// The step in force this period (0 when off).
    pub fn step() -> usize {
        if on() {
            crate::perf_log::periods() as usize % STEPS.len()
        } else {
            0
        }
    }

    /// The step's name, for the perf line (None when off).
    pub fn name() -> Option<&'static str> {
        on().then(|| STEPS[step()])
    }
}

/// GPU time of a drawn frame's sections, by timestamp queries (when the
/// device has them — lib.rs `wgpu_options` asks while the perf log is on;
/// src/perf_log.rs reads the totals): the sun's and the lights' shadow
/// maps, the probe's faces, the scene pass, the see-through passes,
/// SSAO, clouds + bloom + DoF, post (`NAMES`). The readings come back a
/// frame or two later.
pub struct GpuTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    /// Read-back buffers and their state: 0 free, 1 map asked, 2 mapped,
    /// 3 the map failed.
    ring: Vec<(wgpu::Buffer, std::sync::Arc<std::sync::atomic::AtomicU8>)>,
    slot: Option<usize>,
    period_ns: f32,
    /// Milliseconds summed over the frames read so far, a section per
    /// pair of marks (`SECTIONS`), then the whole frame; and how many frames.
    sum: [f64; GpuTimer::SECTIONS + 1],
    frames: u32,
    /// The last frame read, whole (ms): a few frames behind the one drawn.
    last_ms: Option<f32>,
}

impl GpuTimer {
    /// The marks render() writes, each between two passes (never inside
    /// one: on a tiling GPU a timestamp inside a pass reads as the whole
    /// pass): 0 the start, 1 after the sun's cascades and the light
    /// shadows, 2 after the environment probe's faces, 3 after the scene
    /// pass (sky, parts, decals, terrain, lines), 4 after the see-through
    /// passes (water, glass, see-through parts: only when the scene
    /// splits), 5 after SSAO, 6 after clouds, bloom and DoF, 7 after the
    /// post passes (tone map, EASU, RCAS).
    const MARKS: u32 = 8;
    /// The sections between the marks, in order.
    pub const SECTIONS: usize = 7;
    /// Their names, for the perf line.
    pub const NAMES: [&'static str; GpuTimer::SECTIONS] = ["shadows", "probe", "scene", "see-through", "ssao", "bloom/dof", "post"];

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let want = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(want) {
            return None;
        }
        let bytes = Self::MARKS as u64 * 8;
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: Self::MARKS,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame timestamps"),
            size: bytes.max(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT),
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ring = (0..3)
            .map(|_| {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frame timestamps read"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (buf, std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)))
            })
            .collect();
        Some(GpuTimer { set, resolve, ring, slot: None, period_ns: queue.get_timestamp_period(), sum: [0.0; GpuTimer::SECTIONS + 1], frames: 0, last_ms: None })
    }

    /// Timestamp `i` at this point of the encoder's work.
    pub fn mark(&self, encoder: &mut wgpu::CommandEncoder, i: u32) {
        if i < Self::MARKS {
            encoder.write_timestamp(&self.set, i);
        }
    }

    /// Before the submit: the marks copied to a free read buffer (none
    /// free: this frame goes unmeasured).
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        use std::sync::atomic::Ordering;
        self.slot = self.ring.iter().position(|(_, s)| s.load(Ordering::Acquire) == 0);
        if let Some(i) = self.slot {
            encoder.resolve_query_set(&self.set, 0..Self::MARKS, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.ring[i].0, 0, Self::MARKS as u64 * 8);
        }
    }

    /// After the submit: asks for this frame's read and takes in the
    /// readings that have arrived (their callbacks run from the queue's
    /// submits and `device.poll`).
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        use std::sync::atomic::Ordering;
        if let Some(i) = self.slot.take() {
            let (buf, state) = &self.ring[i];
            state.store(1, Ordering::Release);
            let s = state.clone();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| s.store(if r.is_ok() { 2 } else { 3 }, Ordering::Release));
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let period = self.period_ns as f64;
        for (buf, state) in &self.ring {
            match state.load(Ordering::Acquire) {
                2 => {
                    if let Ok(data) = buf.slice(..).get_mapped_range() {
                        let t: Vec<u64> = data.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
                        if t.len() >= Self::MARKS as usize {
                            let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * period / 1e6;
                            for i in 0..Self::SECTIONS {
                                self.sum[i] += ms(t[i], t[i + 1]);
                            }
                            let last = Self::MARKS as usize - 1;
                            self.sum[Self::SECTIONS] += ms(t[0], t[last]);
                            self.frames += 1;
                            self.last_ms = Some(ms(t[0], t[last]) as f32);
                        }
                    }
                    buf.unmap();
                    state.store(0, Ordering::Release);
                }
                3 => state.store(0, Ordering::Release),
                _ => {}
            }
        }
    }

    /// The sums so far (ms per section, then the whole) and the frames in them.
    pub fn totals(&self) -> ([f64; GpuTimer::SECTIONS + 1], u32) {
        (self.sum, self.frames)
    }

    /// The GPU time of the last 3D frame read (ms; the read runs a few
    /// frames behind), None before the first.
    pub fn last_ms(&self) -> Option<f32> {
        self.last_ms
    }
}

/// A render pipeline of the 3D view. Each is made on first use ([`Pipes`]),
/// so starting takes the shader module alone and a frame compiles only
/// what it draws; a thread warms the rest in the background.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pipe {
    Sky,
    Opaque,
    OpaqueTwoSided,
    DecalDepth,
    Transparent,
    TransparentCulled,
    Glass,
    GlassCulled,
    Overlay,
    /// The see-through five drawing into Studio's scene encoding
    /// (shader.wgsl fs_part_enc; SeeThrough), and the passes that copy the
    /// scene into it and decode the covered pixels back.
    TransparentEnc,
    TransparentCulledEnc,
    GlassEnc,
    GlassCulledEnc,
    OverlayEnc,
    SeeEncode,
    SeeDecode,
    Lines,
    /// Lines drawn over everything (vertex alpha ≥ 4: handles, the
    /// selection's), with no depth test.
    LinesOnTop,
    Water,
    Terrain,
    Shadow,
    TerrainShadow,
    TerrainShadowBack,
    LightShadow,
    GlassResolve,
    SsaoDown,
    SsaoMain,
    SsaoBlurX,
    SsaoBlurY,
    SsaoApply,
    BloomBright,
    BloomDown,
    BloomUp,
    DofH,
    DofV,
    /// One of the near field's passes (NEAR_ENTRIES).
    Near(usize),
    Post,
    /// The frame scaled up to the panel (quality.rs `upscale`): EASU, then RCAS.
    Easu,
    Rcas,
}

impl Pipe {
    const COUNT: usize = 38 + NEAR_PASSES;

    fn index(self) -> usize {
        match self {
            Pipe::Sky => 0,
            Pipe::Opaque => 1,
            Pipe::OpaqueTwoSided => 2,
            Pipe::DecalDepth => 3,
            Pipe::Transparent => 4,
            Pipe::TransparentCulled => 5,
            Pipe::Glass => 6,
            Pipe::GlassCulled => 7,
            Pipe::Overlay => 8,
            Pipe::Lines => 9,
            Pipe::Water => 10,
            Pipe::Terrain => 11,
            Pipe::Shadow => 12,
            Pipe::TerrainShadow => 13,
            Pipe::TerrainShadowBack => 14,
            Pipe::LightShadow => 15,
            Pipe::GlassResolve => 16,
            Pipe::SsaoDown => 17,
            Pipe::SsaoMain => 18,
            Pipe::SsaoBlurX => 19,
            Pipe::SsaoBlurY => 20,
            Pipe::SsaoApply => 21,
            Pipe::BloomBright => 22,
            Pipe::BloomDown => 23,
            Pipe::BloomUp => 24,
            Pipe::DofH => 25,
            Pipe::DofV => 26,
            Pipe::Post => 27,
            Pipe::Easu => 28,
            Pipe::Rcas => 29,
            Pipe::LinesOnTop => 30,
            Pipe::TransparentEnc => 31,
            Pipe::TransparentCulledEnc => 32,
            Pipe::GlassEnc => 33,
            Pipe::GlassCulledEnc => 34,
            Pipe::OverlayEnc => 35,
            Pipe::SeeEncode => 36,
            Pipe::SeeDecode => 37,
            Pipe::Near(i) => 38 + i.min(NEAR_PASSES - 1),
        }
    }

    /// The same draw into the see-through encoding (SeeThrough), for the
    /// five that have one; itself otherwise.
    fn encoded(self) -> Pipe {
        match self {
            Pipe::Transparent => Pipe::TransparentEnc,
            Pipe::TransparentCulled => Pipe::TransparentCulledEnc,
            Pipe::Glass => Pipe::GlassEnc,
            Pipe::GlassCulled => Pipe::GlassCulledEnc,
            Pipe::Overlay => Pipe::OverlayEnc,
            p => p,
        }
    }

    /// Every pipeline, the ones a first frame needs first (the warm-up
    /// thread's order).
    fn all() -> Vec<Pipe> {
        let mut v = vec![
            Pipe::Sky,
            Pipe::Opaque,
            Pipe::OpaqueTwoSided,
            Pipe::Post,
            Pipe::Easu,
            Pipe::Rcas,
            Pipe::Lines,
            Pipe::LinesOnTop,
            Pipe::Shadow,
            Pipe::Transparent,
            Pipe::TransparentCulled,
            Pipe::DecalDepth,
            Pipe::Overlay,
            Pipe::SeeEncode,
            Pipe::SeeDecode,
            Pipe::TransparentEnc,
            Pipe::TransparentCulledEnc,
            Pipe::OverlayEnc,
            Pipe::Terrain,
            Pipe::TerrainShadow,
            Pipe::Glass,
            Pipe::GlassCulled,
            Pipe::GlassEnc,
            Pipe::GlassCulledEnc,
            Pipe::GlassResolve,
            Pipe::Water,
            Pipe::LightShadow,
            Pipe::SsaoDown,
            Pipe::SsaoMain,
            Pipe::SsaoBlurX,
            Pipe::SsaoBlurY,
            Pipe::SsaoApply,
            Pipe::BloomBright,
            Pipe::BloomDown,
            Pipe::BloomUp,
            Pipe::DofH,
            Pipe::DofV,
        ];
        v.extend((0..NEAR_PASSES).map(Pipe::Near));
        v.push(Pipe::TerrainShadowBack);
        v
    }

    /// Whether the settings can draw with this pipeline at all (the
    /// warm-up skips the others; turned on later they compile on first use).
    fn wanted(self, s: &crate::quality::RenderSettings) -> bool {
        match self {
            Pipe::Shadow | Pipe::TerrainShadow => s.sun_cascades > 0,
            Pipe::TerrainShadowBack => false, // (a diagnostic: RUSTUDIO_CULL_BACK)
            Pipe::LightShadow => s.light_shadows,
            Pipe::SsaoDown | Pipe::SsaoMain | Pipe::SsaoBlurX | Pipe::SsaoBlurY | Pipe::SsaoApply => s.ssao,
            Pipe::BloomBright | Pipe::BloomDown | Pipe::BloomUp => s.bloom_levels > 0,
            Pipe::DofH | Pipe::DofV | Pipe::Near(_) => s.dof,
            Pipe::Easu | Pipe::Rcas => s.upscales(),
            // (RUSTUDIO_LINEAR_BLEND=1 never draws the encoded ones)
            Pipe::TransparentEnc | Pipe::TransparentCulledEnc | Pipe::GlassEnc | Pipe::GlassCulledEnc | Pipe::OverlayEnc | Pipe::SeeEncode | Pipe::SeeDecode => {
                std::env::var_os("RUSTUDIO_LINEAR_BLEND").is_none()
            }
            _ => true,
        }
    }
}

/// The driver's pipeline cache kept on disk (wgpu PIPELINE_CACHE, Vulkan):
/// a second start finds its pipelines compiled.
struct DiskCache {
    cache: wgpu::PipelineCache,
    path: std::path::PathBuf,
}

impl DiskCache {
    fn open(device: &wgpu::Device, info: &wgpu::AdapterInfo) -> Option<Self> {
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) || std::env::var_os("RUSTUDIO_NO_PIPELINE_CACHE").is_some() {
            return None;
        }
        let key = wgpu::util::pipeline_cache_key(info)?;
        let path = crate::paths::cache().join(format!("{key}.bin"));
        let data = std::fs::read(&path).ok();
        // SAFETY: the data is what this program wrote from `get_data` of
        // a cache for the same adapter (the key); wgpu and the driver
        // validate its header and fall back to an empty cache otherwise.
        let cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("rustudio pipelines"), data: data.as_deref(), fallback: true })
        };
        Some(Self { cache, path })
    }

    fn save(&self) {
        let Some(data) = self.cache.get_data() else { return };
        if std::fs::read(&self.path).is_ok_and(|old| old == data) {
            return;
        }
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension("tmp");
        if std::fs::write(&tmp, &data).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }
}

/// What every pipeline is made from: the shader module, the layouts, the
/// scene's format and sample count (quality.rs).
struct PipeFactory {
    device: wgpu::Device,
    shader: wgpu::ShaderModule,
    disk: Option<DiskCache>,
    samples: u32,
    scene_format: wgpu::TextureFormat,
    /// Globals + textures.
    layout: wgpu::PipelineLayout,
    /// + the glass scene snapshot (fs_part).
    part_layout: wgpu::PipelineLayout,
    water_layout: wgpu::PipelineLayout,
    lshadow_layout: wgpu::PipelineLayout,
    shadow_layout: wgpu::PipelineLayout,
    terrain_layout: wgpu::PipelineLayout,
    /// Globals + textures + the scene's depth (post, DoF).
    post_layout: wgpu::PipelineLayout,
    /// Globals + the scene's depth (SSAO, the glass depth snapshot).
    ssao_layout: wgpu::PipelineLayout,
    /// Globals + the scene or its encoded copy (SeeThrough).
    see_layout: wgpu::PipelineLayout,
}

impl PipeFactory {
    fn build(&self, p: Pipe) -> wgpu::RenderPipeline {
        use wgpu::CompareFunction as C;
        use wgpu::PrimitiveTopology as T;
        let device = &self.device;
        let shader = &self.shader;
        let cache = self.disk.as_ref().map(|d| &d.cache);
        let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 8 => Float32x2, 12 => Float32x4, 11 => Float32x4];
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attrs,
        };
        let instance_attrs = wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 9 => Float32x4, 10 => Float32x4, 13 => Float32x4];
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance_attrs,
        };
        let line_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let line_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &line_attrs,
        };
        let terrain_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Snorm8x4, 2 => Uint8x4, 3 => Uint8x4, 4 => Uint8x4, 5 => Float32x4];
        let terrain_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::terrain::TerrainVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &terrain_attrs,
        };
        let parts = [Some(vertex_layout.clone()), Some(instance_layout.clone())];
        let ms = wgpu::MultisampleState { count: self.samples, mask: !0, alpha_to_coverage_enabled: false };
        let depth = |write: bool, compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };
        // A scene pass's pipeline: multisampled into the scene's format.
        let make = |label: &str,
                    vs: &str,
                    fs: &str,
                    buffers: &[Option<wgpu::VertexBufferLayout>],
                    topology: wgpu::PrimitiveTopology,
                    cull: Option<wgpu::Face>,
                    depth_write: bool,
                    depth_compare: wgpu::CompareFunction,
                    blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(if fs == "fs_part" || fs == "fs_part_enc" || fs == "fs_decal_depth" { &self.part_layout } else { &self.layout }),
                vertex: wgpu::VertexState { module: shader, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { topology, cull_mode: cull, ..Default::default() },
                depth_stencil: depth(depth_write, depth_compare),
                multisample: ms,
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        // (the encoded twins draw into the see-through copy)
                        format: if fs == "fs_part_enc" { SEE_FORMAT } else { self.scene_format },
                        blend,
                        write_mask: if fs == "fs_decal_depth" { wgpu::ColorWrites::empty() } else { wgpu::ColorWrites::ALL },
                    })],
                }),
                multiview_mask: None,
                cache,
            })
        };
        // A depth-only pass into a shadow map, both sides unless culled
        // (`fs`: a fragment stage and its target's format, for moments).
        let depth_only = |label: &str, layout: &wgpu::PipelineLayout, vs: &str, buffers: &[Option<wgpu::VertexBufferLayout>], cull: Option<wgpu::Face>, fs: Option<(&str, wgpu::TextureFormat)>| {
            let targets = [Some(wgpu::ColorTargetState { format: fs.map_or(LSHADOW_FORMAT, |f| f.1), blend: None, write_mask: wgpu::ColorWrites::ALL })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module: shader, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { topology: T::TriangleList, cull_mode: cull, ..Default::default() },
                depth_stencil: depth(true, C::LessEqual),
                multisample: Default::default(),
                fragment: fs.map(|(entry, _)| wgpu::FragmentState { module: shader, entry_point: Some(entry), compilation_options: Default::default(), targets: &targets }),
                multiview_mask: None,
                cache,
            })
        };
        // A full-screen pass (vs_post) into one target.
        let fullscreen = |label: &str, layout: &wgpu::PipelineLayout, fs: &str, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module: shader, entry_point: Some("vs_post"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache,
            })
        };
        let premul = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        match p {
            Pipe::Sky => make("sky", "vs_sky", "fs_sky", &[], T::TriangleList, None, false, C::Always, None),
            Pipe::Opaque => make("parts", "vs_part", "fs_part", &parts, T::TriangleList, Some(wgpu::Face::Back), true, C::Greater, None),
            Pipe::OpaqueTwoSided => make("parts-two-sided", "vs_part", "fs_part", &parts, T::TriangleList, None, true, C::Greater, None),
            // fs_part writes premultiplied colour (round 26): see-through
            // Neon adds its glow with cover 0, everything else blends as before.
            Pipe::DecalDepth => make("decal opaque texel depth", "vs_part", "fs_decal_depth", &parts, T::TriangleList, None, true, C::Greater, None),
            Pipe::Transparent => make("parts-transparent", "vs_part", "fs_part", &parts, T::TriangleList, None, false, C::GreaterEqual, premul),
            // See-through built-in shapes show their front faces only, as in
            // Studio (round 19: from inside a translucent part it draws nothing).
            Pipe::TransparentCulled => make("parts-transparent-culled", "vs_part", "fs_part", &parts, T::TriangleList, Some(wgpu::Face::Back), false, C::GreaterEqual, premul),
            // GlassTransparentFS already composites the opaque scene. Replace
            // RGB and preserve its energy-derived alpha; ordinary transparents
            // remain premultiplied One / OneMinusSrcAlpha.
            Pipe::Glass => make("glass-composite", "vs_part", "fs_part", &parts, T::TriangleList, None, false, C::Greater, None),
            Pipe::GlassCulled => make("glass-composite-culled", "vs_part", "fs_part", &parts, T::TriangleList, Some(wgpu::Face::Back), false, C::Greater, None),
            // Hover highlight: the same geometry again, on top of itself.
            Pipe::Overlay => make("parts-overlay", "vs_part", "fs_part", &parts, T::TriangleList, None, false, C::GreaterEqual, premul),
            // The same five, drawing into the see-through encoding
            // (shader.wgsl see_out; SeeThrough).
            Pipe::TransparentEnc => make("parts-transparent-enc", "vs_part", "fs_part_enc", &parts, T::TriangleList, None, false, C::GreaterEqual, premul),
            Pipe::TransparentCulledEnc => make("parts-transparent-culled-enc", "vs_part", "fs_part_enc", &parts, T::TriangleList, Some(wgpu::Face::Back), false, C::GreaterEqual, premul),
            Pipe::GlassEnc => make("glass-composite-enc", "vs_part", "fs_part_enc", &parts, T::TriangleList, None, false, C::Greater, None),
            Pipe::GlassCulledEnc => make("glass-composite-culled-enc", "vs_part", "fs_part_enc", &parts, T::TriangleList, Some(wgpu::Face::Back), false, C::Greater, None),
            Pipe::OverlayEnc => make("parts-overlay-enc", "vs_part", "fs_part_enc", &parts, T::TriangleList, None, false, C::GreaterEqual, premul),
            // The scene (per sample) into the encoding, and the covered
            // pixels back (in the pass that also draws the editor's lines:
            // with the scene's depth attached, never tested).
            Pipe::SeeEncode | Pipe::SeeDecode => device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if p == Pipe::SeeEncode { "see-through encode" } else { "see-through decode" }),
                layout: Some(&self.see_layout),
                vertex: wgpu::VertexState { module: shader, entry_point: Some("vs_post"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: if p == Pipe::SeeDecode { depth(false, C::Always) } else { None },
                multisample: ms,
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(if p == Pipe::SeeEncode { "fs_see_encode" } else { "fs_see_decode" }),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: if p == Pipe::SeeEncode { SEE_FORMAT } else { self.scene_format },
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache,
            }),
            Pipe::Lines => make("lines", "vs_line", "fs_line", &[Some(line_layout.clone())], T::LineList, None, false, C::GreaterEqual, Some(wgpu::BlendState::ALPHA_BLENDING)),
            Pipe::LinesOnTop => make("lines-on-top", "vs_line", "fs_line", &[Some(line_layout)], T::LineList, None, false, C::Always, Some(wgpu::BlendState::ALPHA_BLENDING)),
            // Terrain water (water.rs): its own depth attachment, the scene's format.
            Pipe::Water => device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("water"),
                layout: Some(&self.water_layout),
                vertex: wgpu::VertexState { module: shader, entry_point: Some("vs_water"), compilation_options: Default::default(), buffers: &parts },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth(true, C::Greater),
                multisample: ms,
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_water"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: self.scene_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache,
            }),
            // Smooth terrain (round 21): its own vertices (terrain.rs
            // TerrainVertex) and per-palette parameters in group 1.
            Pipe::Terrain => device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("terrain"),
                layout: Some(&self.terrain_layout),
                vertex: wgpu::VertexState { module: shader, entry_point: Some("vs_terrain"), compilation_options: Default::default(), buffers: &[Some(terrain_vertex_layout.clone())] },
                primitive: wgpu::PrimitiveState { topology: T::TriangleList, cull_mode: Some(wgpu::Face::Back), ..Default::default() },
                depth_stencil: depth(true, C::Greater),
                multisample: ms,
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_terrain"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: self.scene_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache,
            }),
            // A cascade of the sun's shadow: depth only, both sides (thin
            // parts). No depth bias: the EVSM test's minimum variance is the bias.
            Pipe::Shadow => depth_only("shadow", &self.shadow_layout, "vs_shadow", &parts, None, None),
            Pipe::TerrainShadow => depth_only("terrain shadow", &self.shadow_layout, "vs_terrain_shadow", &[Some(terrain_vertex_layout.clone())], None, None),
            Pipe::TerrainShadowBack => depth_only("terrain shadow", &self.shadow_layout, "vs_terrain_shadow", &[Some(terrain_vertex_layout.clone())], Some(wgpu::Face::Back), None),
            // Local light shadow faces: EVSM moments + depth, both sides.
            Pipe::LightShadow => depth_only("light shadow", &self.lshadow_layout, "vs_light_shadow", &parts, None, Some(("fs_light_shadow", LSHADOW_FORMAT))),
            Pipe::GlassResolve => fullscreen("glass depth", &self.ssao_layout, "fs_glass_depth", wgpu::TextureFormat::R32Float, None),
            Pipe::SsaoDown => fullscreen("ssao depth", &self.ssao_layout, "fs_ssao_down", wgpu::TextureFormat::R32Float, None),
            Pipe::SsaoMain => fullscreen("ssao", &self.ssao_layout, "fs_ssao", wgpu::TextureFormat::Rg32Float, None),
            Pipe::SsaoBlurX => fullscreen("ssao blur across", &self.ssao_layout, "fs_ssao_blur_x", wgpu::TextureFormat::Rg32Float, None),
            Pipe::SsaoBlurY => fullscreen("ssao blur down", &self.ssao_layout, "fs_ssao_blur_y", wgpu::TextureFormat::Rg32Float, None),
            // SSAOApply: the scene × the occlusion.
            Pipe::SsaoApply => fullscreen(
                "ssao apply",
                &self.ssao_layout,
                "fs_ssao_apply",
                self.scene_format,
                Some(wgpu::BlendState {
                    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::Src, operation: wgpu::BlendOperation::Add },
                    alpha: wgpu::BlendComponent::OVER,
                }),
            ),
            Pipe::BloomBright => fullscreen("bloom bright", &self.layout, "fs_bloom_bright", self.scene_format, None),
            Pipe::BloomDown => fullscreen("bloom pyramid down", &self.layout, "fs_bloom_down", self.scene_format, None),
            Pipe::BloomUp => fullscreen(
                "bloom pyramid up",
                &self.layout,
                "fs_bloom_up",
                self.scene_format,
                Some(wgpu::BlendState {
                    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
            ),
            Pipe::DofH => fullscreen("dof prepare", &self.post_layout, "fs_dof_h", DOF_FORMAT, None),
            Pipe::DofV => fullscreen("dof down", &self.layout, "fs_dof_v", DOF_FORMAT, None),
            Pipe::Near(i) => fullscreen("dof near", &self.post_layout, NEAR_ENTRIES[i.min(NEAR_PASSES - 1)], DOF_FORMAT, None),
            // Exposure, tone mapping, bloom and colour correction into the display texture.
            Pipe::Post => fullscreen("post", &self.post_layout, "fs_post", DISPLAY_FORMAT, None),
            // The display texture scaled up to the panel's size (Target::upscale).
            Pipe::Easu => fullscreen("upscale easu", &self.layout, "fs_easu", DISPLAY_FORMAT, None),
            Pipe::Rcas => fullscreen("upscale rcas", &self.layout, "fs_rcas", DISPLAY_FORMAT, None),
        }
    }
}

/// The 3D view's pipelines, each made on first use (and by the warm-up
/// thread `Renderer::new` starts, in `Pipe::all` order, so a scene opened
/// a few seconds in finds most of them ready). Shared with that thread.
pub struct Pipes {
    factory: PipeFactory,
    cells: [std::sync::OnceLock<wgpu::RenderPipeline>; Pipe::COUNT],
    /// Set when the renderer moved on to another build (`rebuild`): the
    /// warm-up thread stops at its next pipeline.
    dropped: std::sync::atomic::AtomicBool,
}

impl Pipes {
    fn new(factory: PipeFactory) -> Self {
        Self { factory, cells: std::array::from_fn(|_| std::sync::OnceLock::new()), dropped: std::sync::atomic::AtomicBool::new(false) }
    }

    /// Tells a warm-up thread still making these to stop.
    fn abandon(&self) {
        self.dropped.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// The pipeline, made now if it is not yet (another thread making it
    /// already is waited for).
    pub fn get(&self, p: Pipe) -> &wgpu::RenderPipeline {
        self.cells[p.index()].get_or_init(|| {
            let _b = crate::bench::scope("pipeline_build");
            self.factory.build(p)
        })
    }

    /// Makes every pipeline `settings` can use (all of them with None),
    /// then writes the driver's cache to disk.
    pub fn build_all(&self, settings: Option<&crate::quality::RenderSettings>) {
        for p in Pipe::all() {
            if self.dropped.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            if settings.is_none_or(|s| p.wanted(s)) {
                self.get(p);
            }
        }
        if let Some(d) = self.factory.disk.as_ref() {
            d.save();
        }
    }

    /// How many are made so far (the perf log).
    pub fn ready(&self) -> usize {
        self.cells.iter().filter(|c| c.get().is_some()).count()
    }
}

pub struct Renderer {
    /// Skinned parts deformed on the GPU (skin_gpu.rs).
    skin: crate::skin_gpu::SkinGpu,
    /// Highlight instances over the finished image (highlight.rs).
    highlight: Option<crate::highlight::HighlightPass>,
    /// The last frame's view-projection (Highlights draw with it).
    last_view_proj: Mat4,
    /// The quality settings in force (quality.rs; `set_settings`), fitted
    /// to the device's multisampling (`msaa_support`).
    settings: crate::quality::RenderSettings,
    /// The sample counts the device has (quality::MsaaSupport): settings
    /// asking for others are fitted to these, in `new` and `set_settings`.
    msaa_support: crate::quality::MsaaSupport,
    /// The settings the pipelines, layouts and atlases were built for
    /// (`RenderSettings::build_key`): a change rebuilds them.
    built_key: crate::quality::BuildKey,
    /// The scene's sample count and colour format, from the settings.
    samples: u32,
    scene_format: wgpu::TextureFormat,
    /// The viewport shader's pipelines, made on first use.
    pipes: std::sync::Arc<Pipes>,
    /// SurfaceGuis / BillboardGuis tested against the scene's depth (round 23).
    world_gui: Option<crate::world_gui::WorldGuiPass>,
    /// Terrain water (round 24, water.rs).
    water: crate::water::WaterPass,
    glass_layout: wgpu::BindGroupLayout,
    material_epoch: Instant,
    /// Fog for the world GUI pass (gui_fog).
    gui_fog: ([f32; 4], [f32; 3]),
    /// The cascades as the beams read them (matrices, spheres, tiles).
    gui_sun: crate::world_gui::SunCascades,
    gui_lightgrid: [f32; 4],
    gui_sun_params: [f32; 4],
    /// Local lights in lights_buf this frame (the effects read them).
    gui_n_lights: u32,
    post_dof: bool,
    post_dof_near: bool,
    /// Group 2 of the post / SSAO pipelines: the scene's depth, an input
    /// and the scene copy (shader.wgsl "SSAO").
    ssao_layout: wgpu::BindGroupLayout,
    post_bloom_size: f32,
    /// Local light shadow maps (Studio's Future path).
    lshadow: LightShadows,
    /// Bumped whenever the static instances change (light shadow cache key).
    static_gen: u64,
    /// Smooth terrain (round 21): per-chunk triangle buffers, drawn with
    /// Studio's terrain shading, and their per-palette parameters.
    terrain_params_buf: wgpu::Buffer,
    terrain_bg: wgpu::BindGroup,
    terrain_chunks: HashMap<[i32; 3], TerrainChunk>,
    /// Decoration grass blades by chunk (color and sun shadow passes).
    terrain_grass: HashMap<[i32; 3], TerrainChunk>,
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    /// The globals alone, for the shadow pass (its map can't be bound there).
    shadow_globals_bg: wgpu::BindGroup,
    globals_layout: wgpu::BindGroupLayout,
    lights_buf: wgpu::Buffer,
    /// The depth map the sun's cascades are drawn into, one at a time.
    shadow_view: wgpu::TextureView,
    /// The sun's shadow cascades: the atlas the shaders read (sun_shadow.rs).
    sun: crate::sun_shadow::SunShadows,
    /// Terrain.Clouds' passes (clouds.rs), made on first use.
    clouds: Option<crate::clouds::CloudsGpu>,
    sky_view: wgpu::TextureView,
    /// Studio's indoor environment (textures/sky/indoor512); white 1×1 until
    /// loaded. The scene probe's slots start as it (env_probe_gpu).
    indoor_view: wgpu::TextureView,
    /// Studio's scene environment probe (env_probe.rs) and the globals its
    /// faces are drawn with.
    probe: env_probe_gpu::ProbeGpu,
    probe_globals_bg: wgpu::BindGroup,
    /// Studio's split-sum BRDF table (brdfLUT.dds, RG16F).
    brdf_view: wgpu::TextureView,
    sky_sampler: wgpu::Sampler,
    /// Studio's light grid: skylight and sun per 4-stud voxel (lightgrid.rs).
    lightgrid_tex: wgpu::Texture,
    lightgrid_view: wgpu::TextureView,
    localgrid_tex: wgpu::Texture,
    localgrid_view: wgpu::TextureView,
    /// How much of each voxel parts fill, for local lights (lightocc.rs).
    lightocc_tex: wgpu::Texture,
    lightocc_view: wgpu::TextureView,
    /// Studio's legacy surface atlas (studs.dds); grey (no change) until loaded.
    studs_view: wgpu::TextureView,
    studs_sampler: wgpu::Sampler,
    /// Material texture packs (matpack.rs): one array layer per pack,
    /// colour + height and normal + roughness + metalness.
    materials: MaterialArrays,
    /// Material arrays being made at a larger texture size than
    /// `materials` (`remake_materials`), and the layers still to upload.
    materials_next: Option<MaterialArrays>,
    materials_next_pending: Vec<u32>,
    /// Size of the loaded skybox faces (0 = none, draw the gradient).
    pub sky_size: u32,
    /// 1×1 black: the emissive mask of textures without one.
    black_view: wgpu::TextureView,
    meshes: Vec<Mesh>,
    /// (mesh, cull back) -> the second slot drawing it the other way.
    cull_alias: HashMap<(u32, bool), u32>,
    tex_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Texture bind groups; index 0 is plain white.
    textures: Vec<wgpu::BindGroup>,
    /// Each one's views (the texture, the emissive mask, the surface
    /// maps): the bind groups are made again when the sampler changes
    /// (`rebuild`, the anisotropy).
    texture_views: Vec<[wgpu::TextureView; 3]>,
    /// Studio's encoded blend for decals and see-through parts (SeeThrough).
    see: SeeThrough,
    /// Per-frame instances: moving and see-through parts, overlay.
    instance_buf: wgpu::Buffer,
    instance_cap: usize,
    static_buf: wgpu::Buffer,
    static_cap: usize,
    items: HashMap<Ref, Item>,
    /// Items drawn from the per-frame list (pending and dynamic).
    per_frame: HashSet<Ref>,
    /// ViewportFrames' parts (vf_render.rs): never in the world's lists;
    /// each frame's items, and a generation bumped whenever one changes.
    vf: vf::Layers,
    st: StaticLayout,
    pending: usize,
    pending_at: Option<Instant>,
    /// Last static rebuild / quiet-part scan (both rate-limited).
    last_rebuild: Instant,
    last_quiet_scan: Instant,
    pub stats: RenderStats,
    line_buf: wgpu::Buffer,
    line_cap: usize,
    target: Option<Target>,
    /// The target before the last size change, kept for the camera's
    /// starting and stopping (the motion scale's size and the render
    /// scale's by turns): taken back, not made again.
    spare_target: Option<Target>,
    pub texture_id: Option<egui::TextureId>,
    /// Counts what changes the picture besides the frame's own inputs
    /// (meshes, textures, items, terrain, sky, grids): the viewport shows
    /// the last frame again while this and [`frame_key`] stand still.
    generation: u64,
    /// The last frame drew terrain water (its waves move with time).
    water_drawn: bool,
    /// GPU time per section, when the device can tell (the perf log).
    gpu_timer: Option<GpuTimer>,
    /// A frame's commands kept for egui's submit (`one_submit_wanted`,
    /// `submit_callback`), until submitted by it or `flush_deferred`.
    deferred: std::sync::Arc<std::sync::Mutex<Option<wgpu::CommandBuffer>>>,
    /// A frame was encoded whose `after_submit` has not run yet.
    post_submit_pending: bool,
}

/// The paint callback that hands a kept frame to egui's submit
/// (`Renderer::submit_callback`): `prepare` returns the commands, egui
/// submits them in front of its own; `paint` has nothing to draw (the
/// view's image is painted as usual).
struct SubmitDeferred(std::sync::Arc<std::sync::Mutex<Option<wgpu::CommandBuffer>>>);

impl egui_wgpu::CallbackTrait for SubmitDeferred {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).take().into_iter().collect()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, _pass: &mut wgpu::RenderPass<'static>, _resources: &egui_wgpu::CallbackResources) {}
}

impl Renderer {
    /// The sample counts `adapter` multisamples every one of the renderer's
    /// formats at: the two scene formats (quality.rs scene_format), the
    /// depth, the see-through copy (SEE_FORMAT) and the highlight masks.
    pub fn msaa_support_of(adapter: &wgpu::Adapter) -> crate::quality::MsaaSupport {
        let mut formats = vec![wgpu::TextureFormat::Rgba16Float, DEPTH_FORMAT, SEE_FORMAT, crate::highlight::MASK_FORMAT];
        // (the 32-bit scene format only where the device renders to it, as
        // quality.rs scene_format picks it; elsewhere it has no counts at all)
        if adapter.features().contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE) {
            formats.push(wgpu::TextureFormat::Rg11b10Ufloat);
        }
        crate::quality::MsaaSupport::of(adapter, &formats)
    }

    /// `info`: the adapter, for the pipeline cache kept on disk (None: no
    /// cache); `msaa_support`: the sample counts it has (`Renderer::
    /// msaa_support_of`), which the settings are fitted to. The pipelines
    /// themselves are made on first use and by a background thread;
    /// `build_all_pipelines` makes them all now.
    pub fn new(
        device: &wgpu::Device, queue: &wgpu::Queue, info: Option<&wgpu::AdapterInfo>,
        msaa_support: crate::quality::MsaaSupport, settings: crate::quality::RenderSettings,
    ) -> Self {
        let asked = settings.normalized();
        let settings = asked.fitted(msaa_support);
        if settings.msaa != asked.msaa {
            eprintln!("render settings: {}x MSAA for the level's {}x (the device has no {}x)", settings.msaa, asked.msaa, asked.msaa);
        }
        let samples = settings.msaa;
        let scene_format = settings.scene_format(device);
        crate::sun_shadow::set_texel_scale(settings.shadow_scale);
        let shader = {
            let _b = crate::bench::scope("shader_module");
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rustudio viewport"),
                source: wgpu::ShaderSource::Wgsl(shader_source(&settings).into()),
            })
        };
        let uniform_entry = Self::uniform_entry();
        let frag = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                uniform_entry,
                frag(
                    1,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                frag(
                    2,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                ),
                frag(
                    3,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                // The sun's shadow cascades: EVSM moments (sun_shadow.rs).
                frag(
                    4,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                frag(
                    7,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                frag(
                    8,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                frag(
                    9,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                ),
                frag(
                    10,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                ),
                frag(
                    11,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                ),
                frag(
                    12,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                ),
                // Studio's indoor environment cube (PrefilteredEnvIndoorTexture).
                frag(
                    13,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                ),
                // PrefilteredEnvBlendTargetTexture: the scene probe's next
                // slot (env_probe.rs), blended in by CB0[32].w (env_spec.w).
                frag(
                    21,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                ),
                // PrecomputedBRDFTexture (PlatformContent textures/brdfLUT.dds).
                frag(
                    14,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                // Local light shadows: EVSM moment atlas, its sampler, the slots.
                frag(
                    15,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                frag(
                    16,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                frag(
                    17,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                frag(
                    19,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                ),
                frag(20, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3, multisampled: false,
                }),
                // Studio's per-material constants and detiling noise (MaterialArrays).
                frag(
                    18,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow globals"),
            entries: &[uniform_entry],
        });
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights"),
            size: (crate::mech::MAX_LIGHTS.max(1) * std::mem::size_of::<LightRaw>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // The cascades' depth map: the largest tile across (a phone's
        // tiles are smaller, sun_shadow::texel_scale).
        let shadow_side = crate::sun_shadow::depth_texels();
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sun shadow map"),
            size: wgpu::Extent3d {
                width: shadow_side,
                height: shadow_side,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());
        let [lg_x, lg_y, lg_z] = crate::lightgrid::dims().map(|n| n as u32);
        let lightgrid_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light grid"),
            size: wgpu::Extent3d {
                width: lg_x,
                height: lg_z,
                depth_or_array_layers: lg_y,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lightgrid_view = lightgrid_tex.create_view(&Default::default());
        let localgrid_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel local lights"),
            size: wgpu::Extent3d { width: lg_x, height: lg_z, depth_or_array_layers: lg_y },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let localgrid_view = localgrid_tex.create_view(&Default::default());
        let [lo_x, lo_y, lo_z] = crate::lightocc::dims().map(|n| n as u32);
        let lightocc_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light occupancy"),
            size: wgpu::Extent3d {
                width: lo_x,
                height: lo_y,
                depth_or_array_layers: lo_z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lightocc_view = lightocc_tex.create_view(&Default::default());
        let sky_view = Self::cube_view(
            device,
            queue,
            1,
            &[
                [255u8; 4].to_vec(),
                vec![255; 4],
                vec![255; 4],
                vec![255; 4],
                vec![255; 4],
                vec![255; 4],
            ],
        );
        let indoor_view = Self::cube_view(
            device,
            queue,
            1,
            &[vec![0u8, 0, 0, 255], vec![0, 0, 0, 255], vec![0, 0, 0, 255], vec![0, 0, 0, 255], vec![0, 0, 0, 255], vec![0, 0, 0, 255]],
        );
        let brdf_view = Self::brdf_lut_view(device, queue);
        let sky_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let studs_view = Self::studs_texture(device, queue, 1, 1, &[128, 128, 128, 255]);
        let materials = MaterialArrays::new(device, 1, 1);
        // (also the material maps' sampler)
        let studs_sampler = Self::repeat_sampler(device, "studs", settings.anisotropy);
        let lshadow = LightShadows::new(device, settings.light_shadows);
        let sun = crate::sun_shadow::SunShadows::new(device, &shadow_view);
        // Studio's scene environment probe (env_probe.rs): binding 13 is its
        // current slot, 21 the slot it blends to.
        let probe = env_probe_gpu::ProbeGpu::new(device, queue, samples, scene_format, DEPTH_FORMAT, std::mem::size_of::<Globals>() as u64);
        let probe_globals_bg = Self::globals_group(
            device,
            &bgl,
            &probe.globals_buf,
            &lights_buf,
            &sky_view,
            &sky_sampler,
            &sun.atlas_view,
            &lightgrid_view,
            &lightocc_view,
            &localgrid_view,
            &studs_view,
            &studs_sampler,
            &materials,
            &probe.slot_views[0],
            &probe.slot_views[1],
            &brdf_view,
            &lshadow,
        );
        let globals_bg = Self::globals_group(
            device,
            &bgl,
            &globals_buf,
            &lights_buf,
            &sky_view,
            &sky_sampler,
            &sun.atlas_view,
            &lightgrid_view,
            &lightocc_view,
            &localgrid_view,
            &studs_view,
            &studs_sampler,
            &materials,
            &probe.slot_views[0],
            &probe.slot_views[1],
            &brdf_view,
            &lshadow,
        );
        let shadow_globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow globals"),
            layout: &shadow_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });
        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // SurfaceAppearance emissive mask (black = none).
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
            ],
        });
        let black_view = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("no emission"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[0, 0, 0, 255],
            )
            .create_view(&Default::default());
        let sampler = Self::repeat_sampler(device, "texture", settings.anisotropy);
        let glass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glass scene"),
            entries: &[
                frag(0, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                frag(1, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport"),
            bind_group_layouts: &[Some(&bgl), Some(&tex_layout)],
            immediate_size: 0,
        });
        let part_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("parts with glass scene"),
            bind_group_layouts: &[Some(&bgl), Some(&tex_layout), Some(&glass_layout)],
            immediate_size: 0,
        });
        let water = crate::water::WaterPass::new(device, queue, &bgl, &tex_layout, samples, settings.anisotropy);
        let see = SeeThrough::new(device, samples);
        let see_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("see-through"),
            bind_group_layouts: &[Some(&bgl), Some(&see.layout)],
            immediate_size: 0,
        });
        let lshadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light shadow"),
            bind_group_layouts: &[Some(&shadow_bgl), Some(&lshadow.face_layout)],
            immediate_size: 0,
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[Some(&shadow_bgl), Some(&sun.pass_layout)],
            immediate_size: 0,
        });
        let terrain_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let terrain_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain params"),
            size: (TERRAIN_PALETTE * 4 * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let terrain_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain params"),
            layout: &terrain_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_params_buf.as_entire_binding(),
            }],
        });
        let terrain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain"),
            bind_group_layouts: &[Some(&bgl), Some(&terrain_bgl)],
            immediate_size: 0,
        });
        let ssao_layout = Self::depth_layout(device, samples);
        let post_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post with depth"),
            bind_group_layouts: &[Some(&bgl), Some(&tex_layout), Some(&ssao_layout)],
            immediate_size: 0,
        });
        let ssao_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao"),
            bind_group_layouts: &[Some(&bgl), Some(&ssao_layout)],
            immediate_size: 0,
        });
        let pipes = std::sync::Arc::new(Pipes::new(PipeFactory {
            device: device.clone(),
            shader,
            disk: info.and_then(|i| DiskCache::open(device, i)),
            samples,
            scene_format,
            layout,
            part_layout,
            water_layout: water.layout.clone(),
            lshadow_layout,
            shadow_layout,
            terrain_layout,
            post_layout,
            ssao_layout: ssao_pl,
            see_layout,
        }));

        let meshes = build_meshes()
            .into_iter()
            .map(|verts| Mesh {
                cull: Some(true),
                storage: false,
                count: verts.len() as u32,
                vbuf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            })
            .collect();

        let instance_cap = 1024;
        let line_cap = 4096;
        let mut r = Renderer {
            skin: crate::skin_gpu::SkinGpu::new(device),
            built_key: settings.build_key(),
            settings,
            msaa_support,
            samples,
            scene_format,
            pipes,
            gui_fog: ([0.0, 1e9, 0.0, 0.0], [0.0; 3]),
            gui_sun: Default::default(),
            gui_lightgrid: [0.0; 4],
            gui_sun_params: [0.0, 580.0, 0.05, 0.0],
            gui_n_lights: 0,
            world_gui: None,
            highlight: None,
            last_view_proj: Mat4::IDENTITY,
            water,
            glass_layout,
            material_epoch: Instant::now(),
            post_dof: false,
            post_dof_near: false,
            ssao_layout,
            post_bloom_size: 24.0,
            terrain_params_buf,
            terrain_bg,
            terrain_chunks: HashMap::new(),
            terrain_grass: HashMap::new(),
            globals_buf,
            globals_bg,
            shadow_globals_bg,
            lshadow,
            static_gen: 0,
            globals_layout: bgl,
            lights_buf,
            shadow_view,
            sun,
            clouds: None,
            sky_view,
            indoor_view,
            probe,
            probe_globals_bg,
            brdf_view,
            sky_sampler,
            lightgrid_tex,
            lightgrid_view,
            localgrid_tex,
            localgrid_view,
            lightocc_tex,
            lightocc_view,
            studs_view,
            studs_sampler,
            materials,
            materials_next: None,
            materials_next_pending: Vec::new(),
            sky_size: 0,
            meshes,
            cull_alias: HashMap::new(),
            tex_layout,
            sampler,
            textures: Vec::new(),
            texture_views: Vec::new(),
            black_view,
            see,
            instance_buf: Self::make_buf(
                device,
                "instances",
                instance_cap * std::mem::size_of::<InstanceRaw>(),
            ),
            instance_cap,
            static_buf: Self::make_buf(
                device,
                "static instances",
                instance_cap * std::mem::size_of::<InstanceRaw>(),
            ),
            static_cap: instance_cap,
            items: HashMap::new(),
            per_frame: HashSet::new(),
            vf: Default::default(),
            st: StaticLayout::default(),
            pending: 0,
            pending_at: None,
            last_rebuild: Instant::now(),
            last_quiet_scan: Instant::now(),
            stats: RenderStats::default(),
            line_buf: Self::make_buf(
                device,
                "lines",
                line_cap * std::mem::size_of::<LineVertex>(),
            ),
            line_cap,
            target: None,
            spare_target: None,
            texture_id: None,
            generation: 0,
            water_drawn: false,
            gpu_timer: GpuTimer::new(device, queue),
            deferred: Default::default(),
            post_submit_pending: false,
        };
        r.add_texture(device, queue, 1, 1, &[255, 255, 255, 255]);
        r.warm_pipelines();
        r
    }

    /// The part textures' and material maps' sampler: repeating,
    /// mipmapped and trilinear, like Studio's (finely tiled textures
    /// average out at a distance instead of sparkling), with the quality
    /// settings' anisotropy (`RenderSettings::anisotropy`; Studio's device
    /// default is mech::TEXTURE_ANISOTROPY).
    fn repeat_sampler(device: &wgpu::Device, label: &str, anisotropy: u16) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: anisotropy.max(1),
            ..Default::default()
        })
    }

    /// Binding 0 of the globals (and the shadow passes' globals-only group).
    fn uniform_entry() -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Group 2 of the post and SSAO pipelines: the scene's depth (as drawn:
    /// multisampled or not), a pass's input and the scene copy.
    fn depth_layout(device: &wgpu::Device, samples: u32) -> wgpu::BindGroupLayout {
        let tex = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty, count: None };
        let float = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao"),
            entries: &[
                tex(0, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: samples > 1 }),
                tex(1, float),
                tex(2, float),
            ],
        })
    }

    /// Starts the thread that makes the pipelines the settings can use,
    /// most needed first (`Pipe::all`), so the first scene drawn finds
    /// them ready instead of compiling them in its frame. Not in tests
    /// (they build on the calling thread and check the errors there).
    fn warm_pipelines(&self) {
        if cfg!(test) || std::env::var_os("RUSTUDIO_NO_PIPELINE_WARMUP").is_some() {
            return;
        }
        let pipes = std::sync::Arc::clone(&self.pipes);
        let settings = self.settings.clone();
        let _ = rt_thread::Builder::new().name("rustudio-pipelines".into()).spawn(move || {
            let t0 = Instant::now();
            pipes.build_all(Some(&settings));
            if crate::perf_log::on() {
                eprintln!("pipelines: {} ready after {:.1} s", pipes.ready(), t0.elapsed().as_secs_f32());
            }
        });
    }

    /// Makes every pipeline now, on this thread (the tests: every
    /// pipeline through the GPU's compiler with the errors caught there).
    #[cfg(test)]
    pub fn build_all_pipelines(&self) {
        self.pipes.build_all(None);
    }

    /// The pipeline for a pass, made on first use.
    fn pipe(&self, p: Pipe) -> &wgpu::RenderPipeline {
        self.pipes.get(p)
    }

    /// Applies new quality settings. Pass switches (SSAO, bloom, DoF,
    /// clouds, cascades, the probe) take effect in the next frame; the
    /// targets are remade when what they hold changes; MSAA, the scene's
    /// format, the shadow scale and the light shadow atlas
    /// (`RenderSettings::build_key`) rebuild the pipelines, the shadow
    /// atlases and the environment probe (its capture starts over).
    pub fn set_settings(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, settings: &crate::quality::RenderSettings) {
        let asked = settings.normalized();
        let settings = asked.fitted(self.msaa_support);
        if settings == self.settings {
            return;
        }
        if settings.msaa != asked.msaa && crate::perf_log::on() {
            eprintln!("render settings: {}x MSAA for the level's {}x (the device has no {}x)", settings.msaa, asked.msaa, asked.msaa);
        }
        let targets_differ = settings.ssao != self.settings.ssao
            || settings.dof != self.settings.dof
            || settings.upscale != self.settings.upscale
            || (settings.bloom_levels > 0) != (self.settings.bloom_levels > 0)
            || settings.bloom_levels.max(1) != self.settings.bloom_levels.max(1);
        let textures_differ = settings.texture_texels != self.settings.texture_texels;
        self.settings = settings;
        self.generation = self.generation.wrapping_add(1);
        if textures_differ && !self.materials.cpu.is_empty() {
            self.remake_materials(device, queue, self.materials.capacity);
        }
        if self.settings.build_key() != self.built_key {
            self.rebuild(device, queue);
            return;
        }
        if targets_differ {
            self.target = None;
            self.spare_target = None;
        }
        if !self.settings.clouds {
            self.clouds = None;
        }
    }

    /// Remakes what the build key covers (see `set_settings`), keeping the
    /// meshes, textures, items, terrain and grids.
    fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let settings = self.settings.clone();
        let samples = settings.msaa;
        let scene_format = settings.scene_format(device);
        crate::sun_shadow::set_texel_scale(settings.shadow_scale);
        if crate::perf_log::on() {
            eprintln!("render settings: rebuilding for {}", settings.describe(scene_format));
        }
        // (the same cache object serves the new pipelines)
        let disk = self.pipes.factory.disk.as_ref().map(|d| DiskCache { cache: d.cache.clone(), path: d.path.clone() });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rustudio viewport"),
            source: wgpu::ShaderSource::Wgsl(shader_source(&settings).into()),
        });
        let shadow_side = crate::sun_shadow::depth_texels();
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow"),
            size: wgpu::Extent3d { width: shadow_side, height: shadow_side, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.shadow_view = shadow_tex.create_view(&Default::default());
        self.sun = crate::sun_shadow::SunShadows::new(device, &self.shadow_view);
        self.lshadow = LightShadows::new(device, settings.light_shadows);
        self.probe = env_probe_gpu::ProbeGpu::new(device, queue, samples, scene_format, DEPTH_FORMAT, std::mem::size_of::<Globals>() as u64);
        self.water = crate::water::WaterPass::new(device, queue, &self.globals_layout, &self.tex_layout, samples, settings.anisotropy);
        // The samplers' anisotropy: the globals group is rebound below; the
        // part textures' bind groups are made again over their views.
        if settings.anisotropy != self.built_key.5 {
            self.studs_sampler = Self::repeat_sampler(device, "studs", settings.anisotropy);
            self.sampler = Self::repeat_sampler(device, "texture", settings.anisotropy);
            for (bg, views) in self.textures.iter_mut().zip(&self.texture_views) {
                *bg = Self::texture_bind_group(device, &self.tex_layout, &self.sampler, views);
            }
        }
        self.ssao_layout = Self::depth_layout(device, samples);
        self.see = SeeThrough::new(device, samples);
        self.highlight = None;
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("shadow globals"), entries: &[Self::uniform_entry()] });
        let pipeline_layout = |label: &str, groups: &[&wgpu::BindGroupLayout]| {
            let groups: Vec<_> = groups.iter().map(|g| Some(*g)).collect();
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &groups, immediate_size: 0 })
        };
        let terrain_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let factory = PipeFactory {
            device: device.clone(),
            shader,
            disk,
            samples,
            scene_format,
            layout: pipeline_layout("viewport", &[&self.globals_layout, &self.tex_layout]),
            part_layout: pipeline_layout("parts with glass scene", &[&self.globals_layout, &self.tex_layout, &self.glass_layout]),
            water_layout: self.water.layout.clone(),
            lshadow_layout: pipeline_layout("light shadow", &[&shadow_bgl, &self.lshadow.face_layout]),
            shadow_layout: pipeline_layout("shadow", &[&shadow_bgl, &self.sun.pass_layout]),
            terrain_layout: pipeline_layout("terrain", &[&self.globals_layout, &terrain_bgl]),
            post_layout: pipeline_layout("post with depth", &[&self.globals_layout, &self.tex_layout, &self.ssao_layout]),
            ssao_layout: pipeline_layout("ssao", &[&self.globals_layout, &self.ssao_layout]),
            see_layout: pipeline_layout("see-through", &[&self.globals_layout, &self.see.layout]),
        };
        self.pipes.abandon();
        self.pipes = std::sync::Arc::new(Pipes::new(factory));
        self.samples = samples;
        self.scene_format = scene_format;
        self.built_key = settings.build_key();
        self.target = None;
        self.spare_target = None;
        self.clouds = None;
        self.world_gui = None;
        self.rebind_globals(device);
        self.warm_pipelines();
    }

    /// The last frame as shown (RGBA8, top row first): width, height,
    /// pixels. Waits for the GPU; for tests.
    pub fn read_frame(&self, rs: &egui_wgpu::RenderState) -> Option<(u32, u32, Vec<u8>)> {
        let target = self.target.as_ref()?;
        self.flush_deferred(&rs.queue);
        let [w, h] = target.size;
        let row = (w * 4).div_ceil(256) * 256;
        let buf = rs.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read frame"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = rs
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read frame"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.display,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        rs.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rs.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range().ok()?;
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let o = (y * row) as usize;
            out.extend_from_slice(&data[o..o + (w * 4) as usize]);
        }
        drop(data);
        buf.unmap();
        Some((w, h, out))
    }

    /// Bumps whenever the static scene is rebuilt: the scene environment
    /// probe's world revision (env_probe.rs).
    pub fn static_gen(&self) -> u64 {
        self.static_gen
    }

    /// Raw RGBA16Float scene before bloom/tone, for render diagnostics.
    pub fn read_hdr(&self, rs: &egui_wgpu::RenderState) -> Option<(u32, u32, Vec<u8>)> {
        let target = self.target.as_ref()?;
        if self.scene_format != wgpu::TextureFormat::Rgba16Float {
            return None; // (the readers expect half floats: HDR settings only)
        }
        self.flush_deferred(&rs.queue);
        {
            if let Ok(path) = std::env::var("RUSTUDIO_DUMP_HDR") {
                let half = [(target.size[0] >> 1).max(1), (target.size[1] >> 1).max(1)];
                if let Some(bloom) = target.bloom.as_ref() {
                    if let Some(texture) = &bloom.bright_capture {
                        if let Some((w,h,data)) = self.read_hdr_texture(rs, texture, half) {
                            let _ = std::fs::write(format!("{path}.extract-{w}x{h}"), data);
                        }
                    }
                    if let Some((w,h,data)) = self.read_hdr_texture(rs, &bloom.textures[0], half) {
                        let _ = std::fs::write(format!("{path}.bloom-{w}x{h}"), data);
                    }
                }
                // The scene environment probe's current slot, level 0 (env_probe.rs).
                for f in 0..6 {
                    let size = [crate::env_probe::SIZE; 2];
                    if let Some((w,h,data)) = self.read_hdr_layer(rs, &self.probe.slots[self.probe.cur], size, f) {
                        let _ = std::fs::write(format!("{path}.probe{f}-{w}x{h}"), data);
                    }
                }
            }
        }
        self.read_hdr_texture(rs, &target.hdr, target.size)
    }

    fn read_hdr_texture(&self, rs: &egui_wgpu::RenderState, texture: &wgpu::Texture, size: [u32; 2]) -> Option<(u32,u32,Vec<u8>)> {
        self.read_hdr_layer(rs, texture, size, 0)
    }

    fn read_hdr_layer(&self, rs: &egui_wgpu::RenderState, texture: &wgpu::Texture, size: [u32; 2], layer: u32) -> Option<(u32,u32,Vec<u8>)> {
        let [w, h] = size;
        let row = (w * 8).div_ceil(256) * 256;
        let buf = rs.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read HDR"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = rs
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read HDR"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        rs.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rs.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range().ok()?;
        let mut out = Vec::with_capacity((w * h * 8) as usize);
        for y in 0..h {
            let o = (y * row) as usize;
            out.extend_from_slice(&data[o..o + (w * 8) as usize]);
        }
        drop(data);
        buf.unmap();
        Some((w, h, out))
    }

    /// Uploads an RGBA8 (sRGB) image; returns its index for [`PartDraw::texture`].
    pub fn add_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> u32 {
        self.add_texture_emissive(device, queue, width, height, rgba, None)
    }

    /// A texture with a SurfaceAppearance emissive mask (RGBA8, its own size;
    /// read as data, not sRGB).
    pub fn add_texture_emissive(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
        mask: Option<(u32, u32, &[u8])>,
    ) -> u32 {
        self.add_texture_surface(device, queue, width, height, rgba, mask, None)
    }

    pub fn add_texture_surface(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue,
        width: u32, height: u32, rgba: &[u8],
        mask: Option<(u32, u32, &[u8])>, surface: Option<(u32, u32, &[u8])>,
    ) -> u32 {
        self.generation += 1;
        let (sw, sh, sp) = surface.unwrap_or((1, 1, &[128, 128, 255, 0]));
        let surface_view = device.create_texture_with_data(queue, &wgpu::TextureDescriptor {
            label: Some("SurfaceAppearance normal roughness metalness"),
            size: wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 },
            mip_level_count: mip_levels(sw, sh), sample_count: 1,
            dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        }, wgpu::util::TextureDataOrder::LayerMajor, &data_mip_chain(sw, sh, sp))
            .create_view(&Default::default());
        let mask_view = mask.map(|(w, h, px)| {
            device
                .create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("emissive mask"),
                        size: wgpu::Extent3d {
                            width: w.max(1),
                            height: h.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    px,
                )
                .create_view(&Default::default())
        });
        let (w, h) = (width.max(1), height.max(1));
        let ready = READY_MIPS.with(|c| {
            let mut c = c.borrow_mut();
            match c.as_ref() {
                Some((p, _)) if *p == rgba.as_ptr() as usize => c.take().map(|m| m.1),
                _ => None,
            }
        });
        let mips = if rgba.len() == (w * h * 4) as usize {
            ready.unwrap_or_else(|| mip_chain(w, h, rgba))
        } else {
            rgba.to_vec()
        };
        let levels = if rgba.len() == (w * h * 4) as usize {
            mip_levels(w, h)
        } else {
            1
        };
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("part texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &mips,
        );
        let view = tex.create_view(&Default::default());
        let views = [view, mask_view.unwrap_or_else(|| self.black_view.clone()), surface_view];
        let bg = Self::texture_bind_group(device, &self.tex_layout, &self.sampler, &views);
        // An EditableImage texture re-uploaded: into its old slot (the old
        // bind group and its textures are freed), not a new one.
        if let Some(i) = REUSE_TEXTURE_SLOT.with(|c| c.take()) {
            if let (Some(slot), Some(v)) = (self.textures.get_mut(i as usize), self.texture_views.get_mut(i as usize)) {
                *slot = bg;
                *v = views;
                return i;
            }
        }
        self.textures.push(bg);
        self.texture_views.push(views);
        (self.textures.len() - 1) as u32
    }

    /// Group 1 of the part pipelines for a texture: it, the sampler, its
    /// emissive mask (or black) and its SurfaceAppearance maps.
    fn texture_bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, views: &[wgpu::TextureView; 3]) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("part texture"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&views[0]) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&views[1]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&views[2]) },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn globals_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        lights: &wgpu::Buffer,
        sky: &wgpu::TextureView,
        sky_sampler: &wgpu::Sampler,
        shadow: &wgpu::TextureView,
        lightgrid: &wgpu::TextureView,
        lightocc: &wgpu::TextureView,
        localgrid: &wgpu::TextureView,
        studs: &wgpu::TextureView,
        studs_sampler: &wgpu::Sampler,
        materials: &MaterialArrays,
        indoor: &wgpu::TextureView,
        indoor_next: &wgpu::TextureView,
        brdf: &wgpu::TextureView,
        lshadow: &LightShadows,
    ) -> wgpu::BindGroup {
        use wgpu::BindingResource as B;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: B::TextureView(sky),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: B::Sampler(sky_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: B::TextureView(shadow),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: B::TextureView(studs),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: B::Sampler(studs_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: B::TextureView(&materials.albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: B::TextureView(&materials.surface_view),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: B::TextureView(lightgrid),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: B::TextureView(lightocc),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: B::TextureView(indoor),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: B::TextureView(indoor_next),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: B::TextureView(brdf),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: B::TextureView(&lshadow.moments_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: B::Sampler(&lshadow.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: lshadow.slots_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: materials.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 20, resource: B::TextureView(localgrid) },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: B::TextureView(&materials.emission_view),
                },
            ],
        })
    }

    /// Studio's split-sum environment BRDF table: PlatformContent
    /// textures/brdfLUT.dds, 256 × 256 R16G16_FLOAT (DX10 header), read by the
    /// part shaders at (roughness, N·V) as PrecomputedBRDFTexture. Without the
    /// file a 1×1 (1, 0): Fresnel F0 alone.
    fn brdf_lut_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
        let file = crate::assets::resolve("rbxasset://textures/brdfLUT.dds", None)
            .and_then(|p| std::fs::read(p).ok());
        let parsed = file.and_then(|b| {
            if b.len() < 148 || &b[0..4] != b"DDS " || &b[84..88] != b"DX10" {
                return None;
            }
            let h = u32::from_le_bytes(b[12..16].try_into().ok()?);
            let w = u32::from_le_bytes(b[16..20].try_into().ok()?);
            let fmt = u32::from_le_bytes(b[128..132].try_into().ok()?);
            let n = (w * h * 4) as usize;
            // DXGI_FORMAT_R16G16_FLOAT = 34.
            (fmt == 34 && b.len() >= 148 + n).then(|| (w, h, b[148..148 + n].to_vec()))
        });
        // (f16 1.0 = 0x3C00)
        let (w, h, data) = parsed.unwrap_or((1, 1, vec![0x00, 0x3C, 0, 0]));
        device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("brdf lut"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rg16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &data,
            )
            .create_view(&Default::default())
    }

    /// Sets Studio's indoor environment cube (textures/sky/indoor512, faces
    /// in wgpu's cube order like `set_sky`).
    pub fn set_indoor(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, size: u32, faces: &[Vec<u8>; 6]) {
        self.generation += 1;
        self.indoor_view = Self::cube_view(device, queue, size, faces);
        // Both probe slots start as this cube (env_probe_gpu fill_default).
        self.probe.fill_default(device, queue, &self.indoor_view, size);
        self.rebind_globals(device);
    }

    /// A cube texture from six square RGBA8 faces in wgpu's order
    /// (+X, −X, +Y, −Y, +Z, −Z). Texels stay gamma-encoded; the shader
    /// squares them as Studio does.
    fn cube_view(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        faces: &[Vec<u8>; 6],
    ) -> wgpu::TextureView {
        // Each face with its mip chain (layer-major): the sky's reflection
        // on rough parts reads a blurred level (shader.wgsl sky_rough).
        let data = cube_mips(size, faces);
        Self::cube_view_mipped(device, queue, size, &data)
    }

    /// A cube texture from faces already with their mip chains (cube_mips).
    fn cube_view_mipped(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        data: &[u8],
    ) -> wgpu::TextureView {
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("sky"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 6,
                },
                mip_level_count: mip_levels(size, size),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );
        tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        })
    }

    fn rebind_globals(&mut self, device: &wgpu::Device) {
        self.globals_bg = Self::globals_group(
            device,
            &self.globals_layout,
            &self.globals_buf,
            &self.lights_buf,
            &self.sky_view,
            &self.sky_sampler,
            &self.sun.atlas_view,
            &self.lightgrid_view,
            &self.lightocc_view,
            &self.localgrid_view,
            &self.studs_view,
            &self.studs_sampler,
            &self.materials,
            &self.probe.slot_views[self.probe.cur],
            &self.probe.slot_views[self.probe.next()],
            &self.brdf_view,
            &self.lshadow,
        );
        self.probe_globals_bg = Self::globals_group(
            device,
            &self.globals_layout,
            &self.probe.globals_buf,
            &self.lights_buf,
            &self.sky_view,
            &self.sky_sampler,
            &self.sun.atlas_view,
            &self.lightgrid_view,
            &self.lightocc_view,
            &self.localgrid_view,
            &self.studs_view,
            &self.studs_sampler,
            &self.materials,
            &self.probe.slot_views[self.probe.cur],
            &self.probe.slot_views[self.probe.next()],
            &self.brdf_view,
            &self.lshadow,
        );
    }

    /// The legacy surface atlas (RGBA8, kept as data: the shader reads its
    /// grey 0.5 as "no change"), with mips averaged in linear light.
    fn studs_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        rgba: &[u8],
    ) -> wgpu::TextureView {
        device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("studs"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: mip_levels(w, h),
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &mip_chain(w, h, rgba),
            )
            .create_view(&Default::default())
    }

    /// Studio's studs.dds (PlatformContent/pc/textures): 128 px wide, four
    /// 512 px bands (studs, weld, inlet, universal), 2 × 8 studs each.
    pub fn set_studs(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        rgba: &[u8],
    ) {
        self.generation += 1;
        self.studs_view = Self::studs_texture(device, queue, w, h, rgba);
        self.rebind_globals(device);
    }

    /// Replaces one terrain chunk's triangles (none = the chunk goes).
    pub fn set_terrain_chunk(
        &mut self,
        device: &wgpu::Device,
        key: [i32; 3],
        verts: &[crate::terrain::TerrainVertex],
    ) {
        self.generation += 1;
        // (the sun's cascades and the lights' shadow maps hold the terrain)
        self.static_gen += 1;
        if verts.is_empty() {
            self.terrain_chunks.remove(&key);
            return;
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in verts {
            lo = lo.min(Vec3::from(v.pos));
            hi = hi.max(Vec3::from(v.pos));
        }
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain chunk"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.terrain_chunks.insert(
            key,
            TerrainChunk {
                grass_ranges: Vec::new(),
                diagnostic_positions: if std::env::var_os("RUSTUDIO_CULL_EXPORT").is_some() { verts.iter().map(|v| v.pos).collect() } else { Vec::new() },
                vbuf,
                count: verts.len() as u32,
                lo,
                hi,
            },
        );
    }

    /// Drops every terrain chunk (another place, or no terrain).
    pub fn clear_terrain(&mut self) {
        self.generation += 1;
        self.terrain_chunks.clear();
        self.terrain_grass.clear();
        self.static_gen += 1;
    }

    /// Replaces one chunk's decoration grass (none = drop it).
    pub fn set_terrain_grass(
        &mut self,
        device: &wgpu::Device,
        key: [i32; 3],
        verts: &[crate::terrain::TerrainVertex],
    ) {
        let had = self.terrain_chunks.remove(&key);
        self.set_terrain_chunk(device, key, verts);
        match self.terrain_chunks.remove(&key) {
            Some(mut c) => {
                // |noise| <= 1; native default direction=.5, amplitude=1.0275.
                let wind = verts.iter().map(|v| 0.2 * v.wind_local[2]).fold(0.0_f32, f32::max);
                let pad = Vec3::new(wind, 0.0, wind);
                c.lo -= pad;
                c.hi += pad;
                self.terrain_grass.insert(key, c);
            }
            None => {
                self.terrain_grass.remove(&key);
            }
        }
        if let Some(h) = had {
            self.terrain_chunks.insert(key, h);
        }
    }

    pub fn set_terrain_grass_geometry(&mut self, device: &wgpu::Device, key: [i32;3], geometry: crate::terrain::GrassGeometry) {
        let mut vertices = Vec::new();
        let mut ranges = Vec::new();
        for cell in geometry.cells {
            let start = vertices.len() as u32;
            let mut ordinals = Vec::new();
            for (ordinal, first) in cell.batch.blades {
                ordinals.push(ordinal);
                vertices.extend_from_slice(&geometry.vertices[first..first+6]);
            }
            ranges.push(GrassRange {start, ordinals, allocated:cell.batch.allocated, near:cell.near, lo:cell.lo, hi:cell.hi});
        }
        if ranges.is_empty() { vertices = geometry.vertices; }
        self.set_terrain_grass(device, key, &vertices);
        if let Some(chunk) = self.terrain_grass.get_mut(&key) {chunk.grass_ranges = ranges;}
    }

    /// Triangles in the drawn terrain chunks.
    pub fn terrain_triangles(&self) -> usize {
        self.terrain_chunks.values().map(|c| c.count as usize / 3).sum()
    }

    /// The terrain shader's per-palette parameters (4 vec4s per entry, see
    /// shader.wgsl TerrainParams).
    pub fn set_terrain_params(&mut self, queue: &wgpu::Queue, params: &[[f32; 4]]) {
        self.generation += 1;
        let n = params.len().min(TERRAIN_PALETTE * 4);
        queue.write_buffer(&self.terrain_params_buf, 0, bytemuck::cast_slice(&params[..n]));
    }

    /// The texture array layer of a material pack, uploading it (and
    /// growing the arrays) the first time. -1 while its maps are not in
    /// memory (released after another renderer's upload): they are asked
    /// for, and `matpack::generation` changes when they are back.
    pub fn material_layer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        maps: &std::sync::Arc<crate::matpack::Maps>,
    ) -> i32 {
        if let Some(l) = self.materials.layers.get(&maps.pack) {
            return *l as i32;
        }
        let Some(pixels) = maps.pixels() else {
            crate::matpack::request_pixels(maps);
            return -1;
        };
        self.generation += 1;
        let layer = self.materials.cpu.len() as u32;
        // Room for this layer first (the arrays remade copy the `layer`
        // layers uploaded so far: this one isn't among them yet). Arrays
        // already being made at the settings' size with room take it as
        // they are.
        let next_fits = self.materials_next.as_ref().is_some_and(|n| n.size == self.texture_size() && layer < n.capacity);
        if layer >= self.materials.capacity || (self.materials.size != self.texture_size() && !next_fits) {
            let cap = (layer + 1).next_power_of_two().max(4).max(self.materials.capacity);
            self.remake_materials(device, queue, cap);
        }
        self.materials.cpu.push(maps.clone());
        self.materials.layers.insert(maps.pack, layer);
        self.materials.upload(queue, layer, &pixels);
        // The arrays being filled at a larger size get it too (below).
        if let Some(next) = self.materials_next.as_mut() {
            next.cpu.push(maps.clone());
            next.layers.insert(maps.pack, layer);
            next.upload(queue, layer, &pixels);
            self.materials_next_pending.retain(|&l| l != layer);
        }
        // On the GPU now: the 14 MB of maps go (see matpack::Maps).
        maps.release_pixels();
        layer as i32
    }

    /// The material arrays' side: the settings' texture quality
    /// (`RenderSettings::texture_texels`), at most the packs' own size.
    fn texture_size(&self) -> u32 {
        self.settings.texture_texels.clamp(1, crate::matpack::LAYER_SIZE)
    }

    /// New material arrays of `capacity` layers at the settings' texture
    /// size. The layers so far (`materials.cpu`, every one uploaded) are
    /// copied on the GPU (their maps are no longer in memory): whole at
    /// the same size, from the matching level down at a smaller one.
    /// Larger arrays can't be filled from the old ones' levels: they are
    /// made aside (`materials_next`), every pack's maps are asked for
    /// again, and they take over once every layer has been uploaded
    /// (`materials_tick`); the old arrays draw meanwhile.
    fn remake_materials(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, capacity: u32) {
        self.generation = self.generation.wrapping_add(1);
        let size = self.texture_size();
        // (never more layers than the old arrays have: a copy past their
        // end is a wgpu validation error, which is fatal)
        let done = (self.materials.cpu.len() as u32).min(self.materials.capacity);
        let capacity = capacity.max(1).max(done);
        if size > self.materials.size && done > 0 {
            // Growing: the arrays aside, at the new size.
            let mut next = MaterialArrays::with_params(device, size, capacity, self.materials.params.clone());
            next.layers = self.materials.layers.clone();
            next.cpu = self.materials.cpu.clone();
            for m in &next.cpu {
                crate::matpack::request_pixels(m);
            }
            self.materials_next = Some(next);
            self.materials_next_pending = (0..done).collect();
            if done >= self.materials.capacity {
                // (room for the layer being added, at the old size)
                let old = std::mem::replace(&mut self.materials, MaterialArrays::new(device, 1, 1));
                let mut grown = MaterialArrays::with_params(device, old.size, capacity, old.params.clone());
                grown.layers = old.layers.clone();
                grown.cpu = old.cpu.clone();
                grown.copy_layers(device, queue, &old, done);
                self.materials = grown;
                self.rebind_globals(device);
            }
            return;
        }
        let params = std::mem::take(&mut self.materials.params);
        let mut made = MaterialArrays::with_params(device, size, capacity, params);
        made.layers = std::mem::take(&mut self.materials.layers);
        made.cpu = std::mem::take(&mut self.materials.cpu);
        made.copy_layers(device, queue, &self.materials, done);
        self.materials = made;
        self.materials_next = None;
        self.materials_next_pending.clear();
        self.rebind_globals(device);
    }

    /// Once a frame: the arrays being made at a larger size take the maps
    /// that have come back, and take over when every layer has them.
    fn materials_tick(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let size = self.texture_size();
        let Some(next) = self.materials_next.as_mut() else { return };
        if next.size != size {
            // The setting moved again meanwhile: start over from the
            // current arrays (a smaller size needs no maps at all).
            self.materials_next = None;
            self.materials_next_pending.clear();
            if self.materials.size != size {
                self.remake_materials(device, queue, self.materials.capacity);
            }
            return;
        }
        let mut pending = std::mem::take(&mut self.materials_next_pending);
        pending.retain(|&layer| {
            let Some(m) = next.cpu.get(layer as usize) else { return false };
            match m.pixels() {
                Some(pixels) => {
                    next.upload(queue, layer, &pixels);
                    m.release_pixels();
                    false
                }
                None => {
                    crate::matpack::request_pixels(m);
                    true
                }
            }
        });
        self.materials_next_pending = pending;
        if self.materials_next_pending.is_empty() {
            let mut next = self.materials_next.take().expect("checked");
            next.params = std::mem::take(&mut self.materials.params);
            queue.write_buffer(&next.params_buf, 0, bytemuck::cast_slice(&next.params));
            self.materials = next;
            self.generation = self.generation.wrapping_add(1);
            self.rebind_globals(device);
        }
    }

    /// Studio's material constants for the parts drawn with material layer
    /// `layer` (see `MaterialArrays`): written when they change.
    pub fn set_material_params(&mut self, queue: &wgpu::Queue, layer: i32, p: [[f32; 4]; 2]) {
        self.generation += 1;
        let Ok(l) = usize::try_from(layer) else {
            return;
        };
        if l >= MAT_PARAM_LAYERS { return; }
        let mut full = [p[0], p[1], [1.0, 1.0, 1.0, 0.0], [0.0; 4], [1.0, 1.0, 1.0, 0.0]];
        full[1][2] = 4.0;
        full[1][3] = 1.0;
        if let Some(c) = self.materials.cpu.get(l).and_then(|m| crate::matpack::full_consts(m.pack)) {
            full[1][2] = c[0];
            full[1][3] = c[16];
            full[2] = [c[6], c[7], c[8], c[17]];
            full[3] = [c[9], c[10], c[11], c[18]];
            full[4] = [c[13], c[14], c[15], c[19]];
        }
        if self.materials.params[l * 5..l * 5 + 5] == full { return; }
        self.materials.params[l * 5..l * 5 + 5].copy_from_slice(&full);
        queue.write_buffer(&self.materials.params_buf, (l * 5 * 16) as u64, bytemuck::cast_slice(&full));
    }

    /// Sets the skybox: six square RGBA8 faces of `size` in wgpu's cube
    /// order (+X, −X, +Y, −Y, +Z, −Z). `size` 0 = no skybox.
    /// `mipped`: the six faces with their mip chains, layer-major (cube_mips).
    pub fn set_sky(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        mipped: Option<&[u8]>,
    ) {
        self.generation += 1;
        match mipped {
            Some(f) if size > 0 => {
                self.sky_view = Self::cube_view_mipped(device, queue, size, f);
                self.sky_size = size;
            }
            _ => self.sky_size = 0,
        }
        self.rebind_globals(device);
    }

    /// Uploads the light grid (lightgrid.rs: skylight and sun bytes, X, then Z, then Y).
    pub fn set_lightgrid(&mut self, queue: &wgpu::Queue, data: &[u8]) {
        self.generation += 1;
        let [nx, ny, nz] = crate::lightgrid::dims().map(|n| n as u32);
        if data.len() != (2 * nx * ny * nz) as usize {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.lightgrid_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2 * nx),
                rows_per_image: Some(nz),
            },
            wgpu::Extent3d {
                width: nx,
                height: nz,
                depth_or_array_layers: ny,
            },
        );
    }

    pub fn set_localgrid(&mut self, queue: &wgpu::Queue, data: &[u8]) {
        self.generation += 1;
        let [nx, ny, nz] = crate::lightgrid::dims().map(|n| n as u32);
        if data.len() != (nx * ny * nz * 4) as usize { return; }
        queue.write_texture(wgpu::TexelCopyTextureInfo {
            texture: &self.localgrid_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }, data, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(nx * 4), rows_per_image: Some(nz) },
        wgpu::Extent3d { width: nx, height: nz, depth_or_array_layers: ny });
    }

    /// Uploads part of the local light channel (lightgrid::LocalPatch).
    pub fn set_localgrid_region(&mut self, queue: &wgpu::Queue, p: &crate::lightgrid::LocalPatch) {
        self.generation += 1;
        let size = [0, 1, 2].map(|k| (p.hi[k] - p.lo[k] + 1) as u32);
        if p.data.len() != (size[0] * size[1] * size[2] * 4) as usize { return; }
        queue.write_texture(wgpu::TexelCopyTextureInfo {
            texture: &self.localgrid_tex, mip_level: 0,
            origin: wgpu::Origin3d { x: p.lo[0] as u32, y: p.lo[2] as u32, z: p.lo[1] as u32 },
            aspect: wgpu::TextureAspect::All,
        }, &p.data, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size[0] * 4), rows_per_image: Some(size[2]) },
        wgpu::Extent3d { width: size[0], height: size[2], depth_or_array_layers: size[1] });
    }

    /// Uploads the parts' light occupancy (lightocc.rs: X, then Y, then Z).
    pub fn set_lightocc(&mut self, queue: &wgpu::Queue, data: &[u8]) {
        self.generation += 1;
        let [nx, ny, nz] = crate::lightocc::dims().map(|n| n as u32);
        if data.len() != (nx * ny * nz) as usize {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.lightocc_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(nx),
                rows_per_image: Some(ny),
            },
            wgpu::Extent3d {
                width: nx,
                height: ny,
                depth_or_array_layers: nz,
            },
        );
    }

    fn make_buf(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(64) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn ensure_target(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2], out: [u32; 2]) {
        if self.target.as_ref().is_some_and(|t| t.size == size && t.out_size == out) {
            return;
        }
        let device = &rs.device;
        if self.spare_target.as_ref().is_some_and(|t| t.size == size && t.out_size == out) {
            std::mem::swap(&mut self.target, &mut self.spare_target);
            let t = self.target.as_ref().expect("the spare target, just swapped in");
            let shown = t.upscale.as_ref().map_or(&t.display_view, |u| &u.view);
            if let Some(id) = self.texture_id {
                rs.renderer.write().update_egui_texture_from_wgpu_texture(device, shown, wgpu::FilterMode::Linear, id);
            }
            return;
        }
        let (samples, scene_format) = (self.samples, self.scene_format);
        let tex_at = |label: &str, size: [u32; 2], format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let tex = |label: &str, format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages| tex_at(label, size, format, samples, usage);
        let attach = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        // (the multisampled scene is read by the see-through encode)
        let msaa = (samples > 1).then(|| tex("viewport-msaa", scene_format, samples, attach).create_view(&Default::default()));
        // (the depth is read by the world GUI pass, water, glass, SSAO, DoF)
        let depth = tex("viewport-depth", DEPTH_FORMAT, samples, attach);
        let hdr = tex("viewport-hdr", scene_format, 1, attach | wgpu::TextureUsages::COPY_SRC);
        let hdr_view = hdr.create_view(&Default::default());
        // The see-through encoding's copy of the scene, at the scene's
        // sample count (SeeThrough): read from the multisampled scene, or
        // the scene itself without MSAA.
        let see_view = tex("viewport-see-through", SEE_FORMAT, samples, attach).create_view(&Default::default());
        let see_encode_bg = self.see.bind(device, msaa.as_ref().unwrap_or(&hdr_view));
        let see_decode_bg = self.see.bind(device, &see_view);
        let scene_copy = tex("viewport-scene-copy", scene_format, 1, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let scene_copy_view = scene_copy.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let water_depth = tex("water surface depth", DEPTH_FORMAT, samples, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let water_bg = self.water.bind(device, &scene_copy_view, &depth_view);
        let glass_depth = tex("glass depth snapshot", wgpu::TextureFormat::R32Float, 1, attach).create_view(&Default::default());
        let glass_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glass scene snapshot"),
            layout: &self.glass_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scene_copy_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&glass_depth) },
            ],
        });
        let black = &self.black_view;
        // A post pass's inputs (group 1): a texture, the sampler, two more.
        let pair = |label: &str, a: &wgpu::TextureView, b: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.tex_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(a) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sky_sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(b) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(black) },
                ],
            })
        };
        // Group 2: the scene's depth, a pass's input, the scene copy.
        let depth_group = |label: &str, input: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.ssao_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&depth_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&scene_copy_view) },
                ],
            })
        };
        // Bloom: half resolution down a level at a time.
        let bloom = (self.settings.bloom_levels > 0).then(|| {
            let n = (self.settings.bloom_levels as usize).min(BLOOM_LEVELS);
            let textures: Vec<_> = (1..=n)
                .map(|level| tex_at("bloom pyramid", [size[0] >> level, size[1] >> level], scene_format, 1, attach | wgpu::TextureUsages::COPY_SRC))
                .collect();
            let levels: Vec<_> = textures.iter().map(|t| t.create_view(&Default::default())).collect();
            let bright_capture = std::env::var_os("RUSTUDIO_DUMP_HDR").map(|_| {
                tex_at("diagnostic bloom extraction", [size[0] >> 1, size[1] >> 1], scene_format, 1, wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST)
            });
            BloomTargets {
                view: levels[0].clone(),
                bright_capture,
                down_bgs: (0..n).map(|i| pair("bloom pyramid down", if i == 0 { &hdr_view } else { &levels[i - 1] }, &hdr_view)).collect(),
                up_bgs: (1..n).map(|i| pair("bloom pyramid up", &levels[i], &hdr_view)).collect(),
                textures,
                levels,
            }
        });
        // Depth of field. Native CheapBokeh's far path keeps full-resolution
        // colour / CoC: downsampling here would blur even as FarIntensity
        // approaches zero. The near field works at quarter resolution.
        let dof = self.settings.dof.then(|| {
            let views = [
                tex("dof across", DOF_FORMAT, 1, attach).create_view(&Default::default()),
                tex("dof down", DOF_FORMAT, 1, attach).create_view(&Default::default()),
            ];
            let near_views: Vec<_> = (0..NEAR_PASSES)
                .map(|i| tex_at("dof near quarter", [size[0] / 4, if i == 0 { size[1] } else { size[1] / 4 }], DOF_FORMAT, 1, attach).create_view(&Default::default()))
                .collect();
            let near_bgs = (0..NEAR_PASSES)
                .map(|i| {
                    let a = if i == 0 { &views[1] } else if i == 8 { &near_views[1] } else { &near_views[i - 1] };
                    let b = if i == 8 || i == 9 { &near_views[7] } else { black };
                    pair("dof near", a, b)
                })
                .collect();
            DofTargets {
                bgs: [pair("dof across", &hdr_view, black), pair("dof down", &views[0], &near_views[NEAR_PASSES - 1])],
                views,
                near_bgs,
                near_views,
            }
        });
        let hdr_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post inputs"),
            layout: &self.tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sky_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(bloom.as_ref().map_or(black, |b| &b.view)) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(dof.as_ref().map_or(black, |d| &d.views[1])) },
            ],
        });
        let post_depth_bg = depth_group("post depth and near", dof.as_ref().map_or(black, |d| &d.near_views[NEAR_PASSES - 1]));
        let depth_bg = depth_group("scene depth", black);
        let half = [size[0].div_ceil(2), size[1].div_ceil(2)];
        let ssao = self.settings.ssao.then(|| {
            let depth_view = tex_at("ssao depth", half, wgpu::TextureFormat::R32Float, 1, attach).create_view(&Default::default());
            let views = [
                tex_at("ssao a", half, wgpu::TextureFormat::Rg32Float, 1, attach).create_view(&Default::default()),
                tex_at("ssao b", half, wgpu::TextureFormat::Rg32Float, 1, attach).create_view(&Default::default()),
            ];
            // down (the depth alone) → AO depth; occlusion reads it → a;
            // blur across a → b; blur down b → a; apply reads a.
            let bgs = [depth_group("ssao", &depth_view), depth_group("ssao blur across", &views[0]), depth_group("ssao blur down", &views[1])];
            SsaoTargets { depth_view, views, bgs }
        });
        let display = tex("viewport", DISPLAY_FORMAT, 1, attach | wgpu::TextureUsages::COPY_SRC);
        let display_view = display.create_view(&Default::default());
        // Drawn under the panel's size: EASU into a panel-sized texture,
        // RCAS into another, and egui shows that one.
        let upscale = (out != size && self.settings.upscale).then(|| {
            let easu_view = tex_at("viewport upscaled", out, DISPLAY_FORMAT, 1, attach).create_view(&Default::default());
            let view = tex_at("viewport sharpened", out, DISPLAY_FORMAT, 1, attach).create_view(&Default::default());
            UpscaleTargets { easu_bg: pair("upscale easu", &display_view, black), rcas_bg: pair("upscale rcas", &easu_view, black), easu_view, view }
        });
        let shown = upscale.as_ref().map_or(&display_view, |u| &u.view);
        let mut egui_renderer = rs.renderer.write();
        match self.texture_id {
            Some(id) => egui_renderer.update_egui_texture_from_wgpu_texture(device, shown, wgpu::FilterMode::Linear, id),
            None => self.texture_id = Some(egui_renderer.register_native_texture(device, shown, wgpu::FilterMode::Linear)),
        }
        self.spare_target = self.target.take();
        self.target = Some(Target {
            size,
            out_size: out,
            upscale,
            msaa,
            see: see_view,
            see_encode_bg,
            see_decode_bg,
            depth: depth_view,
            hdr,
            hdr_view,
            scene_copy_view,
            scene_copy,
            water_bg,
            water_depth,
            glass_bg,
            glass_depth,
            hdr_bg,
            post_depth_bg,
            depth_bg,
            dof,
            bloom,
            ssao,
            display,
            display_view,
        });
    }

    /// World GUIs and effects enter the HDR scene before bloom, DoF and
    /// colour correction (UILitFogFS / ParticleCustomFS). AlwaysOnTop and
    /// screen GUIs are composed separately by egui.
    pub fn draw_world_gui(
        &mut self,
        rs: &egui_wgpu::RenderState,
        batches: &[crate::world_gui::GuiBatch],
        fx: Option<&crate::world_gui::FxFrame>,
        tone_curve: Option<[f32; 5]>,
    ) -> bool {
        let Some(t) = self.target.as_ref() else { return false };
        // (drawn over this frame's scene: a frame kept for egui's submit
        // goes first)
        if !batches.iter().all(|b| b.indices.is_empty()) || fx.is_some_and(|f| !f.draws.is_empty()) {
            self.flush_deferred(&rs.queue);
        }
        let (scene_format, samples) = (self.scene_format, self.samples);
        let pass = self
            .world_gui
            .get_or_insert_with(|| crate::world_gui::WorldGuiPass::new(&rs.device, &rs.queue, DISPLAY_FORMAT, scene_format, samples));
        pass.sun_atlas = self.sun.atlas_view.clone();
        pass.sun_grid = self.lightgrid_view.clone();
        pass.lightgrid = self.gui_lightgrid;
        pass.sun_params = self.gui_sun_params;
        pass.sun_cascades = self.gui_sun;
        // (effects light themselves from the frame's local lights)
        pass.lights = self.lights_buf.clone();
        pass.n_lights = self.gui_n_lights;
        let drew = pass.effects(rs, &t.hdr_view, t.size, &t.depth, batches, fx, tone_curve, self.gui_fog.0, self.gui_fog.1);
        if drew {
            self.finish_again(rs);
        }
        drew
    }

    /// See the `generation` field: bumped by everything that changes what
    /// a frame would draw other than the frame's own inputs.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether the last frame drew terrain water (which moves with time).
    pub fn water_drawn(&self) -> bool {
        self.water_drawn
    }

    /// The 3D target's size in pixels, once drawn.
    pub fn target_size(&self) -> Option<[u32; 2]> {
        self.target.as_ref().map(|t| t.size)
    }

    /// One line for the perf log: the settings in force and how many of
    /// the pipelines they can use are compiled (`Pipes`).
    pub fn describe(&self) -> String {
        let wanted = Pipe::all().into_iter().filter(|p| p.wanted(&self.settings)).count();
        let mut out = format!("{}; pipelines {} of {} built", self.settings.describe(self.scene_format), self.pipes.ready(), wanted);
        if let Some(t) = self.gpu_timer.as_ref() {
            out.push_str(&format!("; GPU timer tick {:.2} ns", t.period_ns));
        }
        if self.defers(bisect::step()) {
            out.push_str("; one submit a frame");
        }
        if let Some(name) = bisect::name() {
            out.push_str(&format!("; GPU bisect step {}: {name}", bisect::step()));
        }
        out
    }

    /// GPU milliseconds summed over the frames measured so far (a value
    /// per section, `GpuTimer::NAMES`, then the whole) and their count;
    /// None when the device has no timestamp queries (see [`GpuTimer`]).
    pub fn gpu_totals(&self) -> Option<([f64; GpuTimer::SECTIONS + 1], u32)> {
        self.gpu_timer.as_ref().map(|t| t.totals())
    }

    /// The GPU time of the last 3D frame the timestamps came back for
    /// (ms); None without timestamps (Automatic quality, app.rs).
    pub fn gpu_last_frame_ms(&self) -> Option<f32> {
        self.gpu_timer.as_ref().and_then(|t| t.last_ms())
    }

    /// Highlights over the finished image (highlight.rs): each one's parts
    /// as drawn this frame (their deformed / clothed meshes included).
    pub fn draw_highlights(&mut self, rs: &egui_wgpu::RenderState, list: &[crate::highlight::HighlightDraw]) {
        if list.is_empty() {
            return;
        }
        // (drawn over this frame's image: a frame kept for egui's submit
        // goes first)
        self.flush_deferred(&rs.queue);
        let Some(t) = self.target.as_ref() else { return };
        let samples = self.samples;
        let pass = self.highlight.get_or_insert_with(|| {
            crate::highlight::HighlightPass::new(&rs.device, DISPLAY_FORMAT, DEPTH_FORMAT, samples, std::mem::size_of::<Vertex>() as u64)
        });
        let mut work = Vec::new();
        for h in list.iter().take(crate::highlight::MAX) {
            let mut meshes = Vec::new();
            for r in &h.items {
                let Some(item) = self.items.get(r) else { continue };
                for d in &item.draws {
                    // (an invisible part, and decals: the part's own shape is enough)
                    if d.alpha <= 0.0 || d.tex_mode == TexMode::Decal {
                        continue;
                    }
                    let Some(m) = self.meshes.get(mesh_slot(d)) else { continue };
                    let raw = to_raw(d, 0.0);
                    let bias = item.bias;
                    meshes.push((&m.vbuf, m.count, crate::highlight::Inst { r0: raw.r0, r1: raw.r1, r2: raw.r2, size: [raw.size[0], raw.size[1], raw.size[2], bias] }));
                }
            }
            if !meshes.is_empty() {
                work.push((h, meshes));
            }
        }
        pass.draw(&rs.device, &rs.queue, t.size, samples, &t.depth, &t.display_view, self.last_view_proj, &work);
        // (drawn at the render size: a view scaled up to the panel is
        // scaled up again, with them)
        if t.upscale.is_some() {
            let mut encoder = rs.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("highlights upscale") });
            self.encode_upscale(&mut encoder, t);
            rs.queue.submit([encoder.finish()]);
        }
    }

    /// The bloom pyramid: bright pass into level 0, down a level at a
    /// time, then back up adding. Nothing without bloom in the settings
    /// (the post pass then adds black). The levels used grow with the
    /// view's size and the bloom's, up to the settings' count.
    fn encode_bloom(&self, encoder: &mut wgpu::CommandEncoder, target: &Target, size: f32) {
        let Some(bloom) = target.bloom.as_ref() else { return };
        let scale = ((target.size[0] as f32 * target.size[1] as f32) / 2073600.0).sqrt();
        let levels = ((3.0 + (scale.log2() + (size / 7.0).log2().max(0.0)).max(-1.0)).min(5.999) as usize).min(bloom.levels.len());
        for i in 0..levels {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom pyramid down"), color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &bloom.levels[i], depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(self.pipe(if i == 0 { Pipe::BloomBright } else { Pipe::BloomDown }));
            pass.set_bind_group(0, &self.globals_bg, &[]); pass.set_bind_group(1, &bloom.down_bgs[i], &[]); pass.draw(0..3, 0..1);
            drop(pass);
            if i == 0 {
                if let Some(texture) = &bloom.bright_capture {
                    encoder.copy_texture_to_texture(bloom.textures[0].as_image_copy(), texture.as_image_copy(), texture.size());
                }
            }
        }
        for i in (1..levels).rev() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom pyramid up"), color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &bloom.levels[i-1], depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(self.pipe(Pipe::BloomUp)); pass.set_bind_group(0, &self.globals_bg, &[]); pass.set_bind_group(1, &bloom.up_bgs[i-1], &[]); pass.draw(0..3, 0..1);
        }
    }

    /// One full-screen pass (vs_post) into `view`, cleared first.
    fn fullscreen_pass(&self, encoder: &mut wgpu::CommandEncoder, label: &str, pipe: Pipe, bg: &wgpu::BindGroup, view: &wgpu::TextureView, depth: Option<&wgpu::BindGroup>) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(self.pipe(pipe));
        pass.set_bind_group(0, &self.globals_bg, &[]);
        pass.set_bind_group(1, bg, &[]);
        if let Some(d) = depth {
            pass.set_bind_group(2, d, &[]);
        }
        pass.draw(0..3, 0..1);
    }

    /// Depth of field: the far field, then the near field when `near`.
    /// Nothing without it in the settings (no targets) or the frame.
    fn encode_dof(&self, encoder: &mut wgpu::CommandEncoder, target: &Target, dof: bool, near: bool) {
        let Some(d) = target.dof.as_ref().filter(|_| dof) else { return };
        self.fullscreen_pass(encoder, "dof across", Pipe::DofH, &d.bgs[0], &d.views[0], Some(&target.depth_bg));
        self.fullscreen_pass(encoder, "dof down", Pipe::DofV, &d.bgs[1], &d.views[1], None);
        if near {
            for i in 0..NEAR_PASSES {
                self.fullscreen_pass(encoder, "dof near", Pipe::Near(i), &d.near_bgs[i], &d.near_views[i], Some(&target.depth_bg));
            }
        }
    }

    /// Exposure, tone mapping, bloom and colour correction into the
    /// display texture; then, drawn under the panel's size, EASU and RCAS
    /// up to it.
    fn encode_post(&self, encoder: &mut wgpu::CommandEncoder, target: &Target) {
        self.fullscreen_pass(encoder, "post", Pipe::Post, &target.hdr_bg, &target.display_view, Some(&target.post_depth_bg));
        self.encode_upscale(encoder, target);
    }

    /// The display texture scaled up to the panel (Target::upscale), when
    /// the settings have the upscale.
    fn encode_upscale(&self, encoder: &mut wgpu::CommandEncoder, target: &Target) {
        if let Some(u) = target.upscale.as_ref() {
            self.fullscreen_pass(encoder, "upscale easu", Pipe::Easu, &u.easu_bg, &u.easu_view, None);
            self.fullscreen_pass(encoder, "upscale rcas", Pipe::Rcas, &u.rcas_bg, &u.view, None);
        }
    }

    /// Whether this frame's commands wait for egui's submit (one
    /// vkQueueSubmit a frame instead of two: on a tiling GPU every submit
    /// ends in a flush and an idle gap). `RUSTUDIO_ONE_SUBMIT=0|1`
    /// decides; otherwise on a phone — and the bisection's step 9 takes
    /// the other way.
    fn one_submit_wanted(&self, bisect: usize) -> bool {
        static WANT: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        let want = *WANT.get_or_init(|| std::env::var_os("RUSTUDIO_ONE_SUBMIT").map(|v| v != "0"));
        let default = want.unwrap_or(cfg!(target_os = "android"));
        if bisect == 9 {
            !default
        } else {
            default
        }
    }

    /// Whether this frame's commands are kept for egui's submit: the
    /// bisection's step 8 submits them itself, to wait for the GPU.
    fn defers(&self, bisect: usize) -> bool {
        self.one_submit_wanted(bisect) && bisect != 8
    }

    /// Submits a frame kept for egui's submit, if one is still waiting
    /// (the view was not painted after all, or something else must draw
    /// over it first). In order: nothing else was submitted in between
    /// that this frame should come after.
    pub fn flush_deferred(&self, queue: &wgpu::Queue) {
        let buffer = self.deferred.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(buffer) = buffer {
            let _submit = crate::bench::scope("vpr_submit");
            queue.submit([buffer]);
        }
    }

    /// The work that follows a frame's submit (readbacks asked for, the
    /// timer's readings taken in): once per frame, after it was really
    /// submitted — right away, or on the next frame for one handed to egui.
    fn after_submit(&mut self, device: &wgpu::Device) {
        if !self.post_submit_pending {
            return;
        }
        if self.deferred.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
            return;
        }
        self.post_submit_pending = false;
        if let Some(c) = self.clouds.as_mut() {
            c.after_submit();
        }
        if let Some(t) = self.gpu_timer.as_mut() {
            t.after_submit(device);
        }
    }

    /// For the frame kept for egui's submit: a paint callback whose
    /// `prepare` hands egui the frame's commands, which egui submits in
    /// front of its own (egui_wgpu::winit::Painter: one submit). Added to
    /// the painter before the view's image (viewport.rs); None when no
    /// frame waits.
    pub fn submit_callback(&self, rect: egui::Rect) -> Option<egui::Shape> {
        if self.deferred.lock().unwrap_or_else(|e| e.into_inner()).is_none() {
            return None;
        }
        Some(egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(rect, SubmitDeferred(self.deferred.clone()))))
    }

    fn finish_again(&self, rs: &egui_wgpu::RenderState) {
        let _b = crate::bench::scope("fx_post_again");
        let Some(target) = self.target.as_ref() else { return };
        self.flush_deferred(&rs.queue);
        let mut encoder = rs.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("effects post") });
        self.encode_bloom(&mut encoder, target, self.post_bloom_size);
        self.encode_dof(&mut encoder, target, self.post_dof, self.post_dof_near);
        self.encode_post(&mut encoder, target);
        rs.queue.submit([encoder.finish()]);
    }

    /// The last frame's fog (`Globals::fog`) and its linear colour, as
    /// shader.wgsl fog_rgb works it out (world GUIs are fogged the same).
    fn gui_fog(g: &Globals) -> ([f32; 4], [f32; 3]) {
        let c = [g.fog_color[0], g.fog_color[1], g.fog_color[2]];
        if g.fog[2] > 1.5 && g.fog[2] < 2.5 {
            let k = |i: usize| g.ambient[i] + g.outdoor[i] + g.sun_color[i] * g.sun_dir[3] * g.sun_dir[1].max(0.0);
            return (g.fog, [c[0] * k(0), c[1] * k(1), c[2] * k(2)]);
        }
        if g.fog[2] > 2.5 {
            // Atmosphere: the haze band at the horizon (the sky behind a
            // world GUI is close to it at the distances fog shows).
            return (g.fog, [g.atmos[0], g.atmos[1], g.atmos[2]]);
        }
        (g.fog, c)
    }

    /// Renders a frame at `size` physical pixels for a panel of `out`
    /// pixels (the frame is scaled up to it when they differ); returns the
    /// texture to show.
    pub fn render(
        &mut self,
        rs: &egui_wgpu::RenderState,
        size: [u32; 2],
        out: [u32; 2],
        frame: &Frame,
    ) -> Option<egui::TextureId> {
        let svc_t0 = Instant::now();
        let size = [size[0].clamp(1, 8192), size[1].clamp(1, 8192)];
        let out = [out[0].clamp(1, 8192), out[1].clamp(1, 8192)];
        self.ensure_target(rs, size, out);
        let device = &rs.device;
        let queue = &rs.queue;
        // A frame handed to egui's submit that nothing submitted (the view
        // was not painted) goes first; then the work that follows a submit.
        self.flush_deferred(queue);
        self.after_submit(device);
        self.materials_tick(device, queue);
        let bisect = bisect::step();
        if bisect == 10 {
            // No 3D frame: the view keeps its last image.
            return self.texture_id;
        }
        // cam_pos.w is unused by the camera math. Keep material animation
        // continuous across scene rebuilds, independently of lighting time.
        let mut globals = frame.globals;
        self.last_view_proj = Mat4::from_cols_array_2d(&frame.globals.view_proj);
        if std::env::var_os("RUSTUDIO_RENDER_DIAGNOSTICS").is_some() {
            static LAST: std::sync::Mutex<Option<[f32; 4]>> = std::sync::Mutex::new(None);
            let state = [globals.cc_tint[3], globals.bloom[1], globals.bloom[0], globals.params[1]];
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            if *last != Some(state) {
                eprintln!("BLOOM intensity={} size={} threshold={} exposure={} dof={:?} pipeline=bright/down/up/composite-before-tone", state[0],state[1],state[2],state[3],globals.dof);
                *last = Some(state);
            }
        }
        // Scene environment probe (env_probe.rs): swap the slots when the
        // blend completed; CB0[32].w is the blend toward the next slot.
        globals.env_spec[3] = 0.0;
        if let Some(pf) = frame.probe.as_ref() {
            if pf.swap {
                self.probe.cur = self.probe.next();
                self.rebind_globals(device);
            }
            globals.env_spec[3] = pf.blend;
        }
        // The quality settings' pass switches (quality.rs): a pass that is
        // off has no targets and nothing reads them; the globals tell the
        // shaders so.
        let settings = self.settings.clone();
        let ssao = frame.ssao && settings.ssao;
        if !ssao {
            globals.ssao[3] = 0.0;
        }
        if settings.bloom_levels == 0 {
            globals.cc_tint[3] = 0.0;
        }
        if !settings.dof {
            globals.dof[2] = 0.0;
            globals.dof[3] = 0.0;
        }
        let shadows = frame.shadows && settings.sun_cascades > 0;
        if !shadows {
            globals.bloom[2] = 0.0;
        }
        self.post_bloom_size = globals.bloom[1];
        // NearIntensity below 0.22 takes the cheap full-resolution pass
        // (fs_dof_v), not the near chain (shader.wgsl DOF_CHEAP_BELOW).
        self.post_dof_near = globals.dof[2] >= DOF_CHEAP_BELOW;
        self.post_dof = globals.dof[2] > 0.0 || globals.dof[3] > 0.0;
        globals.cam_pos[3] = self.material_epoch.elapsed().as_secs_f32();
        // Deterministic isolated render probes (no effect in ordinary runs).
        if let Ok(t) = std::env::var("RUSTUDIO_MATERIAL_TIME") {
            if let Ok(t) = t.parse::<f32>() { globals.cam_pos[3] = t; }
        }
        if let Some(n) = crate::grass_temporal::index() {
            globals.cam_pos[3] = crate::grass_temporal::time(n);
            use std::io::Write;
            let dir=std::env::var("RUSTUDIO_GRASS_TEMPORAL").unwrap();
            let mut f=std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/frames.jsonl")).unwrap();
            writeln!(f,"{}",serde_json::json!({"frame":n,"shader_time":globals.cam_pos[3],"camera":globals.cam_pos,"spheres":globals.sun_sphere,"matrices":globals.sun_casc,"rects":globals.sun_rect,"viewport":size,"wall":self.material_epoch.elapsed().as_secs_f64()})).unwrap();
        }
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        self.vf.base = Some(globals);
        self.gui_fog = Self::gui_fog(&frame.globals);
        self.gui_lightgrid = frame.globals.lightgrid;
        self.gui_sun_params = [if frame.globals.sun_color[3] < 0.5 { 1.0 } else { 0.0 }, crate::mech::SUN_DIST, crate::mech::HYBRID_LERP_SLOPE, if shadows { 1.0 } else { 0.0 }];
        self.gui_sun = crate::world_gui::SunCascades { local: frame.globals.sun_casc, sphere: frame.globals.sun_sphere, rect: frame.globals.sun_rect };
        // (cascades past the settings' count are never drawn: a zero
        // matrix puts every point outside their tiles, to the voxel sun)
        for c in self.gui_sun.local.iter_mut().skip(settings.sun_cascades as usize) {
            *c = [[0.0; 4]; 4];
        }
        let mut stats = RenderStats {
            rebuilds: self.stats.rebuilds,
            ..Default::default()
        };

        self.requeue_quiet();
        // New parts join the static batches once they've settled.
        let settled = self
            .pending_at
            .is_none_or(|t| t.elapsed().as_secs_f32() >= crate::mech::STATIC_SETTLE_S);
        let compact = self.st.hidden > 1024 && self.st.hidden * 4 > self.st.raw.len();
        if (self.pending > 0
            && (self.st.batches.is_empty()
                || ((settled || self.pending > crate::mech::PENDING_MAX)
                    && self.last_rebuild.elapsed().as_secs_f32() >= crate::mech::STATIC_REBUILD_MIN_S)))
            || compact
        {
            self.last_rebuild = Instant::now();
            let _p = crate::bench::scope("render_static_rebuild");
            self.rebuild_static(device, queue);
            stats.rebuilds += 1;
        }
        self.flush_static_writes(queue);

        let cam = Vec3::new(
            frame.globals.cam_pos[0],
            frame.globals.cam_pos[1],
            frame.globals.cam_pos[2],
        );
        // (the perf log's sections of a 3D frame's CPU time: vpr_cull the
        // visibility and draw lists, vpr_shadows / vpr_scene / vpr_post
        // the passes' encoding, vpr_submit the driver taking the frame)
        let section = crate::bench::scope("vpr_cull");
        let planes = frustum(frame.cull_view_proj);
        // A part whose bounding radius covers less than `detail_px` pixels
        // is skipped (quality.rs: mech::LOD_MIN_PX at the top level), and
        // so is anything wholly beyond the view distance.
        let focal = size[1] as f32 * 0.5 / (frame.fov_y * 0.5).tan().max(1e-3);
        let lod_k = settings.detail_px / focal;
        let view_distance = settings.view_distance;

        // Static batches: frustum test per cell, then trim the smallest parts.
        let mut static_ranges: Vec<(Pass, (usize, usize), u32, u32)> = Vec::new();
        let mut see_bounds = SeeBounds::new(Mat4::from_cols_array_2d(&frame.globals.view_proj), size);
        for b in &self.st.batches {
            if !box_visible(&planes, b.lo, b.hi) {
                stats.culled_batches += 1;
                continue;
            }
            let near = box_distance(cam, b.lo, b.hi);
            if near > view_distance {
                stats.culled_batches += 1;
                continue;
            }
            let r_min = lod_k * near;
            let radii = &self.st.radii[b.start as usize..(b.start + b.count) as usize];
            let n = radii.partition_point(|r| *r >= r_min) as u32;
            stats.lod_skipped += (b.count - n) as usize;
            if n == 0 {
                continue;
            }
            if b.pass == Pass::Decal {
                see_bounds.add_box(b.lo, b.hi);
            }
            match static_ranges.last_mut() {
                Some(last) if last.0 == b.pass && last.1 == b.key && last.3 == b.start => {
                    last.3 = b.start + n
                }
                _ => static_ranges.push((b.pass, b.key, b.start, b.start + n)),
            }
        }

        // Per-frame items: moving, new and see-through parts.
        let mut dyn_opaque: Vec<((usize, usize), InstanceRaw)> = Vec::new();
        let mut dyn_decal: Vec<((usize, usize), InstanceRaw)> = Vec::new();
        let mut blended: Vec<(f32, (usize, usize), InstanceRaw)> = Vec::new();
        // Decals / Textures on a solid (or invisible) host: part of the
        // surface, so drawn with the opaque scene, before terrain water and
        // glass read it. Studio's water covers a hull's Textures below the
        // waterline (Pirate Island's Cutter); after the water they showed
        // through it. Decals on see-through hosts stay sorted with them.
        let mut surface_decals: Vec<(f32, (usize, usize), InstanceRaw)> = Vec::new();
        let mut water: Vec<((usize, usize), InstanceRaw)> = Vec::new();
        for r in &self.per_frame {
            let Some(item) = self.items.get(r) else {
                continue;
            };
            let solid_host = item.draws.first().is_none_or(|h| {
                h.tex_mode != TexMode::Decal
                    && (!shows(h) || (h.alpha >= 0.999 && h.material_id != 1584))
            });
            for d in &item.draws {
                let rad = radius_of(d);
                if !shows(d) || !sphere_visible(&planes, d.xf.pos, rad) {
                    continue;
                }
                let dist = (d.xf.pos - cam).length();
                if rad < lod_k * (dist - rad).max(0.0) || dist - rad > view_distance {
                    stats.lod_skipped += 1;
                    continue;
                }
                let raw = to_raw(d, item.bias);
                // (a skinned mesh's vertices can leave its box)
                let see_bound = |b: &mut SeeBounds| {
                    if self.meshes.get(mesh_slot(d)).is_some_and(|m| m.storage) {
                        b.set_full();
                    }
                    b.add_sphere(d.xf.pos, rad);
                };
                match pass_of(d) {
                    Some(Pass::Opaque) => dyn_opaque.push((draw_key(d), raw)),
                    Some(Pass::Decal) => {
                        see_bound(&mut see_bounds);
                        dyn_decal.push((draw_key(d), raw))
                    }
                    // A see-through union is drawn two-sided (its winding
                    // isn't trusted): from inside its box every face Studio
                    // would draw faces away, so nothing shows (Place B's
                    // camera inside a Neon union). File meshes have known
                    // sides: Studio draws a translucent sheet's front from
                    // inside its box and culls its back (both sides when
                    // DoubleSided; checks/translucent-inside), as the GPU
                    // does here. (Not terrain water, material 5, nor
                    // decals: their boxes span the map / the host.)
                    None if d.custom_mesh.is_some() && !self.culling_known(mesh_slot(d)) && d.material != 5.0 && d.tex_mode != TexMode::Decal && camera_inside(d, cam) => {}
                    // Terrain water: its own pass (water.rs).
                    None if d.custom_mesh.is_some() && d.material == 5.0 => water.push((draw_key(d), raw)),
                    None if solid_host && d.tex_mode == TexMode::Decal && d.texture.is_some() => {
                        see_bound(&mut see_bounds);
                        surface_decals.push((dist, draw_key(d), raw))
                    }
                    None => {
                        see_bound(&mut see_bounds);
                        blended.push((dist, draw_key(d), raw))
                    }
                }
            }
        }
        dyn_opaque.sort_by_key(|e| e.0);
        dyn_decal.sort_by_key(|e| e.0);
        // Glass samples the opaque scene and must precede other transparents.
        // Preserve the existing back-to-front distance order within each group.
        blended.sort_by(|a, b| transparent_order(a.2.extra[0], a.0, b.2.extra[0], b.0));
        // Back to front; equal distances keep source order (stable sort).
        surface_decals.sort_by(|a, b| b.0.total_cmp(&a.0));
        for d in frame.overlay.iter().filter(|d| shows(d)) {
            if self.meshes.get(mesh_slot(d)).is_some_and(|m| m.storage) {
                see_bounds.set_full();
            }
            see_bounds.add_sphere(d.xf.pos, radius_of(d));
        }
        let see_rects = see_bounds.rects();
        let overlay: Vec<((usize, usize), InstanceRaw)> = frame
            .overlay
            .iter()
            .filter(|d| shows(d))
            .map(|d| (draw_key(d), to_raw(&PartDraw { hover: true, ..*d }, 0.0)))
            .collect();
        let raw: Vec<InstanceRaw> = dyn_opaque
            .iter()
            .map(|e| e.1)
            .chain(dyn_decal.iter().map(|e| e.1))
            .chain(surface_decals.iter().map(|e| e.2))
            .chain(blended.iter().map(|e| e.2))
            .chain(overlay.iter().map(|e| e.1))
            .chain(water.iter().map(|e| e.1))
            .collect();
        if raw.len() > self.instance_cap {
            self.instance_cap = raw.len().next_power_of_two();
            self.instance_buf = Self::make_buf(
                device,
                "instances",
                self.instance_cap * std::mem::size_of::<InstanceRaw>(),
            );
        }
        if !raw.is_empty() {
            queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&raw));
        }
        // Runs of equal keys in a per-frame list → (key, start, end) in `raw`.
        let runs = |keys: &mut dyn Iterator<Item = (usize, usize)>,
                    base: u32|
         -> Vec<((usize, usize), u32, u32)> {
            let mut out: Vec<((usize, usize), u32, u32)> = Vec::new();
            for (i, k) in keys.enumerate() {
                let at = base + i as u32;
                match out.last_mut() {
                    Some(l) if l.0 == k && l.2 == at => l.2 = at + 1,
                    _ => out.push((k, at, at + 1)),
                }
            }
            out
        };
        let mut base = 0u32;
        let opaque_runs = runs(&mut dyn_opaque.iter().map(|e| e.0), base);
        base += dyn_opaque.len() as u32;
        let decal_runs = runs(&mut dyn_decal.iter().map(|e| e.0), base);
        base += dyn_decal.len() as u32;
        let surface_decal_runs = runs(&mut surface_decals.iter().map(|e| e.1), base);
        base += surface_decals.len() as u32;
        let glass_count = blended.iter().take_while(|e| (e.2.extra[0] as u32 & 7) == 3).count();
        let glass_end = base + glass_count as u32;
        let mut blend_runs = runs(&mut blended[..glass_count].iter().map(|e| e.1), base);
        blend_runs.extend(runs(&mut blended[glass_count..].iter().map(|e| e.1), glass_end));
        base += blended.len() as u32;
        let overlay_runs = runs(&mut overlay.iter().map(|e| e.0), base);
        base += overlay.len() as u32;
        let water_runs = runs(&mut water.iter().map(|e| e.0), base);
        stats.items = self.items.len();
        stats.static_instances = self.st.raw.len() - self.st.hidden;
        stats.batches = self.st.batches.len();
        stats.per_frame_items = self.per_frame.len();
        stats.drawn_instances = static_ranges
            .iter()
            .map(|r| (r.3 - r.2) as usize)
            .sum::<usize>()
            + raw.len();
        if frame.lines.len() > self.line_cap {
            self.line_cap = frame.lines.len().next_power_of_two();
            self.line_buf = Self::make_buf(
                device,
                "lines",
                self.line_cap * std::mem::size_of::<LineVertex>(),
            );
        }
        // The lines drawn on top of everything (vertex alpha ≥ 4, see
        // fs_line) go last, for their own pipeline (LinesOnTop): the
        // first `line_split` vertices are depth tested, the rest not.
        let line_count = (frame.lines.len() / 2 * 2) as u32;
        let mut line_split = line_count;
        if line_count > 0 {
            let on_top = |pair: &[LineVertex]| pair[0].color[3] >= 3.5;
            if frame.lines.chunks_exact(2).any(on_top) {
                let (tested, top): (Vec<&[LineVertex]>, Vec<&[LineVertex]>) =
                    frame.lines.chunks_exact(2).partition(|pair| !on_top(pair));
                line_split = tested.len() as u32 * 2;
                let sorted: Vec<LineVertex> = tested.into_iter().chain(top).flatten().copied().collect();
                queue.write_buffer(&self.line_buf, 0, bytemuck::cast_slice(&sorted));
            } else {
                queue.write_buffer(&self.line_buf, 0, bytemuck::cast_slice(&frame.lines[..line_count as usize]));
            }
        }

        let n_lights = frame.lights.len().min(crate::mech::MAX_LIGHTS);
        self.gui_n_lights = n_lights as u32;
        // (without local light shadows in the settings no light keeps the
        // shadow slot lighting.rs gave it: the atlas is never drawn)
        let unshadowed: Vec<LightRaw>;
        let lights: &[LightRaw] = if settings.light_shadows {
            &frame.lights[..n_lights]
        } else {
            unshadowed = frame.lights[..n_lights]
                .iter()
                .map(|l| {
                    let mut l = *l;
                    l.extra[3] = -1.0;
                    l
                })
                .collect();
            &unshadowed
        };
        if n_lights > 0 {
            queue.write_buffer(&self.lights_buf, 0, bytemuck::cast_slice(lights));
        }
        drop(section);
        let section = crate::bench::scope("vpr_shadows");

        // The sun's shadow cascades (sun_shadow.rs): which to redraw.
        let diagnostic_back = std::env::var_os("RUSTUDIO_CULL_BACK").is_some() || std::env::var_os("RUSTUDIO_CULL_SWITCH").is_some_and(|p| std::path::Path::new(&p).exists());
        let scene_gen = self.static_gen.wrapping_add(if diagnostic_back { 1u64 << 63 } else { 0 });
        static GRASS_SHADOWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let grass_shadows = *GRASS_SHADOWS.get_or_init(|| std::env::var_os("RUSTUDIO_NO_GRASS_SHADOW").is_none());
        let mut sun_due = if shadows {
            self.sun.upload(queue, &frame.sun_cascades);
            self.sun.plan(&frame.sun_cascades, scene_gen, !opaque_runs.is_empty() || !self.terrain_grass.is_empty())
        } else {
            self.sun.forget();
            [false; crate::sun_shadow::CASCADES]
        };
        // Only the settings' first cascades are drawn (the shader hands the
        // rest to the voxel sun: SUN_CASCADES).
        for due in sun_due.iter_mut().skip(settings.sun_cascades as usize) {
            *due = false;
        }

        let target = self.target.as_ref()?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewport"),
        });
        crate::gpu_timing::begin(device, queue, &mut encoder);
        if let Some(t) = &self.gpu_timer {
            t.mark(&mut encoder, 0);
        }
        // Skinned parts' vertices for this frame, before anything draws them.
        self.skin.encode(&mut encoder);
        crate::gpu_timing::mark(&mut encoder, "gpu: skinning");
        match frame.clouds.as_ref().filter(|_| frame.sky && settings.clouds) {
            Some(look) => {
                let (scene_format, samples) = (self.scene_format, self.samples);
                let clouds = self.clouds.get_or_insert_with(|| {
                    crate::clouds::CloudsGpu::new(device, queue, &self.globals_layout, &shader_source(&settings), scene_format, samples)
                });
                clouds.encode(device, queue, &mut encoder, &self.globals_bg, look, -frame.globals.sun_dir[1], target.size);
            }
            None => {
                if let Some(c) = self.clouds.as_mut() {
                    c.idle();
                }
            }
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: clouds");
        for (ci, cascade) in frame.sun_cascades.iter().enumerate() {
            if !sun_due[ci] {
                continue;
            }
            // Casters: static opaque cells that can reach the cascade's box
            // (trimmed like the view), plus this frame's moving opaque parts.
            let sp = frustum(cascade.caster_view_proj());
            let mut shadow_ranges: Vec<((usize, usize), u32, u32)> = Vec::new();
            for b in self.st.batches.iter().filter(|b| b.pass == Pass::Opaque) {
                if !box_visible(&sp, b.lo, b.hi) {
                    continue;
                }
                let r_min = lod_k * box_distance(cam, b.lo, b.hi) * crate::mech::SHADOW_LOD_SCALE;
                let radii = &self.st.radii[b.start as usize..(b.start + b.count) as usize];
                let n = radii.partition_point(|r| *r >= r_min) as u32;
                if n == 0 {
                    continue;
                }
                match shadow_ranges.last_mut() {
                    Some(last) if last.0 == b.key && last.2 == b.start => last.2 = b.start + n,
                    _ => shadow_ranges.push((b.key, b.start, b.start + n)),
                }
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sun shadow cascade"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let side = cascade.tile.min(crate::sun_shadow::depth_texels());
                pass.set_viewport(0.0, 0.0, side as f32, side as f32, 0.0, 1.0);
                pass.set_scissor_rect(0, 0, side, side);
                pass.set_pipeline(self.pipe(Pipe::Shadow));
                pass.set_bind_group(0, &self.shadow_globals_bg, &[]);
                self.sun.bind(&mut pass, ci);
                let cast =
                    |pass: &mut wgpu::RenderPass, key: (usize, usize), range: std::ops::Range<u32>| {
                        if let Some(mesh) = self.meshes.get(key.0) {
                            crate::runtime::domains::services::perf::draw(crate::runtime::domains::services::perf::SHADOW, 1, (mesh.count / 3) as u64 * (range.end - range.start) as u64);
                            pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                            pass.draw(0..mesh.count, range);
                        }
                    };
                pass.set_vertex_buffer(1, self.static_buf.slice(..));
                for r in &shadow_ranges {
                    cast(&mut pass, r.0, r.1..r.2);
                }
                pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                for r in &opaque_runs {
                    cast(&mut pass, r.0, r.1..r.2);
                }
                if !self.terrain_chunks.is_empty() || !self.terrain_grass.is_empty() {
                    // Grass remains two-sided, including in the opt-in solid
                    // terrain BACK-culling attribution capture.
                    pass.set_pipeline(self.pipe(Pipe::TerrainShadow));
                    if let Some(dir) = std::env::var_os("RUSTUDIO_CULL_EXPORT") {
                        let dir = std::path::PathBuf::from(dir);
                        std::fs::create_dir_all(&dir).unwrap();
                        let vp = cascade.view_proj();
                        let mut clip = Vec::<[f32;4]>::new();
                        let mut chunks = Vec::new();
                        for (key,c) in &self.terrain_chunks {
                            if box_visible(&sp,c.lo,c.hi) {
                                chunks.push((*key,c.count));
                                clip.extend(c.diagnostic_positions.iter().map(|p| (vp * Vec3::from(*p).extend(1.0)).to_array()));
                            }
                        }
                        std::fs::write(dir.join(format!("cascade{ci}-clip.bin")),bytemuck::cast_slice(&clip)).unwrap();
                        std::fs::write(dir.join(format!("cascade{ci}.json")),serde_json::to_vec_pretty(&serde_json::json!({"vp":vp.to_cols_array(),"tile":cascade.tile,"chunks":chunks,"vertices":clip.len()})).unwrap()).unwrap();
                    }
                    for (solid,c) in self.terrain_chunks.values().map(|c|(true,c)).chain(self.terrain_grass.values().filter(|_| grass_shadows).map(|c|(false,c))) {
                        pass.set_pipeline(self.pipe(if solid && diagnostic_back { Pipe::TerrainShadowBack } else { Pipe::TerrainShadow }));
                        if box_visible(&sp, c.lo, c.hi) {
                            crate::runtime::domains::services::perf::draw(crate::runtime::domains::services::perf::SHADOW, 1, (c.count / 3) as u64);
                            pass.set_vertex_buffer(0, c.vbuf.slice(..));
                            if c.grass_ranges.is_empty() { pass.draw(0..c.count, 0..1); }
                        else { for range in &c.grass_ranges {
                            let camera = Vec3::new(frame.globals.cam_pos[0], frame.globals.cam_pos[1], frame.globals.cam_pos[2]);
                            pass.draw(range.start..range.start + range.count(camera), 0..1);
                        }}
                        }
                    }
                }
            }
            self.sun.blur(&mut encoder, cascade, ci);
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: sun shadows");
        // Local light shadow maps (Studio's Future path): the lights the
        // light list gave a slot (LightRaw.extra.w), each a cube of EVSM
        // moments; redrawn only when those lights or the static scene change.
        {
            let slots: Vec<(usize, Vec3, f32)> = frame.lights[..n_lights]
                .iter()
                .filter(|l| l.extra[3] >= 0.0)
                .map(|l| (l.extra[3] as usize, Vec3::new(l.pos_range[0], l.pos_range[1], l.pos_range[2]), l.pos_range[3]))
                .filter(|s| s.0 < crate::mech::LSHADOW_MAX_LIGHTS)
                .collect();
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                self.static_gen.hash(&mut h);
                for (i, p, r) in &slots {
                    i.hash(&mut h);
                    p.to_array().map(f32::to_bits).hash(&mut h);
                    r.to_bits().hash(&mut h);
                }
                h.finish()
            };
            if settings.light_shadows && !slots.is_empty() && self.lshadow.key != Some(key) {
                self.lshadow.key = Some(key);
                let n = crate::mech::LSHADOW_MAX_LIGHTS;
                let mut slot_data = vec![0f32; n * 100];
                let mut face_data = vec![0u8; n * 6 * LSHADOW_FACE_STRIDE as usize];
                for (slot, pos, range) in &slots {
                    let inv = 1.0 / range.max(1e-3);
                    for (f, (vp, axis)) in LightShadows::faces(*pos, *range).iter().enumerate() {
                        slot_data[slot * 100 + f * 16..slot * 100 + f * 16 + 16].copy_from_slice(&vp.to_cols_array());
                        let mut u = [0f32; 24];
                        u[..16].copy_from_slice(&vp.to_cols_array());
                        u[16..20].copy_from_slice(&[pos.x, pos.y, pos.z, inv]);
                        u[20..24].copy_from_slice(&[axis.x, axis.y, axis.z, 0.0]);
                        let o = (slot * 6 + f) * LSHADOW_FACE_STRIDE as usize;
                        face_data[o..o + 96].copy_from_slice(bytemuck::cast_slice(&u));
                    }
                    slot_data[slot * 100 + 96..slot * 100 + 100].copy_from_slice(&[pos.x, pos.y, pos.z, inv]);
                }
                queue.write_buffer(&self.lshadow.slots_buf, 0, bytemuck::cast_slice(&slot_data));
                queue.write_buffer(&self.lshadow.face_buf, 0, &face_data);
                let far_p = crate::mech::EVSM_POS_EXP.exp() as f64;
                let far_n = -(-crate::mech::EVSM_NEG_EXP).exp() as f64;
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("light shadows"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.lshadow.moments_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: far_p, g: far_p * far_p, b: far_n, a: far_n * far_n }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.lshadow.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(self.pipe(Pipe::LightShadow));
                pass.set_bind_group(0, &self.shadow_globals_bg, &[]);
                pass.set_vertex_buffer(1, self.static_buf.slice(..));
                let cols = crate::mech::LSHADOW_ATLAS / crate::mech::LSHADOW_FACE;
                let face_px = crate::mech::LSHADOW_FACE;
                for (slot, pos, range) in &slots {
                    // The static cells the light reaches.
                    let near: Vec<&Batch> = self
                        .st
                        .batches
                        .iter()
                        .filter(|b| b.pass == Pass::Opaque)
                        .filter(|b| (pos.clamp(b.lo, b.hi) - *pos).length_squared() <= range * range)
                        .collect();
                    for f in 0..6u32 {
                        let k = *slot as u32 * 6 + f;
                        let (x, y) = ((k % cols) * face_px, (k / cols) * face_px);
                        pass.set_viewport(x as f32, y as f32, face_px as f32, face_px as f32, 0.0, 1.0);
                        pass.set_scissor_rect(x, y, face_px, face_px);
                        pass.set_bind_group(1, &self.lshadow.face_bg, &[k * LSHADOW_FACE_STRIDE as u32]);
                        for b in &near {
                            if let Some(mesh) = self.meshes.get(b.key.0) {
                                crate::runtime::domains::services::perf::draw(crate::runtime::domains::services::perf::SHADOW, 1, (mesh.count / 3) as u64 * b.count as u64);
                                pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                                pass.draw(0..mesh.count, b.start..b.start + b.count);
                            }
                        }
                    }
                }
            }
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: light shadows");
        if let Some(t) = &self.gpu_timer {
            t.mark(&mut encoder, 1);
        }
        drop(section);
        let section = crate::bench::scope("vpr_scene");
        // With terrain water in view the scene is drawn in three passes:
        // the opaque scene; then (after copying its colour) the water,
        // which looks through it; then decals and see-through parts.
        let has_glass = blended.iter().any(|e| (e.2.extra[0] as u32 & 7) == 3);
        // See-through Neon blends against the opaque snapshot too
        // (shader.wgsl neon_over_scene).
        let has_neon = blended.iter().any(|e| (e.2.extra[0] as u32 & 7) == 1);
        let split = !water_runs.is_empty() || has_glass || has_neon;
        self.water_drawn = !water_runs.is_empty();
        let mut calls = 0usize;
        // One instanced draw per run of (mesh, texture).
        let draw = |pass: &mut wgpu::RenderPass,
                    key: (usize, usize),
                    range: std::ops::Range<u32>,
                    calls: &mut usize| {
            let Some(mesh) = self.meshes.get(key.0) else {
                return;
            };
            pass.set_bind_group(1, self.textures.get(key.1).unwrap_or(&self.textures[0]), &[]);
            pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
            crate::runtime::domains::services::perf::draw(crate::runtime::domains::services::perf::SCENE, 1, (mesh.count / 3) as u64 * (range.end - range.start) as u64);
            pass.draw(0..mesh.count, range);
            *calls += 1;
        };
        // Scene environment probe work (env_probe.rs): Studio draws one cube
        // face a frame at the probe box's centre (Studio; 90 degrees,
        // far 3000), the opaque scene without the sky, then builds the
        // pyramid and convolves mips 1 and 2.
        if let Some(pf) = frame.probe.as_ref() {
            for task in &pf.tasks {
                if let crate::env_probe::Task::Face(f) = *task {
                    let (fwd, up) = crate::env_probe::face_basis(f);
                    let centre = pf.centre;
                    let fov = std::f32::consts::FRAC_PI_2;
                    let reverse = Mat4::perspective_rh(fov, 1.0, crate::env_probe::FAR, crate::mech::NEAR);
                    let ordinary = Mat4::perspective_rh(fov, 1.0, crate::mech::NEAR, crate::env_probe::FAR);
                    let view = Mat4::look_to_rh(centre, fwd, up);
                    let view_rel = Mat4::look_to_rh(Vec3::ZERO, fwd, up);
                    let mut pg = globals;
                    pg.view_proj = (reverse * view).to_cols_array_2d();
                    pg.inv_view_proj = (ordinary.as_dmat4() * view_rel.as_dmat4()).inverse().as_mat4().to_cols_array_2d();
                    pg.cam_pos = [centre.x, centre.y, centre.z, globals.cam_pos[3]];
                    pg.env_spec[3] = 0.0;
                    queue.write_buffer(&self.probe.globals_buf, 0, bytemuck::bytes_of(&pg));
                    let face_planes = frustum(ordinary * view);
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("env probe face"),
                            color_attachments: &[Some(match self.probe.face_msaa.as_ref() {
                                Some(msaa) => wgpu::RenderPassColorAttachment {
                                    view: msaa,
                                    depth_slice: None,
                                    resolve_target: Some(self.probe.face_resolve()),
                                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Discard },
                                },
                                None => wgpu::RenderPassColorAttachment {
                                    view: self.probe.face_resolve(),
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                                },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &self.probe.face_depth,
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_bind_group(0, &self.probe_globals_bg, &[]);
                        pass.set_bind_group(2, &target.glass_bg, &[]);
                        pass.set_vertex_buffer(1, self.static_buf.slice(..));
                        for b in self.st.batches.iter().filter(|b| b.pass == Pass::Opaque) {
                            if !box_visible(&face_planes, b.lo, b.hi) {
                                continue;
                            }
                            pass.set_pipeline(self.pipe(if self.culls_back(b.key.0) { Pipe::Opaque } else { Pipe::OpaqueTwoSided }));
                            draw(&mut pass, b.key, b.start..b.start + b.count, &mut calls);
                        }
                        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                        for r in &opaque_runs {
                            pass.set_pipeline(self.pipe(if self.culls_back(r.0.0) { Pipe::Opaque } else { Pipe::OpaqueTwoSided }));
                            draw(&mut pass, r.0, r.1..r.2, &mut calls);
                        }
                        if !self.terrain_chunks.is_empty() {
                            pass.set_pipeline(self.pipe(Pipe::Terrain));
                            pass.set_bind_group(1, &self.terrain_bg, &[]);
                            for c in self.terrain_chunks.values() {
                                if box_visible(&face_planes, c.lo, c.hi) {
                                    pass.set_vertex_buffer(0, c.vbuf.slice(..));
                                    pass.draw(0..c.count, 0..1);
                                }
                            }
                        }
                    }
                    self.probe.store_face(device, &mut encoder, f);
                } else {
                    self.probe.encode(device, &mut encoder, *task);
                }
            }
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: env probe");
        if let Some(t) = &self.gpu_timer {
            t.mark(&mut encoder, 2);
        }
        // Decals, then see-through parts back to front, then the hover overlay.
        // Studio's encoded blend (SeeThrough), when anything blends this frame:
        // the five see-through pipelines' encoded twins draw it.
        let encoded = !self.see.linear
            && (!surface_decal_runs.is_empty()
                || !decal_runs.is_empty()
                || !blend_runs.is_empty()
                || !overlay_runs.is_empty()
                || static_ranges.iter().any(|r| r.0 == Pass::Decal));
        let see_pipe = |p: Pipe| if encoded { p.encoded() } else { p };
        let surface_decals = |pass: &mut wgpu::RenderPass, calls: &mut usize| {
            if !surface_decal_runs.is_empty() {
                pass.set_pipeline(self.pipe(see_pipe(Pipe::Transparent)));
                pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                for r in &surface_decal_runs {
                    draw(pass, r.0, r.1..r.2, calls);
                }
                pass.set_bind_group(1, &self.textures[0], &[]);
            }
        };
        let see_through = |pass: &mut wgpu::RenderPass, calls: &mut usize| {
            pass.set_bind_group(2, &target.glass_bg, &[]);
            pass.set_pipeline(self.pipe(see_pipe(Pipe::Transparent)));
            pass.set_vertex_buffer(1, self.static_buf.slice(..));
            for r in static_ranges.iter().filter(|r| r.0 == Pass::Decal) {
                draw(pass, r.1, r.2..r.3, calls);
            }
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));
            for r in &decal_runs {
                draw(pass, r.0, r.1..r.2, calls);
            }
            for r in &blend_runs {
                pass.set_pipeline(self.pipe(see_pipe(if r.1 < glass_end {
                    if self.culls_back(r.0 .0) { Pipe::GlassCulled } else { Pipe::Glass }
                } else if self.culls_back(r.0 .0) {
                    Pipe::TransparentCulled
                } else {
                    Pipe::Transparent
                })));
                draw(pass, r.0, r.1..r.2, calls);
            }
            if !overlay_runs.is_empty() {
                pass.set_pipeline(self.pipe(see_pipe(Pipe::Overlay)));
                for r in &overlay_runs {
                    draw(pass, r.0, r.1..r.2, calls);
                }
            }
        };
        // The scene's colour attachment: the multisampled texture resolved
        // into the scene, or the scene itself without MSAA.
        let scene_attachment = |first: bool, last: bool| match target.msaa.as_ref() {
            Some(msaa) => wgpu::RenderPassColorAttachment {
                view: msaa,
                depth_slice: None,
                resolve_target: Some(&target.hdr_view),
                ops: wgpu::Operations {
                    load: if first { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load },
                    store: if last { wgpu::StoreOp::Discard } else { wgpu::StoreOp::Store },
                },
            },
            None => wgpu::RenderPassColorAttachment {
                view: &target.hdr_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if first { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load },
                    store: wgpu::StoreOp::Store,
                },
            },
        };
        let scene_pass = |encoder: &mut wgpu::CommandEncoder, label: &str, first: bool, last: bool| {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(scene_attachment(first, last))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: if first { wgpu::LoadOp::Clear(0.0) } else { wgpu::LoadOp::Load },
                        // Kept: the world GUI pass tests against it (round 23).
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime()
        };
        // The see-through encoding: the MSAA scene copied in (per sample),
        // then a pass drawing into it with the scene's depth.
        let see_begin = |encoder: &mut wgpu::CommandEncoder| {
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("see-through encode"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target.see,
                        depth_slice: None,
                        resolve_target: None,
                        // (not cleared: only the tiles written below are
                        // drawn into and decoded, SeeBounds)
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(self.pipe(Pipe::SeeEncode));
                pass.set_bind_group(0, &self.globals_bg, &[]);
                pass.set_bind_group(1, &target.see_encode_bg, &[]);
                for &[x, y, w, h] in &see_rects {
                    pass.set_scissor_rect(x, y, w, h);
                    pass.draw(0..3, 0..1);
                }
            }
            crate::gpu_timing::mark(encoder, "gpu: see-through encode (bounded)");
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("see-through"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target.see,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &target.depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_bind_group(2, &target.glass_bg, &[]);
            pass
        };
        // ... and the covered pixels decoded back into the scene (resolved);
        // the pass stays open for the editor's lines.
        let see_end = |encoder: &mut wgpu::CommandEncoder, last: bool| {
            let mut pass = scene_pass(encoder, "see-through decode", false, last);
            pass.set_pipeline(self.pipe(Pipe::SeeDecode));
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_bind_group(1, &target.see_decode_bg, &[]);
            for &[x, y, w, h] in &see_rects {
                pass.set_scissor_rect(x, y, w, h);
                pass.draw(0..3, 0..1);
            }
            // (the editor's lines follow over the whole view)
            pass.set_scissor_rect(0, 0, target.size[0], target.size[1]);
            pass
        };
        // The GPU bisection's steps (`bisect`): what this frame leaves out.
        let no_lines = matches!(bisect, 1 | 4 | 7);
        let no_sky = matches!(bisect, 2 | 4 | 7);
        let no_parts = matches!(bisect, 3 | 4 | 5 | 7);
        let no_scene_pass = matches!(bisect, 5 | 7);
        let no_post = matches!(bisect, 6 | 7);
        // Lines: the depth-tested ones, then the ones on top of everything
        // (their own pipeline, instead of a fragment shader writing its
        // depth, which costs a tiling GPU the early depth test for every
        // draw after it).
        let lines = |pass: &mut wgpu::RenderPass| {
            if line_count > 0 && !no_lines {
                pass.set_bind_group(0, &self.globals_bg, &[]);
                pass.set_bind_group(1, &self.textures[0], &[]);
                pass.set_vertex_buffer(0, self.line_buf.slice(..));
                if line_split > 0 {
                    pass.set_pipeline(self.pipe(Pipe::Lines));
                    pass.draw(0..line_split, 0..1);
                }
                if line_split < line_count {
                    pass.set_pipeline(self.pipe(Pipe::LinesOnTop));
                    pass.draw(line_split..line_count, 0..1);
                }
            }
        };
        if !no_scene_pass {
            let mut pass = scene_pass(&mut encoder, "viewport", true, !split && !encoded);
            pass.set_bind_group(2, &target.glass_bg, &[]);
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_bind_group(1, &self.textures[0], &[]);
            if frame.sky && !no_sky {
                pass.set_pipeline(self.pipe(Pipe::Sky));
                pass.draw(0..3, 0..1);
            }

            // Loaded meshes are two-sided unless mesh_cull_slot culls them
            // (file meshes, unless a MeshPart is DoubleSided: as Studio).
            let opaque_pipe = |key: (usize, usize)| {
                self.pipe(if self.culls_back(key.0) { Pipe::Opaque } else { Pipe::OpaqueTwoSided })
            };

            pass.set_vertex_buffer(1, self.static_buf.slice(..));
            for r in static_ranges.iter().filter(|r| r.0 == Pass::Opaque && !no_parts) {
                pass.set_pipeline(opaque_pipe(r.1));
                draw(&mut pass, r.1, r.2..r.3, &mut calls);
            }
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));
            for r in opaque_runs.iter().filter(|_| !no_parts) {
                pass.set_pipeline(opaque_pipe(r.0));
                draw(&mut pass, r.0, r.1..r.2, &mut calls);
            }
            // Only opaque authored decal texels contribute scene depth. The
            // shader rejects other blended materials and fractional coverage;
            // the invisible host never enters this pass. Keep the ordinary
            // sorted color pass for layer order and semitransparent edges.
            pass.set_pipeline(self.pipe(Pipe::DecalDepth));
            for r in surface_decal_runs.iter().chain(&blend_runs).filter(|_| !no_parts) {
                draw(&mut pass, r.0, r.1..r.2, &mut calls);
            }
            // Terrain chunks in view.
            if !self.terrain_chunks.is_empty() && !no_parts {
                pass.set_pipeline(self.pipe(Pipe::Terrain));
                pass.set_bind_group(1, &self.terrain_bg, &[]);
                for c in self.terrain_chunks.values().chain(self.terrain_grass.values()) {
                    if box_visible(&planes, c.lo, c.hi) && box_distance(cam, c.lo, c.hi) <= view_distance {
                        crate::runtime::domains::services::perf::draw(crate::runtime::domains::services::perf::SCENE, 1, (c.count / 3) as u64);
                        pass.set_vertex_buffer(0, c.vbuf.slice(..));
                        if c.grass_ranges.is_empty() { pass.draw(0..c.count, 0..1); }
                        else { for range in &c.grass_ranges {
                            let camera = Vec3::new(frame.globals.cam_pos[0], frame.globals.cam_pos[1], frame.globals.cam_pos[2]);
                            pass.draw(range.start..range.start + range.count(camera), 0..1);
                        }}
                    }
                }
                pass.set_bind_group(1, &self.textures[0], &[]);
            }
            if !encoded {
                if !no_parts {
                    surface_decals(&mut pass, &mut calls);
                }
                lines(&mut pass);
                if !split && !no_parts {
                    see_through(&mut pass, &mut calls);
                }
            }
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: scene (opaque, terrain, unencoded blends)");
        if let Some(t) = &self.gpu_timer {
            t.mark(&mut encoder, 3);
        }
        // Decals on solid hosts belong to the surface: before terrain water
        // and glass read the scene.
        if encoded && split && !surface_decal_runs.is_empty() && !no_parts {
            let mut pass = see_begin(&mut encoder);
            surface_decals(&mut pass, &mut calls);
            drop(pass);
            crate::gpu_timing::mark(&mut encoder, "gpu: see-through draws");
            drop(see_end(&mut encoder, false));
            crate::gpu_timing::mark(&mut encoder, "gpu: see-through decode (bounded)");
        }
        if split && !no_parts {
            let u = self.water.uniforms(&frame.water, target.size);
            queue.write_buffer(&self.water.buf, 0, bytemuck::bytes_of(&u));
            encoder.copy_texture_to_texture(
                target.hdr.as_image_copy(),
                target.scene_copy.as_image_copy(),
                wgpu::Extent3d {
                    width: target.size[0],
                    height: target.size[1],
                    depth_or_array_layers: 1,
                },
            );
            if !water_runs.is_empty() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("water"),
                    color_attachments: &[Some(scene_attachment(false, false))],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &target.water_depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(self.pipe(Pipe::Water));
                pass.set_bind_group(0, &self.globals_bg, &[]);
                pass.set_bind_group(2, &target.water_bg, &[]);
                pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                for r in &water_runs {
                    draw(&mut pass, r.0, r.1..r.2, &mut calls);
                }
            }
            // A separate immutable snapshot prevents sampling an active depth attachment.
            if has_glass {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("glass depth snapshot"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &target.glass_depth,
                        depth_slice: None, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
                });
                pass.set_pipeline(self.pipe(Pipe::GlassResolve));
                pass.set_bind_group(0, &self.globals_bg, &[]);
                pass.set_bind_group(1, &target.depth_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            if !encoded {
                let mut pass = scene_pass(&mut encoder, "see-through", false, true);
                pass.set_bind_group(0, &self.globals_bg, &[]);
                see_through(&mut pass, &mut calls);
            }
        }
        if encoded {
            let mut pass = see_begin(&mut encoder);
            if !split {
                surface_decals(&mut pass, &mut calls);
            }
            see_through(&mut pass, &mut calls);
            drop(pass);
            crate::gpu_timing::mark(&mut encoder, "gpu: see-through draws");
            let mut pass = see_end(&mut encoder, true);
            lines(&mut pass);
            drop(pass);
            crate::gpu_timing::mark(&mut encoder, "gpu: see-through decode (bounded)");
        }
        // (only when something ran: an empty span would still read a few
        // nanoseconds of "see-through" in frames without any)
        if split || encoded {
            crate::gpu_timing::mark(&mut encoder, "gpu: water, glass, split see-through");
        }
        stats.draw_calls = calls;
        drop(section);
        let section = crate::bench::scope("vpr_post");
        // Scene colour snapshot (water/glass reads are complete) for SSAO's
        // encoded-scene reads (shader.wgsl ssao_scene); nothing after it
        // reads the copy otherwise (clouds take their own), so it is only
        // made when SSAO is on.
        if let Some(t) = &self.gpu_timer {
            t.mark(&mut encoder, 4);
        }
        if ssao && target.ssao.is_some() {
            encoder.copy_texture_to_texture(target.hdr.as_image_copy(), target.scene_copy.as_image_copy(),
                wgpu::Extent3d { width: target.size[0], height: target.size[1], depth_or_array_layers: 1 });
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: scene copy");
        // Bloom: bright pass, blur across, blur down (half resolution).
        // SSAO, then its multiply into the scene.
        // (Off: nothing reads its textures; the post passes take only the
        // scene's depth from that bind group.)
        if let Some(ao) = target.ssao.as_ref().filter(|_| ssao) {
            let steps: [(&str, Pipe, &wgpu::BindGroup, &wgpu::TextureView, bool); 5] = [
                ("ssao depth", Pipe::SsaoDown, &target.depth_bg, &ao.depth_view, true),
                ("ssao", Pipe::SsaoMain, &ao.bgs[0], &ao.views[0], true),
                ("ssao blur across", Pipe::SsaoBlurX, &ao.bgs[1], &ao.views[1], true),
                ("ssao blur down", Pipe::SsaoBlurY, &ao.bgs[2], &ao.views[0], true),
                ("ssao apply", Pipe::SsaoApply, &ao.bgs[1], &target.hdr_view, false),
            ];
            for (label, pipe, bg, view, clear) in steps {
                if label == "ssao apply" && (std::env::var_os("RUSTUDIO_TERRAIN_CAPTURE_NO_AO").is_some() || std::env::var_os("RUSTUDIO_TERRAIN_CAPTURE_ID").is_some()) { continue; }
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if clear { wgpu::LoadOp::Clear(wgpu::Color::WHITE) } else { wgpu::LoadOp::Load },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(self.pipe(pipe));
                pass.set_bind_group(0, &self.globals_bg, &[]);
                pass.set_bind_group(1, bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: ssao");
        if let Some(t) = &self.gpu_timer {
            t.mark(&mut encoder, 5);
        }
        // Terrain.Clouds over the sky (Studio composes them after SSAO).
        if let Some(c) = self.clouds.as_mut() {
            encoder.copy_texture_to_texture(target.hdr.as_image_copy(), target.scene_copy.as_image_copy(),
                wgpu::Extent3d { width: target.size[0], height: target.size[1], depth_or_array_layers: 1 });
            c.composite(device, &mut encoder, &self.globals_bg, &target.hdr_view, &target.scene_copy_view, &target.depth);
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: clouds composite");
        if !no_post {
            self.encode_bloom(&mut encoder, target, globals.bloom[1]);
            crate::gpu_timing::mark(&mut encoder, "gpu: bloom");
            self.encode_dof(&mut encoder, target, self.post_dof, self.post_dof_near);
            crate::gpu_timing::mark(&mut encoder, "gpu: dof");
        }
        if let Some(t) = &self.gpu_timer {
            t.mark(&mut encoder, 6);
        }
        if !no_post {
            self.encode_post(&mut encoder, target);
            crate::gpu_timing::mark(&mut encoder, "gpu: post");
        }
        crate::gpu_timing::mark(&mut encoder, "gpu: post");
        self.probe.flush(queue);
        if let Some(t) = self.gpu_timer.as_mut() {
            t.mark(&mut encoder, 7);
            t.resolve(&mut encoder);
        }
        drop(section);
        // The frame's commands: submitted now, or kept for egui's submit of
        // this same frame (one vkQueueSubmit a frame: `one_submit`), which
        // `submit_callback` puts them in front of.
        let buffer = encoder.finish();
        self.post_submit_pending = true;
        if self.defers(bisect) {
            *self.deferred.lock().unwrap_or_else(|e| e.into_inner()) = Some(buffer);
        } else {
            {
                let _submit = crate::bench::scope("vpr_submit");
                queue.submit([buffer]);
            }
            if bisect == 8 {
                // The CPU waits for the GPU to finish the frame: its wall time.
                let _sync = crate::bench::scope("vpr_gpu_sync");
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            }
            self.after_submit(device);
        }
        self.stats = stats;
        crate::runtime::domains::services::perf::end_frame(svc_t0.elapsed().as_secs_f64(), 0.0);
        self.texture_id
    }
}

const BUILTIN_MESHES: usize = 5;

impl Renderer {
    /// The mesh index to draw loaded mesh `index` single-sided (back faces
    /// culled) or two-sided with: `index` itself, or, when another part
    /// already draws that shared mesh the other way (a DoubleSided MeshPart
    /// beside single-sided ones), a second slot over the same vertices.
    /// Skinned slots are a part's own and are set in place.
    pub fn mesh_cull_slot(&mut self, index: u32, cull_back: bool) -> u32 {
        let Some(m) = self.meshes.get_mut(index as usize + BUILTIN_MESHES) else { return index };
        if m.storage || m.cull.is_none_or(|c| c == cull_back) {
            if m.cull != Some(cull_back) {
                // (what a frame draws changed: ViewReuse)
                self.generation += 1;
            }
            m.cull = Some(cull_back);
            return index;
        }
        if let Some(&a) = self.cull_alias.get(&(index, cull_back)) {
            return a;
        }
        let alias = Mesh { vbuf: m.vbuf.clone(), count: m.count, cull: Some(cull_back), storage: false };
        self.meshes.push(alias);
        let a = (self.meshes.len() - 1 - BUILTIN_MESHES) as u32;
        self.cull_alias.insert((index, cull_back), a);
        a
    }

    /// Whether the mesh at this slot is drawn with back faces culled:
    /// built-in shapes always, loaded meshes when mesh_cull_slot says so.
    fn culls_back(&self, slot: usize) -> bool {
        slot < BUILTIN_MESHES || self.meshes.get(slot).is_some_and(|m| m.cull == Some(true))
    }

    /// Whether the slot's sides are known (built-in shapes, file meshes)
    /// rather than drawn two-sided for an unknown winding (CSG).
    fn culling_known(&self, slot: usize) -> bool {
        slot < BUILTIN_MESHES || self.meshes.get(slot).is_some_and(|m| m.cull.is_some())
    }
}

/// Index into `Renderer::meshes`: built-in shapes first, then loaded meshes.
fn mesh_slot(p: &PartDraw) -> usize {
    match p.custom_mesh {
        Some(i) => BUILTIN_MESHES + i as usize,
        None => p.mesh as usize,
    }
}

impl Renderer {
    /// Uploads a triangle list (positions in the unit box, normals scaled to
    /// match) and returns its index for [`PartDraw::custom_mesh`].
    pub fn add_mesh(
        &mut self,
        device: &wgpu::Device,
        positions: &[Vec3],
        normals: &[Vec3],
        uvs: &[glam::Vec2],
    ) -> u32 {
        self.add_mesh_colored(device, positions, normals, uvs, &[])
    }

    /// `add_mesh` with per-vertex colours (sRGB 0..1, alpha 1), or none.
    pub fn add_mesh_colored(
        &mut self,
        device: &wgpu::Device,
        positions: &[Vec3],
        normals: &[Vec3],
        uvs: &[glam::Vec2],
        colors: &[glam::Vec4],
    ) -> u32 {
        self.add_mesh_projected(device, positions, normals, uvs, colors, &[])
    }

    pub fn add_mesh_projected(
        &mut self, device: &wgpu::Device, positions: &[Vec3], normals: &[Vec3],
        uvs: &[glam::Vec2], colors: &[glam::Vec4], planes: &[glam::Vec4],
    ) -> u32 {
        self.add_mesh_with(device, positions, normals, uvs, colors, planes, false)
    }

    /// `add_mesh_projected`; `storage` also lets the skinning pass write it.
    #[allow(clippy::too_many_arguments)]
    fn add_mesh_with(
        &mut self, device: &wgpu::Device, positions: &[Vec3], normals: &[Vec3],
        uvs: &[glam::Vec2], colors: &[glam::Vec4], planes: &[glam::Vec4], storage: bool,
    ) -> u32 {
        self.generation += 1;
        let verts: Vec<Vertex> = positions
            .iter()
            .zip(normals)
            .enumerate()
            .map(|(i, (p, n))| Vertex {
                pos: p.to_array(),
                normal: n.to_array(),
                projection: planes.get(i).map(|p| p.to_array()).unwrap_or([0.0; 4]),
                uv: uvs.get(i).map(|u| u.to_array()).unwrap_or([0.0, 0.0]),
                color: colors.get(i).map(|c| c.to_array()).unwrap_or([0.0; 4]),
            })
            .collect();
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("loaded mesh"),
            contents: bytemuck::cast_slice(&verts),
            usage: if storage {
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE
            } else {
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
            },
        });
        self.meshes.push(Mesh {
            vbuf,
            count: verts.len() as u32,
            cull: None,
            storage,
        });
        (self.meshes.len() - 1 - BUILTIN_MESHES) as u32
    }

    /// Uploads ready-made vertices (an EditableMesh's live copy,
    /// runtime/editable.rs) and returns the index for `custom_mesh`.
    pub fn add_mesh_vertices(&mut self, device: &wgpu::Device, verts: &[Vertex]) -> u32 {
        self.generation += 1;
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("live mesh"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        self.meshes.push(Mesh { vbuf, count: verts.len() as u32, cull: None, storage: false });
        (self.meshes.len() - 1 - BUILTIN_MESHES) as u32
    }

    /// Overwrites vertices `first..` of loaded mesh `index` in place (no new
    /// buffer, only these bytes go up). False when they don't fit.
    pub fn write_mesh_vertices(&mut self, queue: &wgpu::Queue, index: u32, first: u32, verts: &[Vertex]) -> bool {
        let Some(m) = self.meshes.get(index as usize + BUILTIN_MESHES) else { return false };
        if m.storage || first as usize + verts.len() > m.count as usize {
            return false;
        }
        if !verts.is_empty() {
            queue.write_buffer(&m.vbuf, first as u64 * std::mem::size_of::<Vertex>() as u64, bytemuck::cast_slice(verts));
            crate::bench::add("live_mesh_bytes", 0.0, (verts.len() * std::mem::size_of::<Vertex>()) as u64);
            self.generation += 1;
        }
        true
    }

    /// Rewrites loaded mesh `index` whole from these arrays (the same
    /// layout `add_mesh_projected` builds), in place. A skinning slot gets
    /// its rest pose here and is queued for the compute pass again with the
    /// palette it already holds. False when the sizes differ.
    #[allow(clippy::too_many_arguments)]
    pub fn rewrite_mesh(
        &mut self, queue: &wgpu::Queue, index: u32, positions: &[Vec3], normals: &[Vec3],
        uvs: &[glam::Vec2], colors: &[glam::Vec4], projection: [f32; 4],
    ) -> bool {
        let Some(m) = self.meshes.get(index as usize + BUILTIN_MESHES) else { return false };
        if m.count as usize != positions.len() || normals.len() != positions.len() {
            return false;
        }
        let projection = if m.storage { [0.0; 4] } else { projection };
        let verts: Vec<Vertex> = positions
            .iter()
            .zip(normals)
            .enumerate()
            .map(|(i, (p, n))| Vertex {
                pos: p.to_array(),
                normal: n.to_array(),
                projection,
                uv: uvs.get(i).map(|u| u.to_array()).unwrap_or([0.0, 0.0]),
                color: colors.get(i).map(|c| c.to_array()).unwrap_or([0.0; 4]),
            })
            .collect();
        queue.write_buffer(&m.vbuf, 0, bytemuck::cast_slice(&verts));
        crate::bench::add("live_mesh_bytes", 0.0, (verts.len() * std::mem::size_of::<Vertex>()) as u64);
        if m.storage {
            self.skin.again(index);
        }
        true
    }

    /// A skinned part's mesh slot deformed on the GPU (skin_gpu.rs): the
    /// slot `index` when it is one made for skinning with the same corner
    /// count, else a new one holding the mesh at rest (uvs, colours); then
    /// this frame's palette is queued for the compute pass. Returns the
    /// index to draw with.
    pub fn skin_mesh(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue, index: Option<u32>,
        req: &crate::skin_gpu::Request,
    ) -> u32 {
        let base = req.base;
        let ok = index
            .and_then(|i| self.meshes.get(i as usize + BUILTIN_MESHES))
            .is_some_and(|m| m.storage && m.count as usize == base.positions.len());
        let idx = match index {
            Some(i) if ok => i,
            _ => {
                if let Some(i) = index {
                    self.skin.drop_slot(i);
                }
                let i = self.add_mesh_with(device, &base.positions, &base.normals, &base.uvs, &base.colors, &[], true);
                let b = base.positions.len() as u64 * std::mem::size_of::<Vertex>() as u64;
                self.skin.bytes += b;
                crate::bench::add("skin_upload_bytes", 0.0, b);
                i
            }
        };
        let m = &self.meshes[idx as usize + BUILTIN_MESHES];
        self.skin.queue(device, queue, idx, &m.vbuf, req);
        // (a pose a frame: never a frame shown again, ViewReuse)
        self.generation += 1;
        idx
    }

    /// GPU bytes written for skinning since the last call (bench).
    pub fn take_skin_bytes(&mut self) -> u64 {
        self.skin.take_bytes()
    }

    /// Rewrites a loaded mesh's vertices in place (skinned parts re-upload
    /// their deformed copy, skin.rs), or adds a new mesh when `index` is
    /// None or its size differs. Returns the index to draw with.
    pub fn set_mesh(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue, index: Option<u32>,
        positions: &[Vec3], normals: &[Vec3], uvs: &[glam::Vec2], colors: &[glam::Vec4],
    ) -> u32 {
        self.generation += 1;
        let slot = index.map(|i| i as usize + BUILTIN_MESHES);
        match slot.and_then(|s| self.meshes.get(s)) {
            Some(m) if m.count as usize == positions.len() => {
                let verts: Vec<Vertex> = positions
                    .iter()
                    .zip(normals)
                    .enumerate()
                    .map(|(i, (p, n))| Vertex {
                        pos: p.to_array(),
                        normal: n.to_array(),
                        projection: [0.0; 4],
                        uv: uvs.get(i).map(|u| u.to_array()).unwrap_or([0.0, 0.0]),
                        color: colors.get(i).map(|c| c.to_array()).unwrap_or([0.0; 4]),
                    })
                    .collect();
                queue.write_buffer(&m.vbuf, 0, bytemuck::cast_slice(&verts));
                index.unwrap_or(0)
            }
            _ => self.add_mesh_colored(device, positions, normals, uvs, colors),
        }
    }
}

/// Sort / batch key: mesh slot, then texture.
fn draw_key(p: &PartDraw) -> (usize, usize) {
    (
        mesh_slot(p),
        if p.tex_mode == TexMode::None {
            0
        } else {
            p.texture.unwrap_or(0) as usize
        },
    )
}

fn tex_mode(p: &PartDraw) -> f32 {
    match (p.tex_mode, p.texture) {
        (_, None) | (TexMode::None, _) => 0.0,
        (TexMode::Over, Some(_)) => 1.0,
        (TexMode::Decal, Some(_)) => 2.0,
        (TexMode::Opaque, Some(_)) => 3.0,
        (TexMode::Cutout, Some(_)) => 4.0,
        (TexMode::OpaqueTinted, Some(_)) => 5.0,
        (TexMode::TintMask, Some(_)) => 6.0,
    }
}

impl Renderer {
    /// Adds or updates one part's draws. Parts that keep changing place move
    /// to the per-frame list; colour-only changes update the static slot.
    pub fn set_item(&mut self, r: Ref, mut draws: Vec<PartDraw>) {
        self.generation += 1;
        // Studio seeds material offsets from an allocation address. A DOM
        // referent supplies the equivalent stable per-instance identity here.
        let identity = u64::from_str_radix(&r.to_string()[16..], 16).unwrap_or(0);
        for d in &mut draws {
            if d.mat_layer >= 0 && d.custom_mesh.is_none() && d.tex_mode == TexMode::None {
                // Studio returns no offset for a MaterialVariant.
                let variant = self.materials.cpu.get(d.mat_layer as usize)
                    .is_some_and(|m| crate::matpack::is_variant_pack(m.pack));
                let off = if variant { [0.0; 2] } else { crate::matpack::uv_offset(d.size, d.material_id, identity) };
                d.uvt[2] = off[0];
                d.uvt[3] = off[1];
            }
        }
        if let Some(layer) = self.vf.layer_of.get(&r).copied() {
            // A ViewportFrame's part: drawn by its frame's pass only.
            let bias = bias_of(&draws);
            self.vf.touch(layer);
            self.vf.prepare(layer, r, &draws, bias);
            self.items.insert(r, Item { draws, bias, state: ItemState::Dynamic, moved: Instant::now() });
            return;
        }
        let Renderer {
            items,
            st,
            per_frame,
            pending,
            pending_at,
            ..
        } = self;
        let static_ok = draws.iter().all(|d| !shows(d) || pass_of(d).is_some());
        let Some(item) = items.get_mut(&r) else {
            let bias = bias_of(&draws);
            items.insert(
                r,
                Item {
                    draws,
                    bias,
                    state: ItemState::Pending,
                    moved: Instant::now(),
                },
            );
            per_frame.insert(r);
            *pending += 1;
            // Settle time counts from the first waiting part: a game that
            // spawns something every few frames would otherwise never settle.
            pending_at.get_or_insert_with(Instant::now);
            return;
        };
        let same_place = item.draws.len() == draws.len()
            && item.draws.iter().zip(&draws).all(|(a, b)| {
                near_xf(&a.xf, &b.xf)
                    && a.size == b.size
                    && pass_of(a) == pass_of(b)
                    && draw_key(a) == draw_key(b)
                    && shows(a) == shows(b)
            });
        match &item.state {
            ItemState::Static(slots) if same_place && static_ok => {
                for (d, &s) in draws.iter().zip(slots) {
                    if s != u32::MAX {
                        st.raw[s as usize] = to_raw(d, item.bias);
                        st.writes.push(s);
                    }
                }
            }
            ItemState::Static(slots) => {
                for &s in slots {
                    if s != u32::MAX {
                        st.raw[s as usize] = HIDDEN;
                        st.writes.push(s);
                        st.hidden += 1;
                    }
                }
                item.state = ItemState::Dynamic;
                per_frame.insert(r);
            }
            ItemState::Pending if !same_place => {
                item.state = ItemState::Dynamic;
                *pending = pending.saturating_sub(1);
            }
            _ => {}
        }
        if !same_place {
            item.moved = Instant::now();
        }
        item.draws = draws;
    }

    /// Dynamic parts that stopped changing go back to Pending, so the next
    /// static rebuild batches them again (a part moved once by a script, or
    /// whose mesh finished loading, is not per-frame for the session).
    fn requeue_quiet(&mut self) {
        if self.last_quiet_scan.elapsed().as_secs_f32() < 1.0 {
            return;
        }
        self.last_quiet_scan = Instant::now();
        let mut n = 0;
        for r in &self.per_frame {
            let Some(item) = self.items.get_mut(r) else { continue };
            if item.state == ItemState::Dynamic
                && item.moved.elapsed().as_secs_f32() >= crate::mech::DYNAMIC_SETTLE_S
                && item.draws.iter().all(|d| !shows(d) || pass_of(d).is_some())
            {
                item.state = ItemState::Pending;
                n += 1;
            }
        }
        if n > 0 {
            self.pending += n;
            self.pending_at.get_or_insert_with(Instant::now);
        }
    }

    pub fn remove_item(&mut self, r: Ref) {
        if self.vf.layer_of.contains_key(&r) {
            self.set_layer(r, None);
            return;
        }
        self.generation += 1;
        let Some(item) = self.items.remove(&r) else {
            return;
        };
        match item.state {
            ItemState::Static(slots) => {
                for s in slots {
                    if s != u32::MAX {
                        self.st.raw[s as usize] = HIDDEN;
                        self.st.writes.push(s);
                        self.st.hidden += 1;
                    }
                }
            }
            ItemState::Pending => self.pending = self.pending.saturating_sub(1),
            ItemState::Dynamic => {}
        }
        self.per_frame.remove(&r);
    }

    pub fn clear_items(&mut self) {
        self.generation += 1;
        let layered = &self.vf.layer_of;
        self.items.retain(|r, _| layered.contains_key(r));
        self.per_frame.clear();
        self.st = StaticLayout::default();
        self.static_gen += 1;
        self.pending = 0;
        self.pending_at = None;
    }

    pub fn item_draws(&self, r: Ref) -> Option<&[PartDraw]> {
        self.items.get(&r).map(|i| i.draws.as_slice())
    }

    /// Re-sorts every settled part into the static buffer.
    fn rebuild_static(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        struct Row {
            pass: Pass,
            key: (usize, usize),
            cell: u64,
            radius: f32,
            raw: InstanceRaw,
            lo: Vec3,
            hi: Vec3,
            owner: Ref,
            index: u32,
        }
        let mut rows: Vec<Row> = Vec::with_capacity(self.items.len());
        for (r, item) in self.items.iter_mut() {
            if item.state == ItemState::Dynamic || self.vf.layer_of.contains_key(r) {
                continue;
            }
            if !item.draws.iter().all(|d| !shows(d) || pass_of(d).is_some()) {
                item.state = ItemState::Dynamic;
                continue;
            }
            item.state = ItemState::Static(vec![u32::MAX; item.draws.len()]);
            for (i, d) in item.draws.iter().enumerate() {
                let Some(pass) = pass_of(d).filter(|_| shows(d)) else {
                    continue;
                };
                let c = (d.xf.pos / CELL).floor();
                let (lo, hi) = aabb_of(d);
                rows.push(Row {
                    pass,
                    key: draw_key(d),
                    cell: morton([c.x as i32, c.y as i32, c.z as i32]),
                    radius: radius_of(d),
                    raw: to_raw(d, item.bias),
                    lo,
                    hi,
                    owner: *r,
                    index: i as u32,
                });
            }
        }
        rows.sort_by(|a, b| {
            (a.pass, a.key, a.cell)
                .cmp(&(b.pass, b.key, b.cell))
                .then(b.radius.total_cmp(&a.radius))
        });
        let mut st = StaticLayout {
            raw: Vec::with_capacity(rows.len()),
            radii: Vec::with_capacity(rows.len()),
            ..Default::default()
        };
        let mut cell_of_batch = u64::MAX;
        for (slot, row) in rows.iter().enumerate() {
            let slot = slot as u32;
            st.raw.push(row.raw);
            st.radii.push(row.radius);
            if let Some(ItemState::Static(slots)) =
                self.items.get_mut(&row.owner).map(|i| &mut i.state)
            {
                slots[row.index as usize] = slot;
            }
            match st.batches.last_mut() {
                Some(b) if b.pass == row.pass && b.key == row.key && cell_of_batch == row.cell => {
                    b.count += 1;
                    b.lo = b.lo.min(row.lo);
                    b.hi = b.hi.max(row.hi);
                }
                _ => {
                    cell_of_batch = row.cell;
                    st.batches.push(Batch {
                        pass: row.pass,
                        key: row.key,
                        lo: row.lo,
                        hi: row.hi,
                        start: slot,
                        count: 1,
                    });
                }
            }
        }
        if st.raw.len() > self.static_cap {
            self.static_cap = st.raw.len().next_power_of_two();
            self.static_buf = Self::make_buf(
                device,
                "static instances",
                self.static_cap * std::mem::size_of::<InstanceRaw>(),
            );
        }
        if !st.raw.is_empty() {
            queue.write_buffer(&self.static_buf, 0, bytemuck::cast_slice(&st.raw));
        }
        self.st = st;
        self.static_gen += 1;
        let items = &self.items;
        self.per_frame
            .retain(|r| items.get(r).is_some_and(|i| i.state == ItemState::Dynamic));
        self.pending = 0;
        self.pending_at = None;
    }

    /// Uploads static slots changed in place (merged into contiguous runs).
    fn flush_static_writes(&mut self, queue: &wgpu::Queue) {
        if self.st.writes.is_empty() {
            return;
        }
        self.static_gen += 1;
        let mut w = std::mem::take(&mut self.st.writes);
        w.sort_unstable();
        w.dedup();
        let size = std::mem::size_of::<InstanceRaw>() as u64;
        let mut i = 0;
        while i < w.len() {
            let start = w[i];
            let mut end = start + 1;
            i += 1;
            while i < w.len() && w[i] == end {
                end += 1;
                i += 1;
            }
            if (end as usize) <= self.st.raw.len() {
                queue.write_buffer(
                    &self.static_buf,
                    start as u64 * size,
                    bytemuck::cast_slice(&self.st.raw[start as usize..end as usize]),
                );
            }
        }
    }
}

/// InstanceRaw.extra.y bit for CastShadow false (vs_shadow drops the part).
const NO_SHADOW_BIT: u32 = 1 << 19;

// selects PlasticReflection only for 256/272; tiled
// materials (including Metal) keep their texture-driven metalness/roughness.
pub(crate) fn material_reflectance(material: u32, reflectance: f32) -> f32 {
    match material {
        256 | 272 | 1568 => reflectance.clamp(0.0, 1.0),
        _ => 0.0,
    }
}

fn to_raw(p: &PartDraw, bias: f32) -> InstanceRaw {
    let r = p.xf.rot;
    let pos = p.xf.pos;
    InstanceRaw {
        r0: [r.x_axis.x, r.x_axis.y, r.x_axis.z, pos.x],
        r1: [r.y_axis.x, r.y_axis.y, r.y_axis.z, pos.y],
        r2: [r.z_axis.x, r.z_axis.y, r.z_axis.z, pos.z],
        size: [p.size.x, p.size.y, p.size.z, bias],
        color: [
            p.color[0], p.color[1], p.color[2],
            // scene::part_draw currently halves ForceField opacity.
            if p.material_id == 1584 { (p.alpha * 2.0).min(1.0) } else { p.alpha },
        ],
        // y: placeholder (bit 0), the six surfaces (3 bits each above it)
        // and NO_SHADOW_BIT (CastShadow false).
        // x: the shading kind (0..7), 8 = map materials by the mesh UVs,
        // 16 × (material layer + 1).
        extra: [
            (if p.material_id == 1584 { 6.0 } else { p.material })
                + if p.custom_mesh.is_some() { 8.0 } else { 0.0 }
                + 16.0 * (p.mat_layer + 1) as f32,
            (p.placeholder as u32 | p.surfaces << 1 | if p.cast_shadow { 0 } else { NO_SHADOW_BIT })
                as f32,
            // z: Reflectance, + 2 when hovered.
            material_reflectance(p.material_id, p.reflectance) + if p.hover { 2.0 } else { 0.0 },
            tex_mode(p),
        ],
        uvt: p.uvt,
        emissive: [p.emissive[0], p.emissive[1], p.emissive[2], p.tex_albedo],
        tint: match p.surface_tint {
            Some(c) => [c[0], c[1], c[2], 1.0],
            None => [1.0, 1.0, 1.0, 0.0],
        },
    }
}

/// shader.wgsl with the measured constants from mech.rs put in front.
/// The viewport shader's WGSL: the tuning constants (mech.rs), the quality
/// settings' constants (MSAA_SAMPLES, SUN_CASCADES, the lite shader's
/// LITE bits, RCAS_SHARPNESS) and shader.wgsl, its multisampled depth
/// bindings made plain ones when MSAA is off.
fn shader_source(settings: &crate::quality::RenderSettings) -> String {
    let quality = format!(
        "const MSAA_SAMPLES: i32 = {};\nconst SUN_CASCADES: i32 = {};\nconst LITE: u32 = {}u;\nconst RCAS_SHARPNESS: f32 = {:?};\n",
        settings.msaa.max(1),
        settings.sun_cascades.min(crate::sun_shadow::CASCADES as u32),
        settings.lite & crate::quality::LITE_ALL,
        crate::mech::UPSCALE_SHARPNESS
    );
    let grass = format!("const GRASS_RADIUS: f32 = {:?};\nconst GRASS_FADE_LEN: f32 = {:?};\nconst GRASS_NEAR_RADIUS: f32 = {:?};\nconst GRASS_NEAR_FADE_LEN: f32 = {:?};\n", crate::mech::GRASS_RADIUS, crate::mech::GRASS_FADE_LEN, crate::mech::GRASS_NEAR_RADIUS, crate::mech::GRASS_NEAR_FADE_LEN);
    use crate::mech as m;
    let [a, b, c, d, e] = m::TONE_CURVE;
    format!(
        "const SUN_DIST: f32 = {:?};\nconst HYBRID_LERP_SLOPE: f32 = {:?};\nconst TONE_A: f32 = {a:?};\nconst TONE_B: f32 = {b:?};\nconst TONE_C: f32 = {c:?};\nconst TONE_D: f32 = {d:?};\nconst TONE_E: f32 = {e:?};\nconst BLOOM_SIGMA_NARROW: f32 = {:?};\nconst BLOOM_SIGMA_WIDE: f32 = {:?};\nconst BLOOM_WIDE_SHARE: f32 = {:?};\nconst BLOOM_GAIN: f32 = {:?};\nconst BLOOM_MAX_RADIUS: i32 = {};\nconst ATMOS_BAND_DEG: f32 = {:?};\nconst ATMOS_BAND_POW: f32 = {:?};\nconst ATMOS_BAND_FLOOR: f32 = {:?};\nconst ENV_SPEC_LOD_SPAN: f32 = {:?};\nconst LIGHTGRID_BORDER: f32 = {:?};\nconst LIGHTOCC_DENSITY: f32 = {:?};\nconst LIGHTOCC_START_VOXELS: f32 = {:?};\nconst LIGHTOCC_LIGHT_SKIP_VOXELS: f32 = {:?};\nconst LIGHTOCC_MAX_STEPS: f32 = {:?};\nconst ATMOS_T_EPS: f32 = {:?};\nconst ATMOS_FOG_MIP: f32 = {:?};\nconst ATMOS_ENV_CUBE_LEVELS: f32 = {:?};\nconst ENV_SPEC_FADE: f32 = {:?};\nconst ENV_ROUGH_BASE: f32 = {:?};\nconst ENV_ROUGH_SCALE: f32 = {:?};\nconst ENV_SPEC_MIP_PER_ROUGHNESS: f32 = {:?};\nconst ENV_REFLECT_UP_GAIN: f32 = {:?};\nconst SSAO_DEPTH_SCALE: f32 = {:?};\nconst SSAO_DEPTH_MAX: f32 = {:?};\nconst SSAO_RADIUS_K: f32 = {:?};\nconst SSAO_RADIUS_MIN_TEXELS: f32 = {:?};\nconst SSAO_RADIUS_MAX_TEXELS: f32 = {:?};\nconst SSAO_SAMPLES: u32 = {}u;\nconst SSAO_DEPTH_GAIN: f32 = {:?};\nconst SSAO_BLUR_K: f32 = {:?};\nconst SSAO_BLUR_DEPTH: f32 = {:?};\nconst SSAO_BLUR_DEPTH_MIN: f32 = {:?};\nconst SSAO_BLUR_DEPTH_MAX: f32 = {:?};\nconst SSAO_UPSAMPLE_DEPTH: f32 = {:?};\nconst EVSM_POS_EXP: f32 = {:?};\nconst EVSM_NEG_EXP: f32 = {:?};\nconst EVSM_SHADOW_BIAS: f32 = {:?};\nconst EVSM_BLEED_CUT: f32 = {:?};\nconst LSHADOW_COLS: u32 = {}u;\nconst LSHADOW_CELL: f32 = {:?};\n{}",
        m::SUN_DIST,
        m::HYBRID_LERP_SLOPE,
        m::BLOOM_SIGMAS_PX[0],
        m::BLOOM_SIGMAS_PX[1],
        m::BLOOM_WIDE_SHARE,
        m::BLOOM_GAIN,
        m::BLOOM_MAX_RADIUS,
        m::ATMOS_BAND_DEG,
        m::ATMOS_BAND_POW,
        m::ATMOS_BAND_FLOOR,
        m::ENV_SPEC_LOD_SPAN,
        m::LIGHTGRID_BORDER,
        m::LIGHTOCC_DENSITY,
        m::LIGHTOCC_START_VOXELS,
        m::LIGHTOCC_LIGHT_SKIP_VOXELS,
        m::LIGHTOCC_MAX_STEPS,
        m::ATMOS_T_EPS,
        m::ATMOS_FOG_MIP,
        m::ATMOS_ENV_CUBE_LEVELS,
        m::ENV_SPEC_FADE,
        m::ENV_ROUGH_BASE,
        m::ENV_ROUGH_SCALE,
        m::ENV_SPEC_MIP_PER_ROUGHNESS,
        m::ENV_REFLECT_UP_GAIN,
        m::SSAO_DEPTH_SCALE,
        m::SSAO_DEPTH_MAX,
        m::SSAO_RADIUS_K,
        m::SSAO_RADIUS_MIN_TEXELS,
        m::SSAO_RADIUS_MAX_TEXELS,
        m::SSAO_SAMPLES,
        m::SSAO_DEPTH_GAIN,
        m::SSAO_BLUR_K,
        m::SSAO_BLUR_DEPTH,
        m::SSAO_BLUR_DEPTH_MIN,
        m::SSAO_BLUR_DEPTH_MAX,
        m::SSAO_UPSAMPLE_DEPTH,
        m::EVSM_POS_EXP,
        m::EVSM_NEG_EXP,
        m::EVSM_SHADOW_BIAS,
        m::EVSM_BLEED_CUT,
        m::LSHADOW_ATLAS / m::LSHADOW_FACE,
        m::LSHADOW_FACE as f32 / m::LSHADOW_ATLAS as f32,
        format!("const TERRAIN_CAPTURE_ID: bool = {};\nconst TERRAIN_CAPTURE_NO_AO: bool = {};\nconst DETILE_LEVELS: f32 = {:?};\nconst WANG_CELLS: f32 = {:?};\nconst MATERIAL_DIFFUSE_FADE_DEPTH: f32 = {:?};\nconst MATERIAL_LQ_DIFFUSE_CUTOFF: f32 = {:?};\n{}", std::env::var_os("RUSTUDIO_TERRAIN_CAPTURE_ID").is_some(), std::env::var_os("RUSTUDIO_TERRAIN_CAPTURE_NO_AO").is_some(), m::DETILE_LEVELS, m::WANG_CELLS, m::MATERIAL_DIFFUSE_FADE_DEPTH, m::MATERIAL_LQ_DIFFUSE_CUTOFF, format!("{quality}{grass}{}", msaa_wgsl(include_str!("shader.wgsl"), settings.msaa)))
    )
}

/// `src` for a scene drawn with `samples` samples: without multisampling
/// its `texture_depth_multisampled_2d` bindings become `texture_depth_2d`
/// and the see-through scene's `texture_multisampled_2d<f32>` a
/// `texture_2d<f32>` (a `textureLoad` with a sample index of 0 then reads
/// mip level 0; the sample loops run once: MSAA_SAMPLES).
pub fn msaa_wgsl(src: &str, samples: u32) -> String {
    if samples > 1 {
        src.to_string()
    } else {
        src.replace("texture_depth_multisampled_2d", "texture_depth_2d").replace("texture_multisampled_2d<f32>", "texture_2d<f32>")
    }
}

fn mip_levels(w: u32, h: u32) -> u32 {
    32 - w.max(h).leading_zeros()
}

/// An RGBA8 sRGB image followed by its mip levels (each half the size,
/// 2×2 texels averaged in linear light; alpha averaged as is).
/// PBR maps contain linear data; gamma-correct colour mips distort normals
/// and roughness. Keep the channels in their encoded data space.
fn data_mip_chain(mut w: u32, mut h: u32, rgba: &[u8]) -> Vec<u8> {
    let mut out = rgba.to_vec();
    let mut prev = rgba.to_vec();
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0; (nw * nh * 4) as usize];
        for y in 0..nh { for x in 0..nw { for c in 0..4 {
            let mut sum = 0u16;
            for (dx, dy) in [(0,0), (1,0), (0,1), (1,1)] {
                sum += prev[(((y*2+dy).min(h-1)*w + (x*2+dx).min(w-1))*4) as usize + c] as u16;
            }
            next[((y*nw+x)*4) as usize+c] = ((sum+2)/4) as u8;
        }}}
        out.extend_from_slice(&next);
        prev = next;
        w = nw; h = nh;
    }
    out
}

#[test]
fn pbr_mips_preserve_linear_channels() {
    let px = [0,0,0,0, 255,255,255,255, 0,0,0,0, 255,255,255,255];
    assert_eq!(&data_mip_chain(2,2,&px)[16..], &[128;4]);
}

/// Six square faces, each with its mip chain, layer-major (a cube upload).
pub fn cube_mips(size: u32, faces: &[Vec<u8>; 6]) -> Vec<u8> {
    let mut data: Vec<u8> = faces.iter().flat_map(|f| mip_chain(size, size, f)).collect();
    ggx_prefilter_cube(size, &mut data);
    data
}

/// Studio's PrefilteredEnvTexture: the sky at ENV_PREFILTER_SIZE px, each
/// mip below it GGX-filtered for roughness mip / ENV_SPEC_MIP_PER_ROUGHNESS
/// (FastConvolutionPS, Manson & Sloan's fast filtering, run per face and
/// level; the part FS reads it at lod roughness × 5). The
/// shader reads those levels of the skybox's own chain (lod offset by the
/// levels above ENV_PREFILTER_SIZE), so they're replaced in place here:
/// box-filtered levels left rough parts with sharp, glossy sky reflections.
/// Importance-sampled GGX (α = roughness², as the FS's sun term), N = V,
/// reading the box chain at the sample's solid angle.
fn ggx_prefilter_cube(size: u32, data: &mut [u8]) {
    use crate::mech as m;
    let levels = mip_levels(size, size) as usize;
    let base = (size.max(1) / m::ENV_PREFILTER_SIZE.min(size.max(1))).trailing_zeros() as usize;
    if base + 1 >= levels {
        return;
    }
    let to_lin: [f32; 256] = std::array::from_fn(|i| (i as f32 / 255.0).powf(2.2));
    let th: [f32; 255] = std::array::from_fn(|i| ((i as f32 + 0.5) / 255.0).powf(2.2));
    let to_srgb = |v: f32| th.partition_point(|&t| t <= v) as u8;
    let face_len: usize = (0..levels).map(|l| ((size >> l).max(1) as usize).pow(2) * 4).sum();
    let level_off = |l: usize| -> usize { (0..l).map(|k| ((size >> k).max(1) as usize).pow(2) * 4).sum() };
    // The source: the box chain from the base level down, linear RGB.
    let src: Vec<(usize, Vec<[f32; 3]>)> = (base..levels)
        .map(|l| {
            let w = (size >> l).max(1) as usize;
            let mut v = Vec::with_capacity(6 * w * w);
            for f in 0..6 {
                let o = f * face_len + level_off(l);
                for t in 0..w * w {
                    let p = &data[o + t * 4..o + t * 4 + 3];
                    v.push([to_lin[p[0] as usize], to_lin[p[1] as usize], to_lin[p[2] as usize]]);
                }
            }
            (w, v)
        })
        .collect();
    let fetch = |level: usize, d: Vec3| -> [f32; 3] {
        let (w, px) = &src[level.min(src.len() - 1)];
        let w = *w;
        let (f, sc, tc) = cube_face_st(d);
        let x = ((sc * w as f32 - 0.5).max(0.0)).min(w as f32 - 1.0);
        let y = ((tc * w as f32 - 0.5).max(0.0)).min(w as f32 - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(w - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let at = |xx: usize, yy: usize| px[f * w * w + yy * w + xx];
        let mut c = [0f32; 3];
        for (xx, yy, wt) in [(x0, y0, (1.0 - fx) * (1.0 - fy)), (x1, y0, fx * (1.0 - fy)), (x0, y1, (1.0 - fx) * fy), (x1, y1, fx * fy)] {
            let s = at(xx, yy);
            for ch in 0..3 {
                c[ch] += s[ch] * wt;
            }
        }
        c
    };
    let n_samples = m::ENV_PREFILTER_SAMPLES;
    let base_w = src[0].0 as f32;
    let texel_sa = 4.0 * std::f32::consts::PI / (6.0 * base_w * base_w);
    // One thread per face: each writes only its own face's levels.
    rt_thread::scope(|scope| {
        for (f, face) in data.chunks_mut(face_len).enumerate() {
            let fetch = &fetch;
            let to_srgb = &to_srgb;
            let level_off = &level_off;
            scope.spawn(move || {
                for l in base + 1..levels {
                    let w = (size >> l).max(1) as usize;
                    let rough = ((l - base) as f32 / m::ENV_SPEC_MIP_PER_ROUGHNESS).min(1.0);
                    let a = rough * rough;
                    let a2 = a * a;
                    let o = level_off(l);
                    for ty in 0..w {
                        for tx in 0..w {
                            let n = cube_face_dir(f, (tx as f32 + 0.5) / w as f32, (ty as f32 + 0.5) / w as f32);
                            let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
                            let tx_ = up.cross(n).normalize();
                            let ty_ = n.cross(tx_);
                            let mut acc = [0f32; 3];
                            let mut wsum = 0f32;
                            for i in 0..n_samples {
                                // Hammersley point → GGX half vector.
                                let u = (i as f32 + 0.5) / n_samples as f32;
                                let v = (i as u32).reverse_bits() as f32 / 4294967296.0;
                                let phi = 2.0 * std::f32::consts::PI * v;
                                let cos_t = ((1.0 - u) / (1.0 + (a2 - 1.0) * u)).sqrt();
                                let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
                                let h = tx_ * (sin_t * phi.cos()) + ty_ * (sin_t * phi.sin()) + n * cos_t;
                                let lv = h * (2.0 * cos_t) - n;
                                let nl = n.dot(lv);
                                if nl <= 0.0 {
                                    continue;
                                }
                                // pdf (N = V) = D / 4; the source mip whose
                                // texel covers this sample's solid angle.
                                let dd = cos_t * cos_t * (a2 - 1.0) + 1.0;
                                let d = a2 / (std::f32::consts::PI * dd * dd);
                                let sample_sa = 1.0 / (n_samples as f32 * d * 0.25 + 1e-4);
                                let mip = (0.5 * (sample_sa / texel_sa).log2() + 1.0).max(0.0);
                                let m0 = mip.floor() as usize;
                                let t = mip - m0 as f32;
                                let c0 = fetch(m0, lv);
                                let c1 = fetch(m0 + 1, lv);
                                for ch in 0..3 {
                                    acc[ch] += (c0[ch] * (1.0 - t) + c1[ch] * t) * nl;
                                }
                                wsum += nl;
                            }
                            let p = o + (ty * w + tx) * 4;
                            for ch in 0..3 {
                                face[p + ch] = to_srgb(acc[ch] / wsum.max(1e-6));
                            }
                        }
                    }
                }
            });
        }
    });
}

/// A unit direction's cube face (wgpu order +X, −X, +Y, −Y, +Z, −Z) and its
/// texture coordinates (s, t) in 0..1 (the Vulkan cube map selection).
fn cube_face_st(d: Vec3) -> (usize, f32, f32) {
    let a = d.abs();
    let (f, sc, tc, ma) = if a.x >= a.y && a.x >= a.z {
        if d.x > 0.0 { (0, -d.z, -d.y, a.x) } else { (1, d.z, -d.y, a.x) }
    } else if a.y >= a.z {
        if d.y > 0.0 { (2, d.x, d.z, a.y) } else { (3, d.x, -d.z, a.y) }
    } else if d.z > 0.0 {
        (4, d.x, -d.y, a.z)
    } else {
        (5, -d.x, -d.y, a.z)
    };
    (f, (sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5)
}

/// The direction through (s, t) of cube face `f` (inverse of cube_face_st).
fn cube_face_dir(f: usize, s: f32, t: f32) -> Vec3 {
    let (a, b) = (2.0 * s - 1.0, 2.0 * t - 1.0);
    match f {
        0 => Vec3::new(1.0, -b, -a),
        1 => Vec3::new(-1.0, -b, a),
        2 => Vec3::new(a, 1.0, b),
        3 => Vec3::new(a, -1.0, -b),
        4 => Vec3::new(a, -b, 1.0),
        _ => Vec3::new(-a, -b, -1.0),
    }
    .normalize()
}

#[test]
fn cube_face_mapping_round_trips() {
    for f in 0..6 {
        for (s, t) in [(0.2, 0.7), (0.5, 0.5), (0.9, 0.1)] {
            let (g, s2, t2) = cube_face_st(cube_face_dir(f, s, t));
            assert_eq!(g, f);
            assert!((s - s2).abs() < 1e-5 && (t - t2).abs() < 1e-5);
        }
    }
}

thread_local! {
    /// Mips made off the main thread for the image being uploaded (its
    /// pixels' address and the chain): see [`with_mips`].
    static READY_MIPS: std::cell::RefCell<Option<(usize, Vec<u8>)>> = const { std::cell::RefCell::new(None) };
}

/// The mip chain add_texture would make for `img` (None when it makes none).
pub fn mip_chain_for(img: &crate::texture::Rgba) -> Option<Vec<u8>> {
    let (w, h) = (img.width.max(1), img.height.max(1));
    (img.pixels.len() == (w * h * 4) as usize).then(|| mip_chain(w, h, &img.pixels))
}

/// Runs `upload` (which adds `pixels` as a texture) with `mips` already made
/// for it: add_texture takes them instead of building the chain itself.
pub fn with_mips<R>(pixels: &[u8], mips: Option<Vec<u8>>, upload: impl FnOnce() -> R) -> R {
    let Some(m) = mips else { return upload() };
    READY_MIPS.with(|c| *c.borrow_mut() = Some((pixels.as_ptr() as usize, m)));
    let r = upload();
    READY_MIPS.with(|c| c.borrow_mut().take());
    r
}

fn mip_chain(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    use std::sync::OnceLock;
    static TO_LIN: OnceLock<[f32; 256]> = OnceLock::new();
    let to_lin = TO_LIN.get_or_init(|| std::array::from_fn(|i| (i as f32 / 255.0).powf(2.2)));
    // round(v^(1/2.2) * 255) without powf: the byte whose rounding interval
    // holds v (thresholds ((s - 0.5) / 255)^2.2). powf per channel per texel
    // made mip chains a large share of a map load's texture time.
    static TH: OnceLock<[f32; 255]> = OnceLock::new();
    let th = TH.get_or_init(|| std::array::from_fn(|i| ((i as f32 + 0.5) / 255.0).powf(2.2)));
    let to_srgb = |v: f32| th.partition_point(|&t| t <= v) as u8;
    // (exactly the chain's size: growing a Vec of level 0 by doubling held
    // half as much again as the chain needs)
    let mut total = rgba.len();
    let (mut cw, mut ch) = (w, h);
    while cw > 1 || ch > 1 {
        (cw, ch) = ((cw / 2).max(1), (ch / 2).max(1));
        total += (cw * ch * 4) as usize;
    }
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(rgba);
    let (mut cw, mut ch) = (w, h);
    let mut prev = rgba.to_vec();
    while cw > 1 || ch > 1 {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(cw - 1);
                    let sy = (y * 2 + dy).min(ch - 1);
                    let o = ((sy * cw + sx) * 4) as usize;
                    for c in 0..3 {
                        acc[c] += to_lin[prev[o + c] as usize];
                    }
                    acc[3] += prev[o + 3] as f32;
                }
                let o = ((y * nw + x) * 4) as usize;
                for c in 0..3 {
                    next[o + c] = to_srgb(acc[c] * 0.25);
                }
                next[o + 3] = (acc[3] * 0.25 + 0.5) as u8;
            }
        }
        out.extend_from_slice(&next);
        prev = next;
        cw = nw;
        ch = nh;
    }
    out
}

pub fn mat_cols(m: Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

// ----------------------------------------------------------------- meshes

/// Collects triangles for a convex shape centred on the origin, flipping any
/// triangle whose winding faces inward so back-face culling is correct.
#[derive(Default)]
struct MeshBuilder {
    verts: Vec<Vertex>,
    /// A point strictly inside the shape.
    center: Vec3,
}

impl MeshBuilder {
    fn tri(&mut self, a: (Vec3, Vec3), b: (Vec3, Vec3), c: (Vec3, Vec3)) {
        let face_n = (b.0 - a.0).cross(c.0 - a.0);
        let centroid = (a.0 + b.0 + c.0) / 3.0;
        let (b, c) = if face_n.dot(centroid - self.center) < 0.0 {
            (c, b)
        } else {
            (b, c)
        };
        let fnorm = face_n.normalize_or(Vec3::Y)
            * if face_n.dot(centroid - self.center) < 0.0 {
                -1.0
            } else {
                1.0
            };
        for (p, n) in [a, b, c] {
            self.verts.push(Vertex {
                pos: p.to_array(),
                normal: n.to_array(),
                uv: box_uv(p, fnorm),
                projection: [0.0; 4],
                color: [0.0; 4],
            });
        }
    }

    /// A flat polygon (convex, given in order).
    fn flat(&mut self, pts: &[Vec3]) {
        let n = (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize();
        let centroid = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
        let n = if n.dot(centroid - self.center) < 0.0 {
            -n
        } else {
            n
        };
        for i in 1..pts.len() - 1 {
            self.tri((pts[0], n), (pts[i], n), (pts[i + 1], n));
        }
    }
}

/// Selection uses the same tessellation and size transform as the visible ball.
pub(crate) fn ball_outline_points() -> impl Iterator<Item = Vec3> {
    (0..=18).flat_map(|i| (0..32).map(move |j| {
        let th = std::f32::consts::PI * i as f32 / 18.0;
        let ph = std::f32::consts::TAU * j as f32 / 32.0;
        Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()) * 0.5
    }))
}

pub(crate) fn outline_triangles(kind: MeshKind) -> &'static [[Vec3; 3]] {
    static TRIANGLES: std::sync::OnceLock<Vec<Vec<[Vec3; 3]>>> = std::sync::OnceLock::new();
    &TRIANGLES.get_or_init(|| build_meshes().iter().map(|vertices|
        vertices.chunks_exact(3).map(|t| [Vec3::from(t[0].pos), Vec3::from(t[1].pos), Vec3::from(t[2].pos)]).collect()
    ).collect())[kind as usize]
}

fn build_meshes() -> Vec<Vec<Vertex>> {
    let h = 0.5;
    let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);

    // Block
    let mut block = MeshBuilder::default();
    for axis in 0..3 {
        for s in [-h, h] {
            let corner = |a: f32, b: f32| {
                let mut p = Vec3::ZERO;
                p[axis] = s;
                p[(axis + 1) % 3] = a;
                p[(axis + 2) % 3] = b;
                p
            };
            block.flat(&[corner(-h, -h), corner(h, -h), corner(h, h), corner(-h, h)]);
        }
    }

    // Ball
    let mut ball = MeshBuilder::default();
    let (rings, segs) = (18, 32);
    let sp = |i: usize, j: usize| {
        let th = std::f32::consts::PI * i as f32 / rings as f32;
        let ph = std::f32::consts::TAU * j as f32 / segs as f32;
        let n = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
        (n * h, n)
    };
    for i in 0..rings {
        for j in 0..segs {
            let (a, b, c, d) = (sp(i, j), sp(i + 1, j), sp(i + 1, j + 1), sp(i, j + 1));
            if i != 0 {
                ball.tri(a, b, d);
            }
            if i != rings - 1 {
                ball.tri(b, c, d);
            }
        }
    }

    // Cylinder (axis along X, like Roblox cylinders)
    let mut cyl = MeshBuilder::default();
    let segs = 32;
    let ring = |j: usize| {
        let a = std::f32::consts::TAU * j as f32 / segs as f32;
        Vec3::new(0.0, a.cos(), a.sin())
    };
    for j in 0..segs {
        let (n0, n1) = (ring(j), ring(j + 1));
        let (p0, p1) = (n0 * h, n1 * h);
        cyl.tri(
            (p0 + v(-h, 0.0, 0.0), n0),
            (p0 + v(h, 0.0, 0.0), n0),
            (p1 + v(h, 0.0, 0.0), n1),
        );
        cyl.tri(
            (p0 + v(-h, 0.0, 0.0), n0),
            (p1 + v(h, 0.0, 0.0), n1),
            (p1 + v(-h, 0.0, 0.0), n1),
        );
        for s in [-h, h] {
            let n = v(s.signum(), 0.0, 0.0);
            cyl.tri(
                (v(s, 0.0, 0.0), n),
                (p0 + v(s, 0.0, 0.0), n),
                (p1 + v(s, 0.0, 0.0), n),
            );
        }
    }

    // Wedge: full height at the back (+Z), sloping down to the front (-Z).
    let mut wedge = MeshBuilder {
        center: v(0.0, -0.2, 0.2),
        ..Default::default()
    };
    wedge.flat(&[v(-h, -h, -h), v(h, -h, -h), v(h, -h, h), v(-h, -h, h)]); // bottom
    wedge.flat(&[v(-h, -h, h), v(h, -h, h), v(h, h, h), v(-h, h, h)]); // back
    wedge.flat(&[v(-h, -h, -h), v(h, -h, -h), v(h, h, h), v(-h, h, h)]); // slope
    wedge.flat(&[v(-h, -h, -h), v(-h, -h, h), v(-h, h, h)]); // left
    wedge.flat(&[v(h, -h, -h), v(h, -h, h), v(h, h, h)]); // right

    // Corner wedge: apex above the right-front corner.
    let mut corner = MeshBuilder {
        center: v(0.25, -0.3, -0.25),
        ..Default::default()
    };
    let apex = v(h, h, -h);
    let (b0, b1, b2, b3) = (v(-h, -h, -h), v(h, -h, -h), v(h, -h, h), v(-h, -h, h));
    corner.flat(&[b0, b1, b2, b3]);
    corner.flat(&[apex, b1, b2]);
    corner.flat(&[apex, b0, b1]);
    corner.flat(&[apex, b2, b3]);
    corner.flat(&[apex, b3, b0]);

    // Explicitly mark only flat oblique primitive faces. Curved normals and
    // authored union projection tags retain their existing material mapping.
    for mesh in [&mut wedge, &mut corner] {
        for vertex in &mut mesh.verts {
            let n = Vec3::from_array(vertex.normal).abs();
            if n.y > 0.0001 && (n.x > 0.0001 || n.z > 0.0001) {
                vertex.projection[0] = -99.0;
            }
        }
    }
    vec![
        block.verts,
        ball.verts,
        cyl.verts,
        wedge.verts,
        corner.verts,
    ]
}

/// The triangles of a built-in Wedge / CornerWedge in unit-part space (the
/// same shapes the meshes above draw); None for other kinds. Decals and
/// Textures are projected onto them (viewport.rs), as Studio clips a
/// Texture on a wedge's Front or Top to its slope.
pub fn shape_triangles(kind: MeshKind) -> Option<Vec<[Vec3; 3]>> {
    let h = 0.5;
    let v = Vec3::new;
    let quad = |q: [Vec3; 4]| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]];
    let mut out = Vec::new();
    match kind {
        MeshKind::Wedge => {
            out.extend(quad([v(-h, -h, -h), v(h, -h, -h), v(h, -h, h), v(-h, -h, h)]));
            out.extend(quad([v(-h, -h, h), v(h, -h, h), v(h, h, h), v(-h, h, h)]));
            out.extend(quad([v(-h, -h, -h), v(h, -h, -h), v(h, h, h), v(-h, h, h)]));
            out.push([v(-h, -h, -h), v(-h, -h, h), v(-h, h, h)]);
            out.push([v(h, -h, -h), v(h, -h, h), v(h, h, h)]);
        }
        MeshKind::CornerWedge => {
            let apex = v(h, h, -h);
            let (b0, b1, b2, b3) = (v(-h, -h, -h), v(h, -h, -h), v(h, -h, h), v(-h, -h, h));
            out.extend(quad([b0, b1, b2, b3]));
            out.push([apex, b1, b2]);
            out.push([apex, b0, b1]);
            out.push([apex, b2, b3]);
            out.push([apex, b3, b0]);
        }
        _ => return None,
    }
    // Winding doesn't matter here: the projection keeps triangles by the
    // normal's side, so orient each one away from the shape's centre.
    let centre = if kind == MeshKind::Wedge { v(0.0, -0.2, 0.2) } else { v(0.25, -0.3, -0.25) };
    for tri in &mut out {
        let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
        if n.dot((tri[0] + tri[1] + tri[2]) / 3.0 - centre) < 0.0 {
            tri.swap(1, 2);
        }
    }
    Some(out)
}

thread_local! {
    static REUSE_TEXTURE_SLOT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// The next part texture upload on this thread replaces slot `index`
/// instead of taking a new one (assets.rs, for EditableImage updates).
pub fn reuse_texture_slot(index: u32) {
    REUSE_TEXTURE_SLOT.with(|c| c.set(Some(index)));
}
