//! `--headless` / `--no-render`: run the app with no window, no compositor
//! and no display server (no gamescope or Xvfb wrapper).
//!
//! `--headless` makes a wgpu device without a surface and paints each egui
//! frame into an offscreen Rgba8Unorm texture the size of `--window WxH`;
//! `ViewportCommand::Screenshot` reads that texture back and answers with an
//! `Event::Screenshot` the next frame, as eframe does for a window, so the
//! harness (--shots, --screenshot) is the same code. `--no-render` makes no
//! device at all: simulation, scripts, physics and audio state run, egui
//! lays the frame out, nothing is drawn, and screenshots come back empty
//! (the harness writes no picture, only the log and frame times).
//!
//! Frames are paced at 60 Hz (a display's vsync, which a windowed test run
//! gets from gamescope) except under --bench, which measures unpaced.

use std::sync::Arc;
use std::time::Duration;
use web_time::Instant;

use egui_wgpu::wgpu;

pub struct Options {
    pub size: [u32; 2],
    pub render: bool,
    pub paced: bool,
}

/// The wgpu device as eframe would make it (rustudio's limits and
/// features: main.rs), without a surface.
fn render_state(config: &egui_wgpu::WgpuConfiguration) -> Result<egui_wgpu::RenderState, String> {
    pollster::block_on(async {
        let instance = config.wgpu_setup.new_instance().await;
        egui_wgpu::RenderState::create(config, &instance, None, egui_wgpu::RendererOptions::default())
            .await
            .map_err(|e| e.to_string())
    })
}

struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

fn target(rs: &egui_wgpu::RenderState, size: [u32; 2]) -> Target {
    let texture = rs.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen frame"),
        size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: rs.target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    Target { texture, view }
}

/// The frame's pixels (the same conversion as egui_wgpu's window capture).
fn read_back(rs: &egui_wgpu::RenderState, t: &Target, size: [u32; 2]) -> egui::ColorImage {
    let [w, h] = size;
    let row = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = rs.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = rs.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        t.texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    rs.queue.submit([enc.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = rs.device.poll(wgpu::PollType::wait_indefinitely());
    if !matches!(rx.recv(), Ok(Ok(()))) {
        return egui::ColorImage::new([0, 0], Vec::new());
    }
    let order = match rs.target_format {
        wgpu::TextureFormat::Bgra8Unorm => [2, 1, 0, 3],
        _ => [0, 1, 2, 3],
    };
    let Ok(data) = slice.get_mapped_range() else {
        return egui::ColorImage::new([0, 0], Vec::new());
    };
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        let r = &data[y * row as usize..][..w as usize * 4];
        for p in r.chunks_exact(4) {
            pixels.push(egui::Color32::from_rgba_premultiplied(p[order[0]], p[order[1]], p[order[2]], p[order[3]]));
        }
    }
    drop(data);
    buffer.unmap();
    egui::ColorImage::new([w as usize, h as usize], pixels)
}

/// Runs the app until it asks to close; returns the exit code.
pub fn run<A: eframe::App>(
    config: egui_wgpu::WgpuConfiguration,
    opts: Options,
    make: impl FnOnce(&egui::Context, Option<egui_wgpu::RenderState>) -> A,
    mut draw: impl FnMut(&mut A, &mut egui::Ui),
) -> i32 {
    let ctx = egui::Context::default();
    let rs = if opts.render {
        match render_state(&config) {
            Ok(rs) => Some(rs),
            Err(e) => {
                eprintln!("--headless: no GPU device without a window: {e}");
                return 1;
            }
        }
    } else {
        None
    };
    let atlas_device_limit = rs.as_ref().map(|rs| rs.device.limits().max_texture_dimension_2d as usize);
    let mut font_limit = None;
    let size = [opts.size[0].max(1), opts.size[1].max(1)];
    let frame = rs.as_ref().map(|rs| target(rs, size));
    let mut app = make(&ctx, rs.clone());
    let start = Instant::now();
    let period = Duration::from_secs_f64(1.0 / 60.0);
    let mut next = Instant::now();
    let mut events: Vec<egui::Event> = vec![egui::Event::WindowFocused(true)];
    loop {
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0] as f32, size[1] as f32),
            )),
            time: Some(start.elapsed().as_secs_f64()),
            predicted_dt: 1.0 / 60.0,
            max_texture_side: font_limit,
            focused: true,
            events: std::mem::take(&mut events),
            ..Default::default()
        };
        {
            let v = raw.viewports.entry(egui::ViewportId::ROOT).or_default();
            v.native_pixels_per_point = Some(1.0);
            v.focused = Some(true);
            v.inner_rect = raw.screen_rect;
            v.outer_rect = raw.screen_rect;
            v.monitor_size = Some(egui::vec2(size[0] as f32, size[1] as f32));
        }
        app.raw_input_hook(&ctx, &mut raw);
        let mut out = ctx.run_ui(raw, |ui| {
            draw(&mut app, ui);
            // RawInput defaults to a 2048 atlas offscreen. Keep its packing
            // for small scenes, but grow to the device limit when epaint
            // would otherwise discard the full atlas and start over.
            if font_limit.is_none() {
                if let Some(limit) = atlas_device_limit {
                    if ctx.fonts(|fonts| {
                        fonts.options().max_texture_side < limit && fonts.font_atlas_fill_ratio() > 0.8
                    }) {
                        font_limit = Some(limit);
                    }
                }
            }
        });
        // (epaint's debug builds assert that a TexturesDelta was handled
        // before it is dropped: applied below where there is a GPU, and
        // there is nothing to apply it to without one)
        let mut textures_delta = std::mem::take(&mut out.textures_delta);
        let mut close = false;
        let mut shots = Vec::new();
        for (_, v) in out.viewport_output.iter() {
            for c in &v.commands {
                match c {
                    egui::ViewportCommand::Close => close = true,
                    egui::ViewportCommand::Screenshot(data) => shots.push(data.clone()),
                    _ => {}
                }
            }
        }
        let ppp = out.pixels_per_point;
        if let (Some(rs), Some(t)) = (rs.as_ref(), frame.as_ref()) {
            let jobs = ctx.tessellate(out.shapes, ppp);
            let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: size, pixels_per_point: ppp };
            let mut renderer = rs.renderer.write();
            for (id, deltas) in &textures_delta.set {
                for delta in deltas {
                    renderer.update_texture(&rs.device, &rs.queue, *id, delta);
                }
            }
            let mut enc = rs.device.create_command_encoder(&Default::default());
            let extra = renderer.update_buffers(&rs.device, &rs.queue, &mut enc, &jobs, &screen);
            {
                let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("offscreen egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &t.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
            }
            rustudio_render::gpu_timing::submit(&rs.device, &rs.queue, "gpu: egui (offscreen)", extra.into_iter().chain([enc.finish()]));
            for id in &textures_delta.free {
                renderer.free_texture(id);
            }
            drop(renderer);
            if !shots.is_empty() {
                let image = Arc::new(read_back(rs, t, size));
                for user_data in shots {
                    events.push(egui::Event::Screenshot {
                        viewport_id: egui::ViewportId::ROOT,
                        user_data,
                        image: image.clone(),
                    });
                }
            } else {
                // (one frame in flight at most, as a presenting window)
                let _ = rs.device.poll(wgpu::PollType::wait_indefinitely());
            }
        } else {
            // No GPU: the harness gets an empty picture (it writes none).
            for user_data in shots {
                events.push(egui::Event::Screenshot {
                    viewport_id: egui::ViewportId::ROOT,
                    user_data,
                    image: Arc::new(egui::ColorImage::new([0, 0], Vec::new())),
                });
            }
        }
        textures_delta.clear();
        if close {
            app.on_exit();
            return 0;
        }
        if opts.paced {
            next += period;
            let now = Instant::now();
            if next > now {
                rt_thread::sleep(next - now);
            } else if now - next > period * 4 {
                // (fell behind: don't run a burst of catch-up frames)
                next = now;
            }
        }
    }
}

/// The interactive window presents in mailbox mode (Settings > rustudio >
/// Performance > Presentation, `RUSTUDIO_PERF_PRESENT=fifo` forces Fifo).
pub fn interactive_mailbox() -> bool {
    rustudio_core::rustudio_settings::performance_value("present") != "fifo"
}

/// eframe's wgpu setup, shared by the desktop window, the player and the
/// Android activity: the renderer samples up to 24 textures in one draw
/// (render.rs), vsync unless measuring. On Android the backends are Vulkan
/// first, then OpenGL ES (devices without a working Vulkan driver), chosen
/// by wgpu in that order; the surface is created from the activity's window
/// on Resumed, so the instance starts without a display handle everywhere.
pub fn wgpu_options(bench_mode: bool) -> egui_wgpu::WgpuConfiguration {
    wgpu_options_with(bench_mode, egui_wgpu::WgpuSetup::CreateNew)
}

/// `wgpu_options` with the last word on the setup: `finish` takes the
/// configured `WgpuSetupCreateNew` and answers the setup eframe runs with
/// (the studio hands a Snapdragon phone's Turnip driver its own objects:
/// `rustudio::wgpu_setup`).
pub fn wgpu_options_with(
    bench_mode: bool,
    finish: impl FnOnce(egui_wgpu::WgpuSetupCreateNew) -> egui_wgpu::WgpuSetup,
) -> egui_wgpu::WgpuConfiguration {
    egui_wgpu::WgpuConfiguration {
        wgpu_setup: {
            let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
            // wgpu's build-config flags turn the Khronos validation layer on
            // in every debug-assertions build (the dev profile): it checked
            // every GPU call on the render thread and tore the device down
            // under it at exit (a SIGSEGV in libVkLayer_khronos_validation
            // while the render thread freed a buffer, a coredump's stack).
            // WGPU_VALIDATION=1 / WGPU_DEBUG=1 still turn them on.
            setup.instance_descriptor.flags = egui_wgpu::wgpu::InstanceFlags::empty().with_env();
            setup.power_preference = egui_wgpu::wgpu::PowerPreference::from_env()
                .unwrap_or(egui_wgpu::wgpu::PowerPreference::HighPerformance);
            #[cfg(target_os = "android")]
            {
                setup.instance_descriptor.backends = egui_wgpu::wgpu::Backends::from_env()
                    .unwrap_or(egui_wgpu::wgpu::Backends::VULKAN | egui_wgpu::wgpu::Backends::GL);
            }
            let descriptor = setup.device_descriptor.clone();
            // A real GPU before a software one, a native API before
            // OpenGL (gpu_caps::rank_adapters): on Windows an HD 4600's GL
            // driver before WARP, its only DX12 adapter. The vendored
            // egui-wgpu tries the next one when a device can't be made.
            setup.native_adapter_selector = Some(std::sync::Arc::new(|adapters, surface| {
                rustudio_render::gpu_caps::rank_adapters(adapters, surface)
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("no graphics adapter can draw to this window ({} found)", adapters.len()))
            }));
            setup.device_descriptor = std::sync::Arc::new(move |adapter| {
                let mut d = descriptor(adapter);
                // Only what the adapter offers (gpu_caps.rs): unchanged on a
                // capable GPU; OpenGL's 16 texture units, WARP's limits and
                // the like get the renderer's reduced binding set.
                d.required_limits = rustudio_render::gpu_caps::required_limits(&adapter.limits());
                // Studio HBAO writes its full-resolution R8 visibility UAV.
                d.required_features |= adapter.features() & (egui_wgpu::wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | egui_wgpu::wgpu::Features::RG11B10UFLOAT_RENDERABLE);
                // RUSTUDIO_GPU_TIMING: timestamp queries (gpu_timing.rs).
                d.required_features |= rustudio_render::gpu_timing::features(adapter.features());
                // The driver's pipeline cache, kept between runs (Vulkan;
                // pipeline_cache.rs): a second start skips the shader compiles
                // of its first.
                d.required_features |= adapter.features() & egui_wgpu::wgpu::Features::PIPELINE_CACHE;
                // GPU memory in blocks sized for memory use (8-64 MB), not
                // wgpu's default 128-256 MB device blocks: the allocator keeps
                // its last empty block, so a 1440 island settled 2.18-2.19 GB
                // with them and 1.79-1.81 GB with these, with the same frame
                // instructions, GPU time and texture load time (RX 9070,
                // Vulkan; Windows RTX 3070: 1.83-2.09 -> 1.58-1.61 GB).
                // RUSTUDIO_GPU_MEMORY=performance: wgpu's default blocks.
                if std::env::var("RUSTUDIO_GPU_MEMORY").ok().as_deref() != Some("performance") {
                    d.memory_hints = egui_wgpu::wgpu::MemoryHints::MemoryUsage;
                }
                d
            });
            finish(setup)
        },
        surface: egui_wgpu::SurfaceConfig {
            // No vsync when measuring or driven for captures: a shot waits
            // for a few frames after each camera move.
            present_mode: present_mode(bench_mode || std::env::var_os("RUSTUDIO_CONTROL").is_some()),
            desired_maximum_frame_latency: Some(frame_latency()),
        },
        ..Default::default()
    }
}

/// The swapchain's present mode: no vsync when measuring (`no_vsync`);
/// else the Presentation setting's — Mailbox (Studio presents without
/// waiting for the refresh, its D3D11 frames run past the display rate
/// with a higher TaskSchedulerTargetFps, and paces them with its own
/// frame cap; composited, so nothing tears; Mailbox is the tear-free
/// equivalent, the vendored egui-wgpu falls back to Fifo on a surface
/// without it, and native_loop.rs paces redraws; FIFO blocked a low-end
/// laptop's main thread on the GPU, a test place 24.6 -> 30.5 fps with
/// mailbox) or `AutoVsync` (FIFO, relaxed where the driver has it) for
/// "Wait for vsync". On Android it is `AutoVsync` whatever the setting:
/// the activity runs eframe's own loop (src/android/mod.rs), with no
/// native_loop to pace Mailbox frames to the refresh, and an unpaced
/// Mailbox loop on a phone draws as fast as the GPU allows — heat and
/// battery for frames the display never shows (the standalone player
/// keeps vsync for the same reason). `RUSTUDIO_PRESENT_MODE` overrides
/// for measuring — `fifo`, `relaxed`, `mailbox`, `immediate`, `vsync`,
/// `novsync`; a mode the surface lacks falls back to FIFO in wgpu, with a
/// warning in its log.
pub fn present_mode(no_vsync: bool) -> egui_wgpu::wgpu::PresentMode {
    use egui_wgpu::wgpu::PresentMode;
    match std::env::var("RUSTUDIO_PRESENT_MODE").ok().as_deref().map(str::trim) {
        Some("fifo") => PresentMode::Fifo,
        Some("relaxed") => PresentMode::FifoRelaxed,
        Some("mailbox") => PresentMode::Mailbox,
        Some("immediate") => PresentMode::Immediate,
        Some("vsync") => PresentMode::AutoVsync,
        Some("novsync") => PresentMode::AutoNoVsync,
        _ if no_vsync => PresentMode::AutoNoVsync,
        _ if interactive_mailbox() && cfg!(not(target_os = "android")) => PresentMode::Mailbox,
        _ => PresentMode::AutoVsync,
    }
}

/// Frames the presentation engine may hold ahead of the one being drawn:
/// the swapchain gets one image more than this. 1 (two images) on a
/// computer, egui's low-latency default. On Android it is 2: the
/// compositor keeps the image on screen until it has latched the next
/// one, so with two images every frame first waits for that — a vsync of
/// nothing, half the refresh rate at best (the first phone measurements:
/// ~31 ms waited per frame at 60 Hz whether the camera moved or not; egui
/// notes the same halving on iOS). Three images let the next frame start
/// while the last one is still queued. `RUSTUDIO_FRAME_LATENCY` (1 to 3)
/// overrides, for measuring.
pub fn frame_latency() -> u32 {
    let default = if cfg!(target_os = "android") { 2 } else { 1 };
    std::env::var("RUSTUDIO_FRAME_LATENCY")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map_or(default, |n| n.clamp(1, 3))
}
