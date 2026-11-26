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

use eframe::App as _;
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
pub fn run(
    config: egui_wgpu::WgpuConfiguration,
    opts: Options,
    make: impl FnOnce(&egui::Context, Option<egui_wgpu::RenderState>) -> crate::app::RustudioApp,
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
        let mut out = ctx.run_ui(raw, |ui| app.frame(ui));
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
            crate::gpu_timing::submit(&rs.device, &rs.queue, "gpu: egui (offscreen)", extra.into_iter().chain([enc.finish()]));
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
