//! Non-intrusive viewport performance HUD (FPS, 1% low, CPU/GPU frame times,
//! 3D render scale, active graphics backend/adapter, and process RSS memory).
//!
//! Strictly honest measurements:
//! - GPU frame time is `Option<f32>` sourced from real timestamp queries
//!   (`rustudio_render::gpu_timing::last_scene_ms()`). When the driver or
//!   adapter does not expose timestamp queries, it displays
//!   `"N/A (driver unsupported)"` and never fabricates numbers.
//! - Memory usage (RSS MB) is sampled at most once per second from
//!   `/proc/self/statm` on Linux/Android and displays `None` (`"N/A"`) if
//!   unavailable.

use std::collections::VecDeque;
use std::time::Duration;
use web_time::Instant;

use egui::{Color32, FontId, RichText, Stroke};

/// Default ring buffer capacity (120 frames ≈ 2 seconds at 60 FPS).
pub const DEFAULT_HISTORY_LEN: usize = 120;

/// Minimum frame samples required before reporting 1% low FPS.
pub const MIN_SAMPLES_FOR_LOW_FPS: usize = 10;

/// Text shown when GPU timestamp queries are unavailable from the driver.
pub const GPU_UNSUPPORTED_LABEL: &str = "N/A (driver unsupported)";

/// Interval between `/proc/self/statm` RSS reads.
const MEM_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// Single frame timing sample stored in the ring buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameSample {
    /// Total frame delta time in milliseconds.
    pub dt_ms: f32,
    /// CPU work time for the frame in milliseconds.
    pub cpu_ms: f32,
    /// Real GPU scene time in milliseconds, or `None` when unsupported/not ready.
    pub gpu_ms: Option<f32>,
}

/// Rolling frame timing and hardware/memory metrics collector.
#[derive(Debug, Clone)]
pub struct PerfMetrics {
    capacity: usize,
    samples: VecDeque<FrameSample>,
    render_scale: f32,
    render_target_px: [u32; 2],
    backend: String,
    adapter_name: String,
    rss_mb: Option<f32>,
    last_mem_sample: Option<Instant>,
}

impl Default for PerfMetrics {
    fn default() -> Self {
        Self::new(DEFAULT_HISTORY_LEN)
    }
}

impl PerfMetrics {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            samples: VecDeque::with_capacity(cap),
            render_scale: 1.0,
            render_target_px: [0, 0],
            backend: String::new(),
            adapter_name: String::new(),
            rss_mb: None,
            last_mem_sample: None,
        }
    }

    /// Number of frame samples currently stored in the ring buffer.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clears all recorded timing samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Records a frame's wall-clock `dt_ms`, `cpu_ms`, and optional real `gpu_ms`.
    /// Also refreshes process RSS memory at most once per second.
    pub fn push_frame(&mut self, dt_ms: f32, cpu_ms: f32, gpu_ms: Option<f32>) {
        if !dt_ms.is_finite() || dt_ms <= 0.0 {
            return;
        }
        let cpu = if cpu_ms.is_finite() && cpu_ms >= 0.0 { cpu_ms } else { 0.0 };
        let gpu = gpu_ms.filter(|ms| ms.is_finite() && *ms >= 0.0);

        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(FrameSample { dt_ms, cpu_ms: cpu, gpu_ms: gpu });
        self.maybe_refresh_memory();
    }

    /// Records a frame and queries `rustudio_render::gpu_timing::last_scene_ms()`
    /// for real GPU timing (remains `None` when unsupported or disabled).
    pub fn push_frame_with_gpu_timing(&mut self, dt_ms: f32, cpu_ms: f32) {
        let gpu_ms = rustudio_render::gpu_timing::last_scene_ms();
        self.push_frame(dt_ms, cpu_ms, gpu_ms);
    }

    /// Updates the current 3D render-resolution scale (`0.25..=1.0`) and the
    /// actual 3D render-target pixel dimensions.
    pub fn set_render_target(&mut self, scale: f32, width: u32, height: u32) {
        if scale.is_finite() && scale > 0.0 {
            self.render_scale = scale;
        }
        self.render_target_px = [width, height];
    }

    /// Sets the active graphics backend label (e.g. `"Vulkan"`, `"GL"`) and GPU adapter name.
    pub fn set_gpu_info(&mut self, backend: impl Into<String>, adapter_name: impl Into<String>) {
        self.backend = backend.into();
        self.adapter_name = adapter_name.into();
    }

    /// Populates graphics backend and adapter name directly from a `wgpu::AdapterInfo`.
    pub fn set_gpu_info_from_wgpu(&mut self, info: &egui_wgpu::wgpu::AdapterInfo) {
        self.backend = format!("{:?}", info.backend);
        self.adapter_name = info.name.clone();
    }

    /// Average frame time in milliseconds across the ring buffer.
    pub fn avg_frame_ms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().map(|s| s.dt_ms).sum();
        sum / self.samples.len() as f32
    }

    /// Smoothed FPS derived from the average frame time in the ring buffer.
    pub fn smoothed_fps(&self) -> f32 {
        let avg = self.avg_frame_ms();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }

    /// Instantaneous FPS derived from the most recent frame's `dt_ms`.
    pub fn instant_fps(&self) -> f32 {
        match self.samples.back() {
            Some(s) if s.dt_ms > 0.0 => 1000.0 / s.dt_ms,
            _ => 0.0,
        }
    }

    /// 1% low FPS computed from the 99th-percentile frame time in the ring buffer.
    /// Returns `None` until at least `MIN_SAMPLES_FOR_LOW_FPS` (10) samples exist.
    pub fn one_percent_low_fps(&self) -> Option<f32> {
        let n = self.samples.len();
        if n < MIN_SAMPLES_FOR_LOW_FPS {
            return None;
        }
        let mut sorted: Vec<f32> = self.samples.iter().map(|s| s.dt_ms).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let p99_idx = (((n as f32) * 0.99).ceil() as usize).saturating_sub(1).min(n - 1);
        let p99_ms = sorted[p99_idx];
        (p99_ms > 0.0).then(|| 1000.0 / p99_ms)
    }

    /// Average CPU work time in milliseconds across the ring buffer.
    pub fn avg_cpu_ms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().map(|s| s.cpu_ms).sum();
        sum / self.samples.len() as f32
    }

    /// Average real GPU scene time in milliseconds over samples that have a
    /// real GPU measurement, or `None` when the driver exposes no GPU timings.
    pub fn avg_gpu_ms(&self) -> Option<f32> {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for s in &self.samples {
            if let Some(ms) = s.gpu_ms {
                sum += ms;
                count += 1;
            }
        }
        (count > 0).then(|| sum / count as f32)
    }

    /// Most recent real GPU scene time in milliseconds, if any sample has one.
    pub fn last_gpu_ms(&self) -> Option<f32> {
        self.samples.iter().rev().find_map(|s| s.gpu_ms)
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    pub fn render_target_px(&self) -> [u32; 2] {
        self.render_target_px
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Resident set size in MiB (sampled at most once per second), or `None` if unavailable.
    pub fn rss_mb(&self) -> Option<f32> {
        self.rss_mb
    }

    fn maybe_refresh_memory(&mut self) {
        let should_sample = self
            .last_mem_sample
            .map_or(true, |t| t.elapsed() >= MEM_SAMPLE_INTERVAL);
        if should_sample {
            self.rss_mb = read_process_rss_mb();
            self.last_mem_sample = Some(Instant::now());
        }
    }
}

/// Formats an optional real GPU frame time. Never fabricates a value when `None`.
pub fn format_gpu_ms(gpu_ms: Option<f32>) -> String {
    match gpu_ms {
        Some(ms) if ms.is_finite() && ms >= 0.0 => format!("{ms:.2} ms"),
        _ => GPU_UNSUPPORTED_LABEL.to_string(),
    }
}

/// Formats the current 3D render scale percentage and actual 3D target pixel dimensions.
pub fn format_render_scale(scale: f32, target_px: [u32; 2]) -> String {
    let pct = (scale * 100.0).round().max(0.0) as u32;
    if target_px[0] > 0 && target_px[1] > 0 {
        format!("{pct}% ({}x{})", target_px[0], target_px[1])
    } else {
        format!("{pct}%")
    }
}

/// Parses RSS in MiB from the contents of Linux/Android `/proc/self/statm`.
/// Format: `<size> <resident> <shared> <text> <lib> <data> <dt>` (in pages).
pub fn parse_proc_statm_rss_mb(statm: &str, page_size_bytes: u64) -> Option<f32> {
    if page_size_bytes == 0 {
        return None;
    }
    let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    let bytes = resident_pages.checked_mul(page_size_bytes)?;
    Some(bytes as f32 / (1024.0 * 1024.0))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_process_rss_mb() -> Option<f32> {
    let text = std::fs::read_to_string("/proc/self/statm").ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    parse_proc_statm_rss_mb(&text, page_size as u64)
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn read_process_rss_mb() -> Option<f32> {
    None
}

/// Non-intrusive translucent performance overlay drawn in the top-left of the 3D viewport.
#[derive(Debug, Clone, Default)]
pub struct PerfOverlay {
    pub metrics: PerfMetrics,
}

impl PerfOverlay {
    pub fn new() -> Self {
        Self { metrics: PerfMetrics::default() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { metrics: PerfMetrics::new(capacity) }
    }

    /// Records a frame with explicit `gpu_ms` (`None` when unsupported).
    pub fn record_frame(&mut self, dt_ms: f32, cpu_ms: f32, gpu_ms: Option<f32>) {
        self.metrics.push_frame(dt_ms, cpu_ms, gpu_ms);
    }

    /// Records a frame and reads real GPU timing from `rustudio_render::gpu_timing::last_scene_ms()`.
    pub fn record_frame_auto_gpu(&mut self, dt_ms: f32, cpu_ms: f32) {
        self.metrics.push_frame_with_gpu_timing(dt_ms, cpu_ms);
    }

    pub fn set_render_target(&mut self, scale: f32, width: u32, height: u32) {
        self.metrics.set_render_target(scale, width, height);
    }

    pub fn set_gpu_info(&mut self, backend: impl Into<String>, adapter_name: impl Into<String>) {
        self.metrics.set_gpu_info(backend, adapter_name);
    }

    pub fn set_gpu_info_from_wgpu(&mut self, info: &egui_wgpu::wgpu::AdapterInfo) {
        self.metrics.set_gpu_info_from_wgpu(info);
    }

    /// Renders a compact, non-blocking translucent HUD card in the top-left of
    /// `viewport_rect`. Uses `interactable(false)` so it never steals touch or
    /// mouse input from the viewport.
    pub fn draw(&self, ctx: &egui::Context, viewport_rect: egui::Rect) {
        let pos = viewport_rect.min + egui::vec2(8.0, 8.0);
        let mono = FontId::monospace(11.0);
        let muted = Color32::from_rgb(175, 182, 194);
        let bright = Color32::from_rgb(238, 242, 248);

        let fps = self.metrics.smoothed_fps();
        let inst_fps = self.metrics.instant_fps();
        let fps_color = if fps >= 55.0 {
            Color32::from_rgb(110, 224, 130)
        } else if fps >= 30.0 {
            Color32::from_rgb(245, 200, 85)
        } else {
            Color32::from_rgb(245, 110, 100)
        };

        let low_str = match self.metrics.one_percent_low_fps() {
            Some(low) => format!("{low:.1}"),
            None => "--".to_string(),
        };
        let frame_ms = self.metrics.avg_frame_ms();
        let cpu_ms = self.metrics.avg_cpu_ms();
        let gpu_str = format_gpu_ms(self.metrics.avg_gpu_ms());
        let scale_str = format_render_scale(
            self.metrics.render_scale(),
            self.metrics.render_target_px(),
        );
        let mem_str = match self.metrics.rss_mb() {
            Some(mb) => format!("{mb:.1} MB"),
            None => "N/A".to_string(),
        };

        egui::Area::new(egui::Id::new("rustudio_perf_overlay"))
            .order(egui::Order::Foreground)
            .interactable(false)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(16, 18, 22, 195))
                    .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 28)))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 2.0);
                        ui.label(
                            RichText::new(format!(
                                "FPS: {fps:.1} (inst {inst_fps:.1}) | 1% low: {low_str}"
                            ))
                            .font(mono.clone())
                            .color(fps_color),
                        );
                        ui.label(
                            RichText::new(format!(
                                "Frame: {frame_ms:.2} ms | CPU: {cpu_ms:.2} ms"
                            ))
                            .font(mono.clone())
                            .color(bright),
                        );
                        ui.label(
                            RichText::new(format!("GPU: {gpu_str}"))
                                .font(mono.clone())
                                .color(if self.metrics.avg_gpu_ms().is_some() { bright } else { muted }),
                        );
                        ui.label(
                            RichText::new(format!("3D Scale: {scale_str}"))
                                .font(mono.clone())
                                .color(bright),
                        );
                        let backend = self.metrics.backend();
                        let adapter = self.metrics.adapter_name();
                        if !backend.is_empty() || !adapter.is_empty() {
                            let gpu_label = match (backend.is_empty(), adapter.is_empty()) {
                                (false, false) => format!("{backend} - {adapter}"),
                                (false, true) => backend.to_string(),
                                (true, false) => adapter.to_string(),
                                (true, true) => String::new(),
                            };
                            ui.label(
                                RichText::new(gpu_label)
                                    .font(mono.clone())
                                    .color(muted),
                            );
                        }
                        ui.label(
                            RichText::new(format!("RAM: {mem_str}"))
                                .font(mono)
                                .color(muted),
                        );
                    });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_smoothed_and_instantaneous() {
        let mut m = PerfMetrics::new(120);
        assert_eq!(m.smoothed_fps(), 0.0);
        assert_eq!(m.instant_fps(), 0.0);
        assert_eq!(m.avg_frame_ms(), 0.0);

        // Push 10 frames at 16.0 ms (62.5 FPS) and 1 frame at 20.0 ms (50.0 FPS).
        for _ in 0..10 {
            m.push_frame(16.0, 4.0, None);
        }
        m.push_frame(20.0, 5.0, None);

        let expected_avg_ms = (10.0 * 16.0 + 20.0) / 11.0;
        assert!((m.avg_frame_ms() - expected_avg_ms).abs() < 1e-4);
        assert!((m.smoothed_fps() - (1000.0 / expected_avg_ms)).abs() < 1e-3);
        assert!((m.instant_fps() - 50.0).abs() < 1e-3);
    }

    #[test]
    fn one_percent_low_requires_min_samples_and_uses_p99() {
        let mut m = PerfMetrics::new(120);
        // Fewer than 10 samples -> None.
        for _ in 0..9 {
            m.push_frame(16.67, 5.0, None);
        }
        assert_eq!(m.one_percent_low_fps(), None);

        // Push 90 more normal frames (total 99 at 10.0 ms = 100 FPS) + 1 spike at 50.0 ms (20 FPS).
        m.clear();
        for _ in 0..99 {
            m.push_frame(10.0, 3.0, None);
        }
        m.push_frame(50.0, 12.0, None);
        // 100 samples sorted: indices 0..=98 are 10.0 ms, index 99 is 50.0 ms.
        // ceil(100 * 0.99) - 1 = 98 -> 10.0 ms -> 100.0 FPS.
        assert!((m.one_percent_low_fps().unwrap() - 100.0).abs() < 1e-3);

        // Now push 2 spikes out of 100 frames (indices 98 and 99 are 50.0 ms).
        m.clear();
        for _ in 0..98 {
            m.push_frame(10.0, 3.0, None);
        }
        m.push_frame(50.0, 12.0, None);
        m.push_frame(80.0, 15.0, None);
        // Index 98 is 50.0 ms -> 20.0 FPS.
        assert!((m.one_percent_low_fps().unwrap() - 20.0).abs() < 1e-3);
    }

    #[test]
    fn gpu_none_is_never_fabricated() {
        let mut overlay = PerfOverlay::new();
        for _ in 0..15 {
            overlay.record_frame(16.0, 4.5, None);
        }
        assert_eq!(overlay.metrics.avg_gpu_ms(), None);
        assert_eq!(overlay.metrics.last_gpu_ms(), None);
        assert_eq!(format_gpu_ms(overlay.metrics.avg_gpu_ms()), GPU_UNSUPPORTED_LABEL);
        assert_eq!(format_gpu_ms(None), "N/A (driver unsupported)");

        // When real GPU timings arrive, they are averaged accurately.
        overlay.record_frame(16.0, 4.5, Some(6.0));
        overlay.record_frame(16.0, 4.5, Some(8.0));
        assert!((overlay.metrics.avg_gpu_ms().unwrap() - 7.0).abs() < 1e-4);
        assert_eq!(overlay.metrics.last_gpu_ms(), Some(8.0));
        assert_eq!(format_gpu_ms(overlay.metrics.avg_gpu_ms()), "7.00 ms");
    }

    #[test]
    fn render_scale_formatting() {
        assert_eq!(format_render_scale(1.0, [1920, 1080]), "100% (1920x1080)");
        assert_eq!(format_render_scale(0.75, [1440, 810]), "75% (1440x810)");
        assert_eq!(format_render_scale(0.67, [1286, 724]), "67% (1286x724)");
        assert_eq!(format_render_scale(0.5, [0, 0]), "50%");
    }

    #[test]
    fn ring_buffer_evicts_oldest_samples() {
        let mut m = PerfMetrics::new(4);
        m.push_frame(100.0, 10.0, Some(50.0));
        for _ in 0..4 {
            m.push_frame(10.0, 2.0, None);
        }
        assert_eq!(m.sample_count(), 4);
        assert!((m.avg_frame_ms() - 10.0).abs() < 1e-4);
        assert_eq!(m.avg_gpu_ms(), None);
    }

    #[test]
    fn proc_statm_parsing() {
        // 25600 resident pages * 4096 bytes = 100.0 MiB
        let mb = parse_proc_statm_rss_mb("50000 25600 4000 100 0 20000 0\n", 4096).unwrap();
        assert!((mb - 100.0).abs() < 1e-4);
        assert_eq!(parse_proc_statm_rss_mb("", 4096), None);
        assert_eq!(parse_proc_statm_rss_mb("50000 25600", 0), None);
    }
}
