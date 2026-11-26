//! A line about the frames every few seconds (`log::info!`, so logcat on
//! Android: `adb logcat -s rustudio`; RUSTUDIO_PERF_LOG=1 elsewhere): how
//! far apart the frames came, how many of them followed the frame before
//! at once (the frame before asked for them; eframe draws only on request
//! or input, so the rest came after a pause and their interval is idle
//! time, not slowness), how much of a back-to-back frame was CPU work
//! (the editor's own on the main thread, eframe's layout and painting)
//! and how much the wait for the GPU and the display, the renderer's GPU
//! time per 3D frame where the device has timestamps, the editor's
//! sections (src/bench.rs) that took the most, and who asked for the
//! frames. The way to tell a GPU-bound frame from a CPU-bound one and
//! from a paced one on a phone, where there's no profiler at hand.
//!
//! An idle editor logs nothing: no input, no frames.

use std::time::Duration;
use web_time::Instant;

const PERIOD: Duration = Duration::from_secs(5);

/// The build this is: a debug APK / binary runs the editor at opt-level 1
/// with debug assertions and overflow checks in every crate, several times
/// slower than a release build (and a phone shows it first).
pub fn build() -> String {
    format!(
        "{} build, {} {}",
        if cfg!(debug_assertions) { "debug" } else { "release" },
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

/// logcat on Android (the `log` crate is an Android dependency), stderr
/// elsewhere.
fn say(line: String) {
    #[cfg(target_os = "android")]
    log::info!("{line}");
    #[cfg(not(target_os = "android"))]
    eprintln!("{line}");
}

#[derive(Default)]
pub struct PerfLog {
    since: Option<Instant>,
    last_frame: Option<Instant>,
    /// Frame intervals (ms), the editor's own work in each and eframe's
    /// whole CPU time for each (ms).
    intervals: Vec<f32>,
    work: Vec<f32>,
    cpu: Vec<f32>,
    /// The intervals and CPU times of the frames that followed an
    /// immediate repaint request (the frame before asked for them, so
    /// the loop never slept): their interval beyond the CPU time is what
    /// the display and the GPU made the frame wait, where a frame after a
    /// pause (input, a timer) just shows how long the editor was idle.
    b2b_intervals: Vec<f32>,
    b2b_cpu: Vec<f32>,
    /// (ms, calls) per section over the period.
    sections: Vec<(&'static str, f64, u64)>,
    /// The renderer's GPU totals at the last line (render::GpuTimer).
    gpu_last: ([f64; GPU_SECTIONS + 1], u32),
    /// Who asked for the frames: (file, line) of the `request_repaint`
    /// calls of the frame before each, with how many frames each asked
    /// for; `unasked` the frames nobody asked for (input, the first ones).
    causes: Vec<((&'static str, u32), u32)>,
    unasked: u32,
    said_device: bool,
}

/// The GPU timer's sections (render::GpuTimer::SECTIONS).
const GPU_SECTIONS: usize = crate::render::GpuTimer::SECTIONS;

/// How many perf lines have been said: the GPU bisection (render.rs
/// `bisect`) changes what it leaves out of the scene with each line, so
/// every line reports one configuration.
static PERIODS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The perf lines said so far.
pub fn periods() -> u32 {
    PERIODS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the log is wanted: always on a phone, by request elsewhere.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| cfg!(target_os = "android") || std::env::var_os("RUSTUDIO_PERF_LOG").is_some())
}

impl PerfLog {
    /// Once per frame: the repaint causes egui kept from the frame before
    /// (`egui::Context::repaint_causes`), each call site counted once a
    /// frame. The line names the top ones: which code keeps an idle editor
    /// drawing (a spinner, a poll, a stuck pointer) shows there.
    pub fn repaint_causes(&mut self, causes: &[egui::RepaintCause]) {
        if causes.is_empty() {
            self.unasked += 1;
            return;
        }
        let mut seen: Vec<(&'static str, u32)> = Vec::with_capacity(causes.len());
        for c in causes {
            let key = (c.file, c.line);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            match self.causes.iter_mut().find(|e| e.0 == key) {
                Some(e) => e.1 += 1,
                None => self.causes.push((key, 1)),
            }
        }
    }

    /// Once per frame, at its end: `work_ms` the editor's own time this
    /// frame, `cpu_ms` eframe's for the previous one (`Frame::info().cpu_usage`),
    /// `back_to_back` whether the frame before asked for this one at once
    /// (`egui::Context::requested_repaint_last_pass`), `gpu` the
    /// renderer's GPU totals so far (`Renderer::gpu_totals`), `extra` the
    /// line's tail (view sizes, counters), `device` the GPU (said once).
    pub fn frame(
        &mut self,
        work_ms: f32,
        cpu_ms: Option<f32>,
        back_to_back: bool,
        gpu: Option<([f64; GPU_SECTIONS + 1], u32)>,
        extra: impl FnOnce() -> String,
        device: impl FnOnce() -> Option<String>,
    ) {
        let now = Instant::now();
        let since = *self.since.get_or_insert(now);
        if let Some(last) = self.last_frame {
            let interval = now.duration_since(last).as_secs_f32() * 1000.0;
            self.intervals.push(interval);
            self.work.push(work_ms);
            if let Some(c) = cpu_ms {
                self.cpu.push(c * 1000.0);
            }
            if back_to_back {
                self.b2b_intervals.push(interval);
                if let Some(c) = cpu_ms {
                    self.b2b_cpu.push(c * 1000.0);
                }
            }
        }
        self.last_frame = Some(now);
        // A frame's sections are gathered whether or not a bench is on
        // (`--bench` takes them per frame itself; the two don't run together).
        for (name, ms, calls) in rustudio_core::bench::drain_sections() {
            match self.sections.iter_mut().find(|e| e.0 == name) {
                Some(e) => {
                    e.1 += ms;
                    e.2 += calls;
                }
                None => self.sections.push((name, ms, calls)),
            }
        }
        let elapsed = now.duration_since(since);
        if elapsed < PERIOD {
            return;
        }
        if !self.said_device {
            self.said_device = true;
            if let Some(d) = device() {
                say(format!("perf: {d}; {}", build()));
            }
        }
        let n = self.intervals.len();
        if n == 0 {
            self.since = Some(now);
            return;
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
        let mut sorted = self.intervals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
        let worst = sorted[sorted.len() - 1];
        let interval = mean(&self.intervals);
        let work = mean(&self.work);
        let cpu = if self.cpu.is_empty() { work } else { mean(&self.cpu) };
        // The wait: back-to-back frames only, where the loop never slept.
        let b2b = self.b2b_intervals.len();
        let paced = if b2b == 0 {
            "no frames back to back".to_string()
        } else {
            let b2b_interval = mean(&self.b2b_intervals);
            let b2b_cpu = if self.b2b_cpu.is_empty() { work } else { mean(&self.b2b_cpu) };
            format!(
                "{b2b} back to back {b2b_interval:.1} ms apart = {b2b_cpu:.1} ms CPU + {:.1} ms GPU/display wait",
                (b2b_interval - b2b_cpu).max(0.0)
            )
        };
        let mut top: Vec<(&'static str, f64)> = self.sections.iter().map(|(name, ms, _)| (*name, *ms / n as f64)).collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<String> = top.iter().take(8).filter(|(_, ms)| *ms >= 0.05).map(|(name, ms)| format!("{name} {ms:.1}")).collect();
        // The 3D frames' GPU time since the last line, per measured frame.
        let gpu_line = match gpu {
            None => "no GPU timestamps".to_string(),
            Some((sum, frames)) => {
                let (last_sum, last_n) = self.gpu_last;
                self.gpu_last = (sum, frames);
                let k = frames.saturating_sub(last_n);
                if k == 0 {
                    "3D GPU: no frame measured".to_string()
                } else {
                    let per: Vec<f64> = (0..=GPU_SECTIONS).map(|i| (sum[i] - last_sum[i]) / k as f64).collect();
                    let parts: Vec<String> = crate::render::GpuTimer::NAMES
                        .iter()
                        .zip(&per)
                        .map(|(name, ms)| format!("{name} {ms:.1}"))
                        .collect();
                    format!("3D GPU {:.1} ms per drawn frame = {} ({k} measured)", per[GPU_SECTIONS], parts.join(" + "))
                }
            }
        };
        // Who asked for the frames (the file names without their directories).
        self.causes.sort_by(|a, b| b.1.cmp(&a.1));
        let mut asked: Vec<String> = self
            .causes
            .iter()
            .take(4)
            .map(|((file, line), k)| format!("{}:{line} x{k}", file.rsplit(['/', '\\']).next().unwrap_or(file)))
            .collect();
        if self.unasked > 0 {
            asked.push(format!("input/other x{}", self.unasked));
        }
        say(format!(
            "perf: {n} frames in {:.1} s: {interval:.1} ms apart (p95 {p95:.1}, worst {worst:.0}), {paced}, {} after a pause; editor {work:.1} ms + egui {:.1} ms on the CPU; {gpu_line}; {}; sections (ms/frame): {}; frames asked by: {}",
            elapsed.as_secs_f32(),
            n - b2b,
            (cpu - work).max(0.0),
            extra(),
            if top.is_empty() { "-".to_string() } else { top.join(", ") },
            if asked.is_empty() { "-".to_string() } else { asked.join(", ") },
        ));
        PERIODS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.since = Some(now);
        self.intervals.clear();
        self.work.clear();
        self.cpu.clear();
        self.b2b_intervals.clear();
        self.b2b_cpu.clear();
        self.sections.clear();
        self.causes.clear();
        self.unasked = 0;
    }
}
