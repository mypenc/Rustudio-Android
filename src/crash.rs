//! Crash detection, diagnostics, and startup recovery for Rustudio (Android + desktop).
//!
//! - Captures Rust panics on any thread to `<crash_dir>/pending_crash.txt` (with
//!   backtrace, device/GPU/memory info, recent Studio logs, and best-effort Android logcat).
//! - Installs async-signal-safe Unix signal handlers (`SIGSEGV`, `SIGABRT`, `SIGBUS`,
//!   `SIGFPE`, `SIGILL`, `SIGTRAP`) that write `<crash_dir>/pending_signal_crash.txt`
//!   using only `open`/`write`/`close` and a lock-free pre-formatted header, then restore
//!   `SIG_DFL` and re-raise so Android cleanly terminates and captures a tombstone.
//! - Detects pending crashes on startup (`load_pending_crash` / `take_pending_crash`),
//!   enriching native signal crashes with fresh `logcat -d -t 500` and device details.
//! - Archives dismissed crashes into `<crash_dir>/archive/` via `dismiss_pending_crash()`.

use std::backtrace::Backtrace;
use std::collections::VecDeque;
use std::ffi::CString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Maximum number of recent Studio log / Output lines kept in memory.
const MAX_RECENT_LOGS: usize = 120;

/// File name for a pending Rust panic crash report.
const PENDING_PANIC_FILE: &str = "pending_crash.txt";

/// File name for a pending native Unix signal crash report.
const PENDING_SIGNAL_FILE: &str = "pending_signal_crash.txt";

/// Marker appended once a signal crash report has been enriched at startup.
const STARTUP_ENRICHMENT_MARKER: &str = "=== Startup Logcat & Device Enrichment ===";

/// Size of the pre-formatted static header buffer read by the async-signal-safe handler.
const SIGNAL_HEADER_CAP: usize = 4096;

/// Where crash reports live.
///
/// On Android: `<filesDir>/crashes` (persistent app data directory).
/// On desktop: `<cache>/rustudio/crashes`.
pub fn dir() -> PathBuf {
    #[cfg(target_os = "android")]
    {
        crate::android::files_dir().join("crashes")
    }
    #[cfg(not(target_os = "android"))]
    {
        rustudio_core::paths::cache().join("rustudio").join("crashes")
    }
}

/// Kind of crash detected from a previous session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashKind {
    Panic,
    Signal,
}

impl CrashKind {
    pub fn label(self) -> &'static str {
        match self {
            CrashKind::Panic => "Rust Panic",
            CrashKind::Signal => "Native Signal Crash",
        }
    }
}

/// Parsed crash report loaded on startup for display and export.
#[derive(Clone, Debug)]
pub struct CrashReport {
    pub kind: CrashKind,
    pub title: String,
    pub timestamp: String,
    pub summary: String,
    pub full_text: String,
    pub source_path: PathBuf,
}

/// Runtime diagnostic snapshot included in crash reports and live diagnostics.
#[derive(Clone, Debug)]
pub struct CrashContext {
    pub app_version: String,
    pub git_commit: String,
    pub target_os: String,
    pub target_arch: String,
    pub build_profile: String,
    pub device_info: DeviceInfo,
    pub gpu_info: String,
    pub recent_logs: VecDeque<String>,
}

/// Android / host device properties snapshot.
#[derive(Clone, Debug, Default)]
pub struct DeviceInfo {
    pub manufacturer: String,
    pub model: String,
    pub android_release: String,
    pub android_sdk: String,
    pub cpu_abi: String,
}

impl DeviceInfo {
    fn collect() -> Self {
        #[cfg(target_os = "android")]
        {
            Self {
                manufacturer: getprop("ro.product.manufacturer"),
                model: getprop("ro.product.model"),
                android_release: getprop("ro.build.version.release"),
                android_sdk: getprop("ro.build.version.sdk"),
                cpu_abi: getprop("ro.product.cpu.abi"),
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            Self {
                manufacturer: String::new(),
                model: String::new(),
                android_release: String::new(),
                android_sdk: String::new(),
                cpu_abi: std::env::consts::ARCH.to_string(),
            }
        }
    }

    fn format_lines(&self) -> String {
        #[cfg(target_os = "android")]
        {
            format!(
                "Manufacturer: {}\nModel: {}\nAndroid Release: {} (SDK {})\nCPU ABI: {}",
                non_empty(&self.manufacturer, "unknown"),
                non_empty(&self.model, "unknown"),
                non_empty(&self.android_release, "unknown"),
                non_empty(&self.android_sdk, "unknown"),
                non_empty(&self.cpu_abi, std::env::consts::ARCH),
            )
        }
        #[cfg(not(target_os = "android"))]
        {
            format!(
                "Platform: {}/{}\nCPU ABI: {}",
                std::env::consts::OS,
                std::env::consts::ARCH,
                non_empty(&self.cpu_abi, std::env::consts::ARCH),
            )
        }
    }
}

fn non_empty<'a>(s: &'a str, fallback: &'a str) -> &'a str {
    let trimmed = s.trim();
    if trimmed.is_empty() { fallback } else { trimmed }
}

#[cfg(target_os = "android")]
fn getprop(key: &str) -> String {
    let fast = crate::android::system_property(key);
    if !fast.trim().is_empty() {
        return fast.trim().to_string();
    }
    std::process::Command::new("getprop")
        .arg(key)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Reads key memory lines from `/proc/meminfo` (Android & Linux).
fn read_meminfo_summary() -> String {
    let Ok(raw) = std::fs::read_to_string("/proc/meminfo") else {
        return "Memory info unavailable".to_string();
    };
    let wanted = ["MemTotal:", "MemFree:", "MemAvailable:", "SwapTotal:", "SwapFree:"];
    let lines: Vec<&str> = raw
        .lines()
        .filter(|l| wanted.iter().any(|k| l.starts_with(k)))
        .collect();
    if lines.is_empty() {
        "Memory info unavailable".to_string()
    } else {
        lines.join("\n")
    }
}

/// Best-effort `logcat -d -t 500` capture on Android.
fn capture_logcat(max_lines: usize) -> String {
    #[cfg(target_os = "android")]
    {
        let count = max_lines.to_string();
        match std::process::Command::new("logcat")
            .args(["-d", "-t", &count])
            .output()
        {
            Ok(out) => {
                let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
                if text.trim().is_empty() && !out.stderr.is_empty() {
                    text = String::from_utf8_lossy(&out.stderr).into_owned();
                }
                if text.trim().is_empty() {
                    "(logcat returned no lines)".to_string()
                } else {
                    text
                }
            }
            Err(e) => format!("(logcat unavailable: {e})"),
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = max_lines;
        "(logcat only available on Android)".to_string()
    }
}

impl CrashContext {
    fn new_default() -> Self {
        Self {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: option_env!("RUSTUDIO_GIT").unwrap_or("unknown").to_string(),
            target_os: std::env::consts::OS.to_string(),
            target_arch: std::env::consts::ARCH.to_string(),
            build_profile: if cfg!(debug_assertions) {
                "debug".to_string()
            } else {
                "release".to_string()
            },
            device_info: DeviceInfo::collect(),
            gpu_info: "Unknown (not yet initialized)".to_string(),
            recent_logs: VecDeque::with_capacity(MAX_RECENT_LOGS),
        }
    }

    fn format_static_header(&self) -> String {
        format!(
            "App: Rustudio {} ({})\nTarget: {}/{} ({})\nPID: {}\nGPU: {}\n{}\n",
            self.app_version,
            self.git_commit,
            self.target_os,
            self.target_arch,
            self.build_profile,
            (if cfg!(target_arch = "wasm32") { 1u32 } else { std::process::id() }),
            self.gpu_info,
            self.device_info.format_lines(),
        )
    }
}

static CONTEXT: OnceLock<Mutex<CrashContext>> = OnceLock::new();

fn context_mutex() -> &'static Mutex<CrashContext> {
    CONTEXT.get_or_init(|| Mutex::new(CrashContext::new_default()))
}

/// Lock-free double-buffered static header for the Unix signal handler.
///
/// Writers update the inactive slot under `CONTEXT` mutex and atomically swap
/// `ACTIVE_HEADER_SLOT` with `Release` ordering. The signal handler loads
/// `ACTIVE_HEADER_SLOT` with `Acquire` ordering and reads that slot without locking.
#[cfg(unix)]
struct SignalHeaderSlot {
    len: AtomicUsize,
    buf: std::cell::UnsafeCell<[u8; SIGNAL_HEADER_CAP]>,
}

#[cfg(unix)]
unsafe impl Sync for SignalHeaderSlot {}

#[cfg(unix)]
impl SignalHeaderSlot {
    const fn new() -> Self {
        Self {
            len: AtomicUsize::new(0),
            buf: std::cell::UnsafeCell::new([0u8; SIGNAL_HEADER_CAP]),
        }
    }
}

#[cfg(unix)]
static SIGNAL_HEADERS: [SignalHeaderSlot; 2] = [SignalHeaderSlot::new(), SignalHeaderSlot::new()];
#[cfg(unix)]
static ACTIVE_HEADER_SLOT: AtomicUsize = AtomicUsize::new(0);

#[cfg(unix)]
fn refresh_signal_header(header_text: &str) {
    let cur = ACTIVE_HEADER_SLOT.load(Ordering::Relaxed);
    let next = 1 - cur;
    let bytes = header_text.as_bytes();
    let n = bytes.len().min(SIGNAL_HEADER_CAP);
    // SAFETY: only called while holding `CONTEXT` mutex (serialized writers),
    // and `next` is the inactive slot not currently published in `ACTIVE_HEADER_SLOT`.
    unsafe {
        let dst = &mut *SIGNAL_HEADERS[next].buf.get();
        dst[..n].copy_from_slice(&bytes[..n]);
    }
    SIGNAL_HEADERS[next].len.store(n, Ordering::Release);
    ACTIVE_HEADER_SLOT.store(next, Ordering::Release);
}

#[cfg(not(unix))]
fn refresh_signal_header(_header_text: &str) {}

/// Pre-computed CString path to `<crash_dir>/pending_signal_crash.txt` so the
/// signal handler never allocates.
static SIGNAL_PATH: OnceLock<CString> = OnceLock::new();
/// Pre-computed CString path in public shared storage (`Documents/Rustudio/` or
/// `Download/Rustudio/`) so even a black-screen native crash leaves a user-visible `.txt`.
static PUBLIC_SIGNAL_PATH: OnceLock<CString> = OnceLock::new();

/// Best-effort automatic export of a crash report `.txt` directly into
/// `Documents/Rustudio/` and `Download/Rustudio/` (via direct write + Android
/// MediaStore.Downloads) so users can recover logs even after a black-screen crash.
fn auto_export_public_crash(file_name: &str, body: &str) {
    #[cfg(target_os = "android")]
    {
        for dir in [
            "/storage/emulated/0/Documents/Rustudio",
            "/storage/emulated/0/Download/Rustudio",
        ] {
            if std::fs::create_dir_all(dir).is_ok() {
                let p = Path::new(dir).join(file_name);
                if std::fs::write(&p, body.as_bytes()).is_ok() {
                    break;
                }
            }
        }
        let _ = crate::android::crash_export::save_to_downloads_android(file_name, body.as_bytes());
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (file_name, body);
    }
}

/// Ensures the first fatal event wins (a panic that later aborts will not
/// overwrite the panic report with an abort tombstone).
static WRITTEN: AtomicBool = AtomicBool::new(false);

/// Tracks whether `take_pending_crash()` has already handed the pending crash
/// to the UI during this process lifetime.
static PENDING_TAKEN: AtomicBool = AtomicBool::new(false);

/// Last report path written by this process.
static LAST_REPORT: Mutex<Option<PathBuf>> = Mutex::new(None);

thread_local! {
    /// Inside `recoverable`: a panic here is caught and handled, not a crash.
    static RECOVERING: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Run `f`, catching a panic in it (the caller falls back). Such a panic is
/// expected, so it writes no crash report.
pub fn recoverable<R>(f: impl FnOnce() -> R) -> rt_thread::Result<R> {
    RECOVERING.with(|r| r.set(r.get() + 1));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    RECOVERING.with(|r| r.set(r.get() - 1));
    out
}

/// Updates the GPU / Vulkan adapter info included in crash reports and the
/// pre-formatted signal handler header.
pub fn set_gpu_info(info: String) {
    let mut ctx = context_mutex().lock().unwrap_or_else(|e| e.into_inner());
    ctx.gpu_info = info;
    let hdr = ctx.format_static_header();
    refresh_signal_header(&hdr);
}

/// Appends a Studio Output / log line into the in-memory ring buffer (last ~120 lines).
pub fn record_log(line: &str) {
    let trimmed = line.trim_end();
    if trimmed.is_empty() {
        return;
    }
    if let Ok(mut ctx) = context_mutex().try_lock() {
        if ctx.recent_logs.len() >= MAX_RECENT_LOGS {
            ctx.recent_logs.pop_front();
        }
        ctx.recent_logs.push_back(trimmed.to_string());
    }
}

/// Alias for recording a diagnostic note into the crash log ring buffer.
pub fn note(line: impl AsRef<str>) {
    record_log(line.as_ref());
}

fn stamp_filename() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

fn stamp_human() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S %z").to_string()
}

/// Command-line arguments with anything that looks like a credential removed.
fn args_line() -> String {
    std::env::args()
        .map(|a| {
            if a.contains("ROBLOSECURITY") || a.contains("WARNING:-DO-NOT-SHARE") || a.len() > 200 {
                "<redacted>".to_string()
            } else {
                a
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns a snapshot of the current `CrashContext` (never panics on poisoned mutex).
pub fn snapshot_context() -> CrashContext {
    context_mutex().lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Builds a full live diagnostic report (app/device/GPU/memory/Studio logs/logcat)
/// that users can export anytime.
pub fn build_live_diagnostic_report() -> String {
    let ctx = snapshot_context();
    let mut out = String::with_capacity(8192);
    out.push_str("=== Rustudio Live Diagnostic Report ===\n");
    out.push_str(&format!("Timestamp: {}\n", stamp_human()));
    out.push_str(&ctx.format_static_header());
    out.push_str(&format!("Args: {}\n", args_line()));
    out.push_str("\n=== Memory (/proc/meminfo) ===\n");
    out.push_str(&read_meminfo_summary());
    out.push_str("\n\n=== Recent Studio Logs ===\n");
    if ctx.recent_logs.is_empty() {
        out.push_str("(no recent Studio logs recorded)\n");
    } else {
        for line in &ctx.recent_logs {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str("\n=== Android Logcat (last 500 lines) ===\n");
    out.push_str(&capture_logcat(500));
    out.push('\n');
    out
}

/// The report path this process wrote, if any.
pub fn last_report() -> Option<PathBuf> {
    LAST_REPORT.lock().ok().and_then(|l| l.clone())
}

fn write_pending_panic_report(body: &str) -> Option<PathBuf> {
    let d = dir();
    std::fs::create_dir_all(&d).ok()?;
    let pending = d.join(PENDING_PANIC_FILE);
    let mut f = std::fs::File::create(&pending).ok()?;
    f.write_all(body.as_bytes()).ok()?;
    let _ = f.sync_all();
    let pub_name = format!("rustudio-crash-{}.txt", (std::time::UNIX_EPOCH + web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).unwrap_or_default()).duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0));
    auto_export_public_crash(&pub_name, body);
    if let Ok(mut l) = LAST_REPORT.lock() {
        *l = Some(pending.clone());
    }
    Some(pending)
}

/// Install the Rust panic hook and Unix fatal-signal handlers.
/// Call as early as possible during startup (after Android dirs are initialized on Android).
pub fn install() {
    let crash_dir = dir();
    let _ = std::fs::create_dir_all(&crash_dir);

    // Initialize CrashContext and pre-format the static signal header.
    {
        let ctx = context_mutex().lock().unwrap_or_else(|e| e.into_inner());
        let hdr = ctx.format_static_header();
        refresh_signal_header(&hdr);
    }

    let sig_path = crash_dir.join(PENDING_SIGNAL_FILE);
    #[cfg(target_os = "android")]
    {
        for dir in [
            "/storage/emulated/0/Documents/Rustudio",
            "/storage/emulated/0/Download/Rustudio",
        ] {
            let _ = std::fs::create_dir_all(dir);
            let p = Path::new(dir).join("rustudio-crash-signal.txt");
            if let Ok(c) = CString::new(p.to_string_lossy().as_bytes()) {
                let _ = PUBLIC_SIGNAL_PATH.set(c);
                break;
            }
        }
    }
    let _ = SIGNAL_PATH.set(CString::new(sig_path.to_string_lossy().as_bytes()).unwrap_or_default());

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if RECOVERING.with(|r| r.get()) > 0 {
            return;
        }
        if !WRITTEN.swap(true, Ordering::SeqCst) {
            let thread = rt_thread::current();
            let thread_name = thread.name().unwrap_or("<unnamed>").to_string();
            let msg = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".into());
            let loc = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "<unknown location>".to_string());
            let bt = Backtrace::force_capture();
            let ctx = snapshot_context();

            let mut body = String::with_capacity(16 * 1024);
            body.push_str("=== Rustudio Crash Report ===\n");
            body.push_str("Crash Type: Rust Panic\n");
            body.push_str(&format!("Timestamp: {}\n", stamp_human()));
            body.push_str(&ctx.format_static_header());
            body.push_str(&format!("Args: {}\n", args_line()));
            body.push_str(&format!("\n=== Panic Details ===\nThread: {thread_name}\nLocation: {loc}\nMessage: {msg}\n"));
            body.push_str(&format!("\n=== Backtrace ===\n{bt}\n"));
            body.push_str("\n=== Memory (/proc/meminfo) ===\n");
            body.push_str(&read_meminfo_summary());
            body.push_str("\n\n=== Recent Studio Logs ===\n");
            if ctx.recent_logs.is_empty() {
                body.push_str("(no recent Studio logs recorded)\n");
            } else {
                for line in &ctx.recent_logs {
                    body.push_str(line);
                    body.push('\n');
                }
            }
            body.push_str("\n=== Android Logcat (last 500 lines) ===\n");
            body.push_str(&capture_logcat(500));
            body.push('\n');

            if let Some(p) = write_pending_panic_report(&body) {
                eprintln!("rustudio: crash report written to {}", p.display());
            }
        }
        default_hook(info);
    }));

    #[cfg(unix)]
    unsafe {
        // Alternate signal stack so stack overflows can still run the handler.
        let size = 64 * 1024;
        let stack = libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if stack != libc::MAP_FAILED {
            let ss = libc::stack_t {
                ss_sp: stack,
                ss_flags: 0,
                ss_size: size,
            };
            libc::sigaltstack(&ss, std::ptr::null_mut());
        }

        for sig in [
            libc::SIGSEGV,
            libc::SIGABRT,
            libc::SIGBUS,
            libc::SIGFPE,
            libc::SIGILL,
            libc::SIGTRAP,
        ] {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_signal as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESETHAND;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(sig, &sa, std::ptr::null_mut());
        }
    }
}

#[cfg(unix)]
fn sig_name(sig: libc::c_int) -> &'static [u8] {
    match sig {
        libc::SIGSEGV => b"SIGSEGV (Segmentation fault)",
        libc::SIGABRT => b"SIGABRT (Abort)",
        libc::SIGBUS => b"SIGBUS (Bus error)",
        libc::SIGFPE => b"SIGFPE (Floating-point exception)",
        libc::SIGILL => b"SIGILL (Illegal instruction)",
        libc::SIGTRAP => b"SIGTRAP (Trap instruction)",
        _ => b"UNKNOWN_SIGNAL",
    }
}

/// Async-signal-safe decimal formatter for `i32` into a stack buffer.
#[cfg(unix)]
fn fmt_i32_stack(mut v: i32, buf: &mut [u8; 16]) -> &[u8] {
    let neg = v < 0;
    let mut u: u32 = if neg { v.wrapping_neg() as u32 } else { v as u32 };
    let mut idx = buf.len();
    if u == 0 {
        idx -= 1;
        buf[idx] = b'0';
    } else {
        while u > 0 {
            idx -= 1;
            buf[idx] = b'0' + (u % 10) as u8;
            u /= 10;
        }
    }
    if neg {
        idx -= 1;
        buf[idx] = b'-';
    }
    &buf[idx..]
}

/// Async-signal-safe hex formatter for a pointer address (`0x...`) into a stack buffer.
#[cfg(unix)]
fn fmt_hex_usize_stack(mut v: usize, buf: &mut [u8; 20]) -> &[u8] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut idx = buf.len();
    if v == 0 {
        idx -= 1;
        buf[idx] = b'0';
    } else {
        while v > 0 {
            idx -= 1;
            buf[idx] = HEX[v & 0xf];
            v >>= 4;
        }
    }
    idx -= 1;
    buf[idx] = b'x';
    idx -= 1;
    buf[idx] = b'0';
    &buf[idx..]
}

#[cfg(unix)]
unsafe fn write_all_fd(fd: libc::c_int, bytes: &[u8]) {
    let mut off = 0;
    while off < bytes.len() {
        let n = libc::write(fd, bytes[off..].as_ptr().cast(), bytes.len() - off);
        if n <= 0 {
            break;
        }
        off += n as usize;
    }
}

#[cfg(unix)]
unsafe fn extract_si_addr(info: *const libc::siginfo_t) -> usize {
    if info.is_null() {
        return 0;
    }
    #[cfg(target_os = "android")]
    {
        (*info).si_addr() as usize
    }
    #[cfg(target_os = "linux")]
    {
        // Linux kernel ABI: siginfo_t starts with (si_signo: i32, si_errno: i32, si_code: i32),
        // followed by `_sifields._sigfault.si_addr: *mut c_void` (aligned to pointer size).
        #[repr(C)]
        struct LinuxSigFault {
            _si_signo: libc::c_int,
            _si_errno: libc::c_int,
            _si_code: libc::c_int,
            si_addr: *mut libc::c_void,
        }
        (*(info.cast::<LinuxSigFault>())).si_addr as usize
    }
    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    {
        (*info).si_addr as usize
    }
}

#[cfg(unix)]
extern "C" fn on_signal(sig: libc::c_int, info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
    // SAFETY: Strictly async-signal-safe operations only (`open`, `write`, `close`,
    // atomic loads, stack buffers, `sigaction`, `raise`).
    unsafe {
        if !WRITTEN.swap(true, Ordering::SeqCst) {
            for p in [SIGNAL_PATH.get(), PUBLIC_SIGNAL_PATH.get()].into_iter().flatten() {
                let fd = libc::open(
                    p.as_ptr(),
                    libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
                    0o644,
                );
                if fd >= 0 {
                    write_all_fd(fd, b"=== Rustudio Native Signal Crash Tombstone ===\n");
                    write_all_fd(fd, b"Crash Type: Native Signal\n");
                    write_all_fd(fd, b"Signal: ");
                    write_all_fd(fd, sig_name(sig));
                    write_all_fd(fd, b" (");
                    let mut num_buf = [0u8; 16];
                    write_all_fd(fd, fmt_i32_stack(sig, &mut num_buf));
                    write_all_fd(fd, b")\n");

                    let fault_addr = extract_si_addr(info);
                    let mut hex_buf = [0u8; 20];
                    write_all_fd(fd, b"Fault Address (si_addr): ");
                    write_all_fd(fd, fmt_hex_usize_stack(fault_addr, &mut hex_buf));
                    write_all_fd(fd, b"\n\n=== Pre-Crash Device & App Header ===\n");

                    let slot_idx = ACTIVE_HEADER_SLOT.load(Ordering::Acquire) & 1;
                    let slot = &SIGNAL_HEADERS[slot_idx];
                    let hdr_len = slot.len.load(Ordering::Acquire).min(SIGNAL_HEADER_CAP);
                    let hdr_bytes = std::slice::from_raw_parts(slot.buf.get() as *const u8, hdr_len);
                    write_all_fd(fd, hdr_bytes);
                    write_all_fd(fd, b"\n");

                    libc::close(fd);
                }
            }
        }

        // Restore SIG_DFL and re-raise so Android / OS cleanly terminates and writes its tombstone.
        let mut dfl: libc::sigaction = std::mem::zeroed();
        dfl.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut dfl.sa_mask);
        libc::sigaction(sig, &dfl, std::ptr::null_mut());
        libc::raise(sig);
    }
}

/// Checks for a pending crash report from the previous session without consuming
/// the one-shot startup flag.
///
/// If a native signal crash tombstone (`pending_signal_crash.txt`) is found and
/// has not yet been enriched, enriches it with fresh `logcat -d -t 500` and
/// current device / `/proc/meminfo` details and persists the enriched report.
pub fn load_pending_crash() -> Option<CrashReport> {
    let crash_dir = dir();
    let panic_path = crash_dir.join(PENDING_PANIC_FILE);
    if panic_path.is_file() {
        let text = std::fs::read_to_string(&panic_path).ok()?;
        if !text.trim().is_empty() {
            return Some(parse_crash_report(CrashKind::Panic, panic_path, text));
        }
    }

    let signal_path = crash_dir.join(PENDING_SIGNAL_FILE);
    if signal_path.is_file() {
        let mut text = std::fs::read_to_string(&signal_path).ok()?;
        if !text.trim().is_empty() {
            if !text.contains(STARTUP_ENRICHMENT_MARKER) {
                let ctx = snapshot_context();
                let enrichment = format!(
                    "\n{STARTUP_ENRICHMENT_MARKER}\nEnriched At: {}\n{}\n\n=== Memory (/proc/meminfo) ===\n{}\n\n=== Android Logcat (last 500 lines) ===\n{}\n",
                    stamp_human(),
                    ctx.device_info.format_lines(),
                    read_meminfo_summary(),
                    capture_logcat(500),
                );
                text.push_str(&enrichment);
                let _ = std::fs::write(&signal_path, &text);
            }
            return Some(parse_crash_report(CrashKind::Signal, signal_path, text));
        }
    }

    None
}

/// Loads any pending crash from the previous session once per process launch.
///
/// The underlying file remains on disk until [`dismiss_pending_crash`] is called
/// so a crash isn't lost if the user closes the app before exporting.
pub fn take_pending_crash() -> Option<CrashReport> {
    if PENDING_TAKEN.swap(true, Ordering::SeqCst) {
        return None;
    }
    load_pending_crash()
}

/// Moves any pending crash files into `<crash_dir>/archive/` so the crash dialog
/// will not pop up again on the next launch.
pub fn dismiss_pending_crash() {
    let crash_dir = dir();
    let archive_dir = crash_dir.join("archive");
    let _ = std::fs::create_dir_all(&archive_dir);
    let ts = stamp_filename();

    for name in [PENDING_PANIC_FILE, PENDING_SIGNAL_FILE] {
        let src = crash_dir.join(name);
        if src.is_file() {
            let dst = archive_dir.join(format!("{ts}-{name}"));
            if std::fs::rename(&src, &dst).is_err() {
                // Fallback if rename fails across filesystems/permissions.
                let _ = std::fs::copy(&src, &dst);
                let _ = std::fs::remove_file(&src);
            }
        }
    }
}

fn parse_crash_report(kind: CrashKind, source_path: PathBuf, full_text: String) -> CrashReport {
    let timestamp = full_text
        .lines()
        .find_map(|l| l.strip_prefix("Timestamp: ").or_else(|| l.strip_prefix("Enriched At: ")))
        .map(|s| s.trim().to_string())
        .or_else(|| file_mtime_string(&source_path))
        .unwrap_or_else(|| "Unknown time".to_string());

    let title = match kind {
        CrashKind::Panic => "Rust Panic".to_string(),
        CrashKind::Signal => {
            let sig = full_text
                .lines()
                .find_map(|l| l.strip_prefix("Signal: "))
                .map(|s| s.trim())
                .unwrap_or("Native Signal");
            format!("Native Crash ({sig})")
        }
    };

    let summary = match kind {
        CrashKind::Panic => full_text
            .lines()
            .find_map(|l| l.strip_prefix("Message: "))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Unexpected panic in previous session.".to_string()),
        CrashKind::Signal => full_text
            .lines()
            .find_map(|l| l.strip_prefix("Signal: "))
            .map(|s| format!("Fatal signal: {}", s.trim()))
            .unwrap_or_else(|| "Fatal native signal in previous session.".to_string()),
    };

    CrashReport {
        kind,
        title,
        timestamp,
        summary,
        full_text,
        source_path,
    }
}

fn file_mtime_string(path: &Path) -> Option<String> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    let dt: chrono::DateTime<chrono::Local> = modified.into();
    Some(dt.format("%Y-%m-%d %H:%M:%S %z").to_string())
}

/// Legacy compatibility helper: returns summary lines for any unseen crash reports
/// (used by Output panel startup logging).
pub fn unseen_reports() -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(report) = load_pending_crash() {
        lines.push(format!(
            "rustudio crashed last time ({}): {}. Report: {}",
            report.title,
            report.summary,
            report.source_path.display()
        ));
    }
    lines
}
