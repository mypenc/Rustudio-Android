//! Per-process diagnostic logs for desktop support.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

const MAX_BYTES: u64 = 4 * 1024 * 1024;
const RETENTION: Duration = Duration::from_secs(14 * 24 * 60 * 60);

static SESSION: OnceLock<Session> = OnceLock::new();

struct Session {
    path: PathBuf,
    file: Mutex<File>,
}

fn dir() -> PathBuf {
    crate::paths::cache().join("rustudio").join("logs")
}

pub fn path() -> Option<&'static Path> {
    SESSION.get().map(|session| session.path.as_path())
}

pub fn open_folder() -> io::Result<()> {
    let dir = dir();
    fs::create_dir_all(&dir)?;
    let opener = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    if std::env::var_os("RUSTUDIO_NO_OPEN").is_none() {
        std::process::Command::new(opener).arg(dir).spawn()?;
    }
    Ok(())
}

fn create_session() -> io::Result<Session> {
    let dir = dir();
    fs::create_dir_all(&dir)?;
    for entry in fs::read_dir(&dir)?.flatten() {
        let name = entry.file_name();
        if name
            .to_str()
            .is_some_and(|s| s.starts_with("rustudio-") && s.ends_with(".log"))
            && entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age > RETENTION)
        {
            let _ = fs::remove_file(entry.path());
        }
    }

    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S-%3f");
    let path = dir.join(format!("rustudio-{stamp}-{}.log", (if cfg!(target_arch = "wasm32") { 1u32 } else { std::process::id() })));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&path)?;
    Ok(Session {
        path,
        file: Mutex::new(file),
    })
}

pub fn init() {
    if SESSION.get().is_some() {
        return;
    }
    match create_session() {
        Ok(session) => {
            if SESSION.set(session).is_ok() {
                if ::log::set_logger(&DiagnosticLogger).is_ok() {
                    ::log::set_max_level(::log::LevelFilter::Info);
                }
            }
        }
        Err(error) => eprintln!("Could not create a diagnostic log: {error}"),
    }
}

struct DiagnosticLogger;

impl ::log::Log for DiagnosticLogger {
    fn enabled(&self, metadata: &::log::Metadata<'_>) -> bool {
        let target = metadata.target();
        metadata.level() <= ::log::Level::Info && target.starts_with("rustudio")
            // The graphics stack's warnings and errors: a lost device or a
            // refused allocation is otherwise only a panic with no cause.
            || metadata.level() <= ::log::Level::Warn
                && ["wgpu", "naga", "egui_wgpu", "eframe"].iter().any(|t| target.starts_with(t))
    }

    fn log(&self, record: &::log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        if let Some(session) = SESSION.get() {
            let text = record.args().to_string();
            if let Ok(mut file) = session.file.lock() {
                write_record(&mut file, record.level(), record.target(), &text);
            }
        }
    }

    fn flush(&self) {}
}

fn write_record(file: &mut File, level: ::log::Level, target: &str, text: &str) {
    // Omit credential-bearing messages and URLs (which can carry signed query parameters).
    let lower = text.to_ascii_lowercase();
    let text = if [
        "roblosecurity",
        "warning:-do-not-share",
        "authorization:",
        "://",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        "[message omitted because it may contain credentials]"
    } else {
        // Bound the size of a single record and quote it to keep one event per line.
        text.get(..text.floor_char_boundary(8192)).unwrap_or(text)
    };

    if file
        .stream_position()
        .is_ok_and(|position| position >= MAX_BYTES)
    {
        if file.set_len(0).and_then(|()| file.rewind()).is_err() {
            return;
        }
        let _ = writeln!(file, "Diagnostic log restarted after reaching 4 MiB");
    }
    let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let _ = writeln!(file, "{now} {level:5} {target} {text:?}");
}

/// The panic hook can run while the logging lock is already held.
pub(crate) fn panic_at(location: &str) {
    if let Some(session) = SESSION.get() {
        if let Ok(mut file) = session.file.try_lock() {
            write_record(
                &mut file,
                ::log::Level::Error,
                "rustudio::crash",
                &format!("Panic at {location}; see crash report"),
            );
        }
    }
}
