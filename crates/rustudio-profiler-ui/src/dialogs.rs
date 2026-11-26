//! Open / Export file dialogs, awaited on a worker thread so the window
//! keeps drawing (and recording) while one is open. Android has no rfd
//! backend: Export writes into the app's data folder
//! (`Android/data/<app>/files/profiles/`, `adb pull`), named in the log,
//! and Open is not offered (the window reads the captures it records).

use std::path::PathBuf;
#[cfg(not(any(target_os = "android", target_arch = "wasm32")))]
use std::sync::mpsc::{channel, Receiver, TryRecvError};
#[cfg(any(target_os = "android", target_arch = "wasm32"))]
use std::sync::mpsc::{Receiver, TryRecvError};

pub enum Done {
    Open(PathBuf),
    Save { path: PathBuf, selection: bool },
}

#[cfg_attr(any(target_os = "android", target_arch = "wasm32"), allow(dead_code))]
enum Want {
    Open,
    Save { selection: bool },
}

#[derive(Default)]
pub struct Dialogs {
    pending: Option<(Want, Receiver<Option<PathBuf>>)>,
}

impl Dialogs {
    #[cfg(not(any(target_os = "android", target_arch = "wasm32")))]
    pub fn open(&mut self) {
        let d = rfd::AsyncFileDialog::new()
            .set_title("Open a profiler capture")
            .add_filter("Profiler captures", &["rsprof", "json", "gz"])
            .add_filter("All files", &["*"]);
        self.start(Want::Open, Box::pin(d.pick_file()));
    }

    /// Android: no file dialog (the app's own picker is the studio's,
    /// src/android/storage.rs, which this crate doesn't see).
    #[cfg(any(target_os = "android", target_arch = "wasm32"))]
    pub fn open(&mut self) {
        log::warn!("profiler: Open needs a file dialog; push a capture with adb and open it from the command line instead");
    }

    #[cfg(not(any(target_os = "android", target_arch = "wasm32")))]
    pub fn save(&mut self, chrome: bool, selection: bool) {
        let (name, filter, ext) = export_name(chrome);
        let d = rfd::AsyncFileDialog::new().set_title("Export the capture").set_file_name(name).add_filter(filter, &[ext]);
        self.start(Want::Save { selection }, Box::pin(d.save_file()));
    }

    /// Android: straight into `<data>/profiles/<name>` (the external files
    /// folder: `adb pull`, or a PC over USB), as if the dialog had answered.
    #[cfg(any(target_os = "android", target_arch = "wasm32"))]
    pub fn save(&mut self, chrome: bool, selection: bool) {
        if self.pending.is_some() {
            return;
        }
        let (name, _, _) = export_name(chrome);
        let dir = rustudio_core::paths::data().join("profiles");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            log::warn!("profiler: {}: {e}", dir.display());
            return;
        }
        let path = dir.join(name);
        log::info!("profiler: exporting to {}", path.display());
        let (tx, rx) = std::sync::mpsc::channel();
        let _ = tx.send(Some(path));
        self.pending = Some((Want::Save { selection }, rx));
    }

    #[cfg(not(any(target_os = "android", target_arch = "wasm32")))]
    fn start(&mut self, want: Want, fut: std::pin::Pin<Box<dyn std::future::Future<Output = Option<rfd::FileHandle>> + Send>>) {
        if self.pending.is_some() {
            return;
        }
        let (tx, rx) = channel();
        let ok = std::thread::Builder::new()
            .name("profiler-dialog".into())
            .spawn(move || {
                let _ = tx.send(pollster::block_on(fut).map(|h| h.path().to_path_buf()));
            })
            .is_ok();
        if ok {
            self.pending = Some((want, rx));
        }
    }

    pub fn poll(&mut self) -> Option<Done> {
        let (_, rx) = self.pending.as_ref()?;
        let path = match rx.try_recv() {
            Ok(p) => p,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => None,
        };
        let (want, _) = self.pending.take()?;
        let path = path?;
        Some(match want {
            Want::Open => Done::Open(path),
            Want::Save { selection } => Done::Save { path, selection },
        })
    }
}

/// The export's file name, filter label and extension.
fn export_name(chrome: bool) -> (String, &'static str, &'static str) {
    let stamp = chrono_like_stamp();
    if chrome {
        (format!("rustudio-profile-{stamp}.json"), "Chrome trace", "json")
    } else {
        (format!("rustudio-profile-{stamp}.rsprof"), "rustudio capture", "rsprof")
    }
}

/// "20261119-2315" from the system clock (UTC; just to make names unique).
fn chrono_like_stamp() -> String {
    let secs = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = secs / 86_400;
    let (y, m, d) = civil(days as i64);
    let hm = secs % 86_400;
    format!("{y:04}{m:02}{d:02}-{:02}{:02}", hm / 3600, hm / 60 % 60)
}

/// Days since 1970-01-01 → (year, month, day).
fn civil(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + (m <= 2) as i64, m, d)
}
