//! Files on Android: the app's own folder in shared storage
//! (`Android/data/<package>/files`, visible to file managers, no permission
//! needed), a file picker drawn in egui over it ([`Picker`], behind
//! `crate::file_dialog::Dialogs`), and places handed over by other apps
//! through "Open with" (a `content:` URI in the launch Intent, copied in
//! through ContentResolver: [`launch_file`]).
//!
//! The system document picker (Storage Access Framework) needs
//! `startActivityForResult` and an `onActivityResult`, which `NativeActivity`
//! doesn't pass on to native code; this in-app picker covers the same files
//! without a Java class in the APK. Shared storage outside the app's folder
//! (Download, Documents, …) is listed once the app has storage access:
//! "All files access" on API 30+ (MANAGE_EXTERNAL_STORAGE, granted on a
//! settings page the picker opens), the storage permission below that
//! ([`super::jni::request_storage_access`]).

use std::path::{Path, PathBuf};
use web_time::Instant;

use crate::file_dialog::{Kind, Request};

/// A folder the picker can start from.
struct Root {
    label: &'static str,
    path: PathBuf,
}

/// The primary shared storage (`/storage/emulated/0`): the app's external
/// folder is `<shared>/Android/data/<package>/files`.
fn shared_storage() -> PathBuf {
    let mut p = super::data_dir();
    for _ in 0..4 {
        if let Some(parent) = p.parent() {
            p = parent.to_path_buf();
        }
    }
    if p.as_os_str().is_empty() || p == Path::new("/") {
        PathBuf::from("/storage/emulated/0")
    } else {
        p
    }
}

fn roots(storage_access: bool) -> Vec<Root> {
    let mut v = vec![Root { label: "rustudio", path: super::data_dir() }];
    let imported = super::files_dir().join("Imported");
    let _ = std::fs::create_dir_all(&imported);
    v.push(Root { label: "Imported", path: imported });
    if super::data_dir() != super::files_dir() {
        v.push(Root { label: "App data", path: super::files_dir() });
    }
    // Shared folders: the usual ones when they can be listed (files this
    // app made are always readable; everything with storage access), and
    // the whole shared storage once access was granted.
    let shared = shared_storage();
    for (label, sub) in [
        ("Download", "Download"),
        ("Documents", "Documents"),
        ("Pictures", "Pictures"),
        ("DCIM", "DCIM"),
        ("Music", "Music"),
    ] {
        let p = shared.join(sub);
        if std::fs::read_dir(&p).is_ok() {
            v.push(Root { label, path: p });
        }
    }
    if storage_access && std::fs::read_dir(&shared).is_ok() {
        v.push(Root { label: "Shared storage", path: shared });
    }
    v
}

/// One folder's listing, sorted folders first.
struct Listing {
    dir: PathBuf,
    folders: Vec<PathBuf>,
    files: Vec<PathBuf>,
}

impl Listing {
    fn read(dir: &Path, request: &Request) -> Self {
        let mut folders = Vec::new();
        let mut files = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                let hidden = p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.'));
                if hidden {
                    continue;
                }
                if p.is_dir() {
                    folders.push(p);
                } else if request.accepts(&p) {
                    files.push(p);
                }
            }
        }
        let key = |p: &PathBuf| p.file_name().map(|n| n.to_string_lossy().to_lowercase());
        folders.sort_by_key(key);
        files.sort_by_key(key);
        Listing { dir: dir.to_path_buf(), folders, files }
    }
}

/// Launches Android's Storage Access Framework document picker
/// (`ACTION_OPEN_DOCUMENT`, `CATEGORY_OPENABLE`, `*/*`, `EXTRA_ALLOW_MULTIPLE`).
fn open_system_picker() -> super::jni::Result<()> {
    use jni::objects::JValue;
    use jni::{jni_sig, jni_str};

    super::jni::with_activity(|env, activity| {
        let action = env.new_string("android.intent.action.OPEN_DOCUMENT")?;
        let intent = env.new_object(
            jni_str!("android/content/Intent"),
            jni_sig!("(Ljava/lang/String;)V"),
            &[JValue::Object(&action)],
        )?;
        let category = env.new_string("android.intent.category.OPENABLE")?;
        env.call_method(
            &intent,
            jni_str!("addCategory"),
            jni_sig!("(Ljava/lang/String;)Landroid/content/Intent;"),
            &[JValue::Object(&category)],
        )?;
        let mime = env.new_string("*/*")?;
        env.call_method(
            &intent,
            jni_str!("setType"),
            jni_sig!("(Ljava/lang/String;)Landroid/content/Intent;"),
            &[JValue::Object(&mime)],
        )?;
        let extra_multi = env.new_string("android.intent.extra.ALLOW_MULTIPLE")?;
        env.call_method(
            &intent,
            jni_str!("putExtra"),
            jni_sig!("(Ljava/lang/String;Z)Landroid/content/Intent;"),
            &[JValue::Object(&extra_multi), JValue::Bool(true)],
        )?;
        env.call_method(
            activity,
            jni_str!("startActivity"),
            jni_sig!("(Landroid/content/Intent;)V"),
            &[JValue::Object(&intent)],
        )?;
        Ok(())
    })
}

/// The in-app file dialog.
pub struct Picker {
    request: Request,
    kind: Kind,
    multi: bool,
    listing: Listing,
    selected: Option<PathBuf>,
    selected_multi: Vec<PathBuf>,
    /// Save: the name to write; new folder: its name.
    name: String,
    new_folder: Option<String>,
    error: Option<String>,
    /// Shared storage may be read (asked again now and then while the
    /// dialog is open: the grant happens on a system page).
    storage_access: bool,
    access_checked: Instant,
}

impl Picker {
    pub fn new(request: Request, kind: Kind) -> Self {
        Self::new_with_mode(request, kind, false)
    }

    pub fn new_multi(request: Request) -> Self {
        Self::new_with_mode(request, Kind::PickFile, true)
    }

    fn new_with_mode(request: Request, kind: Kind, multi: bool) -> Self {
        let start = request
            .directory
            .clone()
            .filter(|d| d.is_dir())
            .unwrap_or_else(|| {
                let d = super::data_dir().join("Places");
                let _ = std::fs::create_dir_all(&d);
                d
            });
        let mut name = request.file_name.clone().unwrap_or_default();
        if kind != Kind::SaveFile {
            name.clear();
        }
        Picker {
            listing: Listing::read(&start, &request),
            request,
            kind,
            multi,
            selected: None,
            selected_multi: Vec::new(),
            name,
            new_folder: None,
            error: None,
            storage_access: super::jni::has_storage_access(),
            access_checked: Instant::now(),
        }
    }

    /// Re-asks the system about storage access about once a second.
    fn poll_access(&mut self, ctx: &egui::Context) {
        if self.access_checked.elapsed().as_secs_f32() > 1.0 {
            self.access_checked = Instant::now();
            let now = super::jni::has_storage_access();
            if now && !self.storage_access {
                // Folders that were unreadable list now.
                self.go(&self.listing.dir.clone());
            }
            self.storage_access = now;
        }
        if !self.storage_access {
            ctx.request_repaint_after(std::time::Duration::from_secs(1));
        }
    }

    fn go(&mut self, dir: &Path) {
        self.listing = Listing::read(dir, &self.request);
        self.selected = None;
        self.error = None;
    }

    /// The answer when the single-file dialog closes: `Some(Some(path))` chosen,
    /// `Some(None)` cancelled, `None` while it stays open.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Option<PathBuf>> {
        self.show_inner(ctx).map(|opt| opt.and_then(|mut v| v.pop()))
    }

    /// The answer when the multi-file dialog closes: `Some(paths)` chosen
    /// (empty when cancelled), `None` while it stays open.
    pub fn show_multi(&mut self, ctx: &egui::Context) -> Option<Vec<PathBuf>> {
        self.show_inner(ctx).map(|opt| opt.unwrap_or_default())
    }

    fn show_inner(&mut self, ctx: &egui::Context) -> Option<Option<Vec<PathBuf>>> {
        let title = self.request.title.clone().unwrap_or_else(|| {
            match self.kind {
                Kind::PickFile if self.multi => "Import Files",
                Kind::PickFile => "Open",
                Kind::PickFolder => "Choose folder",
                Kind::SaveFile => "Save as",
            }
            .to_string()
        });
        self.poll_access(ctx);
        let roots = roots(self.storage_access);
        let screen = ctx.content_rect();
        let size = egui::vec2((screen.width() - 32.0).min(720.0), (screen.height() - 32.0).min(560.0));
        let mut result: Option<Option<Vec<PathBuf>>> = None;
        let mut open = true;
        egui::Window::new(title)
            .id(egui::Id::new("android-file-picker"))
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .default_size(size)
            .min_size(size)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.spacing_mut().button_padding = egui::vec2(10.0, 8.0);
                // Roots and the path.
                ui.horizontal_wrapped(|ui| {
                    for r in &roots {
                        let here = self.listing.dir.starts_with(&r.path);
                        if ui.selectable_label(here, r.label).clicked() {
                            self.go(&r.path);
                        }
                    }
                });
                if !self.storage_access {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new(
                                "Grant storage access to browse Download, Pictures, Documents, and Music directly with multi-select right inside Rustudio.",
                            )
                            .weak()
                            .size(12.0),
                        );
                        if ui.small_button("Grant storage access").clicked() {
                            if let Err(e) = super::jni::request_storage_access() {
                                self.error = Some(format!("Could not open the permission page: {e}"));
                            }
                        }
                    });
                }
                ui.horizontal_wrapped(|ui| {
                    let up = self.listing.dir.parent().map(Path::to_path_buf);
                    let at_root = roots.iter().any(|r| r.path == self.listing.dir);
                    if ui.add_enabled(up.is_some() && !at_root, egui::Button::new("⬆ Up")).clicked() {
                        if let Some(u) = up {
                            self.go(&u);
                        }
                    }
                    if ui.button("New folder").clicked() {
                        self.new_folder = Some(String::new());
                    }
                    if ui.button("Open System Picker").clicked() {
                        if let Err(e) = open_system_picker() {
                            self.error = Some(format!("Could not open system picker: {e}"));
                        }
                    }
                    ui.add(egui::Label::new(egui::RichText::new(shorten(&self.listing.dir)).weak()).truncate());
                });
                if let Some(nf) = &mut self.new_folder {
                    let mut make = false;
                    let mut cancel = false;
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        let r = ui.add(egui::TextEdit::singleline(nf).desired_width(220.0));
                        if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            make = true;
                        }
                        make |= ui.button("Create").clicked();
                        cancel = ui.button("Cancel").clicked();
                    });
                    if make && !nf.trim().is_empty() {
                        let dir = self.listing.dir.join(safe_name(nf.trim()));
                        match std::fs::create_dir_all(&dir) {
                            Ok(()) => {
                                self.new_folder = None;
                                self.go(&dir);
                            }
                            Err(e) => self.error = Some(format!("{}: {e}", dir.display())),
                        }
                    } else if cancel {
                        self.new_folder = None;
                    }
                }
                ui.separator();
                let list_height = ui.available_height() - 56.0;
                egui::ScrollArea::vertical().max_height(list_height.max(80.0)).auto_shrink([false, false]).show(ui, |ui| {
                    let row = egui::vec2(ui.available_width(), 36.0);
                    let mut enter: Option<PathBuf> = None;
                    for f in &self.listing.folders {
                        let name = format!("📁 {}", f.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
                        let sel = self.kind == Kind::PickFolder && self.selected.as_deref() == Some(f.as_path());
                        let r = ui.add_sized(row, egui::Button::selectable(sel, name));
                        if r.double_clicked() || (r.clicked() && self.kind != Kind::PickFolder) {
                            enter = Some(f.clone());
                        } else if r.clicked() {
                            self.selected = Some(f.clone());
                        }
                    }
                    for f in &self.listing.files {
                        let label = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        if self.multi {
                            let mut checked = self.selected_multi.iter().any(|p| p == f);
                            let mut toggled = false;
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut checked, "").changed() {
                                    toggled = true;
                                }
                                let btn_w = ui.available_width().max(40.0);
                                let r = ui.add_sized(
                                    egui::vec2(btn_w, 36.0),
                                    egui::Button::selectable(checked, format!("📄 {label}")),
                                );
                                if r.clicked() {
                                    checked = !checked;
                                    toggled = true;
                                }
                            });
                            if toggled {
                                if checked {
                                    if !self.selected_multi.iter().any(|p| p == f) {
                                        self.selected_multi.push(f.clone());
                                    }
                                } else {
                                    self.selected_multi.retain(|p| p != f);
                                }
                            }
                        } else {
                            let sel = self.selected.as_deref() == Some(f.as_path());
                            let r = ui.add_sized(row, egui::Button::selectable(sel, format!("📄 {label}")));
                            if r.clicked() {
                                self.selected = Some(f.clone());
                                if self.kind == Kind::SaveFile {
                                    self.name = label.clone();
                                }
                            }
                            if r.double_clicked() && self.kind == Kind::PickFile {
                                result = Some(Some(vec![f.clone()]));
                            }
                        }
                    }
                    if self.listing.folders.is_empty() && self.listing.files.is_empty() {
                        ui.add_space(12.0);
                        ui.weak("Nothing here. Files put in this folder with a file manager, or opened\nfrom one with rustudio, show up in this list.");
                    }
                    if let Some(d) = enter {
                        self.go(&d);
                    }
                });
                ui.separator();
                if let Some(e) = &self.error {
                    ui.colored_label(ui.visuals().error_fg_color, e);
                }
                ui.horizontal(|ui| {
                    if self.multi {
                        if ui.add_enabled(!self.listing.files.is_empty(), egui::Button::new("Select All")).clicked() {
                            for f in &self.listing.files {
                                if !self.selected_multi.iter().any(|p| p == f) {
                                    self.selected_multi.push(f.clone());
                                }
                            }
                        }
                        if ui.add_enabled(!self.selected_multi.is_empty(), egui::Button::new("Clear")).clicked() {
                            self.selected_multi.clear();
                        }
                    } else if self.kind == Kind::SaveFile {
                        ui.label("Name:");
                        ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(240.0));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.multi {
                            let count = self.selected_multi.len();
                            let label = format!("Import Selected ({count})");
                            if ui.add_enabled(count > 0, egui::Button::new(label)).clicked() {
                                result = Some(Some(self.selected_multi.clone()));
                            }
                        } else {
                            let (verb, ready) = match self.kind {
                                Kind::PickFile => ("Open", self.selected.is_some()),
                                Kind::PickFolder => ("Choose", true),
                                Kind::SaveFile => ("Save", !self.name.trim().is_empty()),
                            };
                            if ui.add_enabled(ready, egui::Button::new(verb)).clicked() {
                                result = Some(match self.kind {
                                    Kind::PickFile => self.selected.clone().map(|p| vec![p]),
                                    Kind::PickFolder => Some(vec![self.selected.clone().unwrap_or_else(|| self.listing.dir.clone())]),
                                    Kind::SaveFile => Some(vec![self.listing.dir.join(self.save_name())]),
                                });
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            result = Some(None);
                        }
                    });
                });
            });
        if !open {
            result = Some(None);
        }
        result
    }

    /// The name to save as, with the first filter's extension when none was
    /// typed (as desktop dialogs add it).
    fn save_name(&self) -> String {
        let name = safe_name(self.name.trim());
        let has_ext = Path::new(&name).extension().is_some();
        match (has_ext, self.request.filters.first().and_then(|(_, e)| e.first())) {
            (false, Some(ext)) => format!("{name}.{ext}"),
            _ => name,
        }
    }
}

/// A file name without path separators or control characters.
fn safe_name(s: &str) -> String {
    let out: String = s.chars().map(|c| if c == '/' || c == '\\' || c.is_control() { '_' } else { c }).collect();
    if out.is_empty() { "untitled".into() } else { out }
}

/// A path with the app's folder prefix replaced by its label.
fn shorten(p: &Path) -> String {
    for r in roots(true) {
        if let Ok(rest) = p.strip_prefix(&r.path) {
            let rest = rest.display().to_string();
            return if rest.is_empty() { r.label.to_string() } else { format!("{} / {}", r.label, rest.replace('/', " / ")) };
        }
    }
    p.display().to_string()
}

/// The place or model this activity was started to open ("Open with
/// rustudio" in a file manager, a share sheet, a download notification):
/// the Intent's data URI, copied into the app's folder when it is a
/// `content:` URI (another app's document, readable only through its
/// ContentResolver) — unless the last start ended while opening that same
/// file ([`skip_crashed`]).
pub fn launch_file(_app: &super::AndroidApp) -> Option<PathBuf> {
    let uri = match super::jni::launch_intent_uri() {
        Ok(u) => u?,
        Err(e) => {
            log::warn!("launch intent: {e}");
            return None;
        }
    };
    log::info!("launch intent: {uri}");
    let path = import_uri(&uri)?;
    if skip_crashed(&path) {
        return None;
    }
    opening_marker(Some(&path));
    Some(path)
}

/// A local path for `uri`, through the running activity.
pub fn import_uri(uri: &str) -> Option<PathBuf> {
    import_uri_with(uri, &|f| super::jni::with_activity(f))
}

/// [`import_uri`] through the Activity of a raw `ANativeActivity` (a
/// second activity of the process: src/android/intent.rs).
///
/// # Safety
/// `activity` must be the live `ANativeActivity`.
pub unsafe fn import_uri_from(activity: *mut ndk_sys::ANativeActivity, uri: &str) -> Option<PathBuf> {
    // SAFETY: the caller's live struct.
    import_uri_with(uri, &|f| unsafe { super::jni::with_native_activity(activity, f) })
}

/// A JNI call made with some Activity: the name and the bytes of a URI.
type Fetch<'a> = &'a mut dyn FnMut(&mut jni::Env, &jni::objects::JObject) -> jni::errors::Result<(Option<String>, Vec<u8>)>;

fn import_uri_with(uri: &str, with: &dyn Fn(Fetch<'_>) -> super::jni::Result<(Option<String>, Vec<u8>)>) -> Option<PathBuf> {
    if let Some(rest) = uri.strip_prefix("file://") {
        let p = PathBuf::from(percent_decode(rest));
        return p.is_file().then_some(p);
    }
    if !uri.starts_with("content://") {
        return None;
    }
    let (name, bytes) = match with(&mut |env, activity| {
        // (a provider without a name column throws: the name is optional,
        // the exception must not stand in the way of the read)
        let name = super::jni::display_name(env, activity, uri).unwrap_or_else(|e| {
            log::warn!("{uri}: no display name: {e}");
            env.exception_clear();
            None
        });
        let bytes = super::jni::content_bytes(env, activity, uri)?;
        Ok((name, bytes))
    }) {
        Ok(x) => x,
        Err(e) => {
            log::error!("{uri}: {e}");
            return None;
        }
    };
    let name = name.filter(|n| !n.is_empty()).unwrap_or_else(|| "place.rbxl".to_string());
    let dir = super::data_dir().join("Imported");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::error!("{}: {e}", dir.display());
        return None;
    }
    // The same file handed over again (the activity recreated with its
    // VIEW intent, "Open with" twice) is the copy already made, not
    // "name (2)": the copies are compared byte for byte.
    let wanted = dir.join(safe_name(&name));
    if let Some(same) = same_import(&wanted, &bytes) {
        log::info!("{name} is already imported as {}", same.display());
        return Some(same);
    }
    let path = unique_path(&wanted);
    match std::fs::write(&path, &bytes) {
        Ok(()) => {
            log::info!("imported {} bytes to {}", bytes.len(), path.display());
            Some(path)
        }
        Err(e) => {
            log::error!("{}: {e}", path.display());
            None
        }
    }
}

/// Among `p` and its `p (n)` copies, the one holding exactly `bytes`.
fn same_import(p: &Path, bytes: &[u8]) -> Option<PathBuf> {
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    let dir = p.parent().unwrap_or(Path::new("."));
    std::iter::once(p.to_path_buf())
        .chain((2..100).map(|n| dir.join(format!("{stem} ({n}){ext}"))))
        .take_while(|c| c.exists())
        .find(|c| std::fs::metadata(c).is_ok_and(|m| m.len() == bytes.len() as u64) && std::fs::read(c).is_ok_and(|b| b == bytes))
}

// ------------------------------------------------------ opened while running

/// Files handed over while the editor runs (a second activity's VIEW
/// intent, src/android/intent.rs), for the editor's next frame.
static OPENED: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());

pub fn push_opened(p: PathBuf) {
    OPENED.lock().unwrap_or_else(|e| e.into_inner()).push(p);
}

/// The files pushed since the last call (the editor opens them).
pub fn take_opened() -> Vec<PathBuf> {
    std::mem::take(&mut *OPENED.lock().unwrap_or_else(|e| e.into_inner()))
}

// ------------------------------------------------------- the opening marker

/// Lines for Output at start (what [`launch_file`] decided).
static NOTICES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

pub fn take_notices() -> Vec<String> {
    std::mem::take(&mut *NOTICES.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The file an open is under way for, until the start has survived
/// [`SETTLED`] past it.
static OPENING: std::sync::Mutex<Option<(PathBuf, Instant)>> = std::sync::Mutex::new(None);
/// A start that ran this long after opening a file counts as having
/// opened it.
const SETTLED: std::time::Duration = std::time::Duration::from_secs(20);

/// `state/opening`: the file the last automatic open was of, left behind
/// when the process ended within [`SETTLED`] of it.
fn marker_path() -> PathBuf {
    super::state_dir().join("opening")
}

fn opening_marker(p: Option<&Path>) {
    let m = marker_path();
    match p {
        Some(p) => {
            let _ = std::fs::create_dir_all(super::state_dir());
            if let Err(e) = std::fs::write(&m, p.to_string_lossy().as_bytes()) {
                log::warn!("{}: {e}", m.display());
            }
            opening(p);
        }
        None => {
            let _ = std::fs::remove_file(&m);
        }
    }
}

/// Whether the last start ended (a crash, a kill) within [`SETTLED`] of
/// opening `p` by itself. The Activity's VIEW intent is delivered again
/// each time its task comes back after the process died, so a place that
/// brings the app down would otherwise bring it down at every launch,
/// with nothing to be done about it on the device. The file is left alone
/// then and Output says so; File ▸ Open Recent opens it on purpose, and
/// once such a start has settled the marker is forgotten.
fn skip_crashed(p: &Path) -> bool {
    let Ok(last) = std::fs::read_to_string(marker_path()) else { return false };
    if Path::new(last.trim()) != p {
        return false;
    }
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let line = format!(
        "rustudio ended within {} s of opening {name} the last time, so it wasn't opened again by itself. \
         File ▸ Open Recent opens it on purpose (adb logcat -s rustudio RustStdoutStderr tells what ended the last start).",
        SETTLED.as_secs()
    );
    log::warn!("{line}");
    NOTICES.lock().unwrap_or_else(|e| e.into_inner()).push(line);
    true
}

/// A file is being opened (the editor's `open_path`): the clock for
/// [`settle`] starts (again).
pub fn opening(p: &Path) {
    *OPENING.lock().unwrap_or_else(|e| e.into_inner()) = Some((p.to_path_buf(), Instant::now()));
}

/// Once a frame: a start that has run [`SETTLED`] past its last open has
/// opened that file for good; the marker it may have left goes.
pub fn settle() {
    let mut o = OPENING.lock().unwrap_or_else(|e| e.into_inner());
    let Some((p, since)) = o.as_ref() else { return };
    if since.elapsed() < SETTLED {
        return;
    }
    if std::fs::read_to_string(marker_path()).is_ok_and(|last| Path::new(last.trim()) == p) {
        opening_marker(None);
    }
    *o = None;
}

/// `name.ext`, else `name (2).ext`, … so an import never overwrites.
fn unique_path(p: &Path) -> PathBuf {
    if !p.exists() {
        return p.to_path_buf();
    }
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    let dir = p.parent().unwrap_or(Path::new("."));
    (2..)
        .map(|n| dir.join(format!("{stem} ({n}){ext}")))
        .find(|c| !c.exists())
        .unwrap_or_else(|| p.to_path_buf())
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = s.get(i + 1..i + 3).and_then(|h| u8::from_str_radix(h, 16).ok()).ok_or(()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
