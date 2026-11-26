//! Roblox Studio's built-in content, downloaded on demand.
//!
//! rustudio ships no Roblox files. When this machine has no Studio install
//! (no `RUSTUDIO_STUDIO_DIR`, no `~/.local/share/rustudio/content` link),
//! rustudio downloads the parts of Studio it reads from Roblox's own
//! deployment CDN (`setup.rbxcdn.com`), the same packages the Studio
//! installer and Linux launchers such as Vinegar fetch, and unpacks them in
//! Studio's folder layout under `~/.local/share/rustudio/studio`:
//!
//! - `content/{avatar,fonts,sky,sounds,textures,models,configs}`: default
//!   characters, fonts, sky, sounds and `rbxasset://` textures,
//! - `PlatformContent/pc/{textures,terrain,fonts}`: studs, terrain colours,
//! - `ExtraContent/{scripts,models,textures}`: the default PlayerModule
//!   (camera + controls), character sounds and other built-in scripts,
//! - `ReflectionMetadata.xml`, taken alone out of `RobloxStudio.zip` with
//!   HTTP range requests (about 30 KB instead of the 120 MB package).
//!
//! About 70 MB to download, a few packages at a time. `rustudio
//! --install-studio-content` does it in the terminal; a normal launch does
//! it in the background the first time (`RUSTUDIO_NO_STUDIO_FETCH=1` turns
//! that off). The install counts as complete — Play uses it — once the
//! packages and ReflectionMetadata.xml are there; the retired textures and
//! the View Selector follow, and are made good at a later start if they
//! don't.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// (package, folder it unpacks into, relative to the Studio root).
const PACKAGES: &[(&str, &str)] = &[
    ("content-avatar.zip", "content/avatar"),
    ("content-configs.zip", "content/configs"),
    ("content-fonts.zip", "content/fonts"),
    ("content-models.zip", "content/models"),
    ("content-sky.zip", "content/sky"),
    ("content-sounds.zip", "content/sounds"),
    ("content-textures2.zip", "content/textures"),
    ("content-textures3.zip", "PlatformContent/pc/textures"),
    ("content-terrain.zip", "PlatformContent/pc/terrain"),
    ("content-platform-fonts.zip", "PlatformContent/pc/fonts"),
    ("extracontent-scripts.zip", "ExtraContent/scripts"),
    ("extracontent-models.zip", "ExtraContent/models"),
    ("extracontent-textures.zip", "ExtraContent/textures"),
    ("BuiltInPlugins.zip", "BuiltInPlugins"),
    ("extracontent-luapackages.zip", "ExtraContent/LuaPackages"),
    ("content-studio_svg_textures.zip", "content/studio_svg_textures"),
];
/// Single files pulled out of a big package by range requests.
const SINGLE_FILES: &[(&str, &str)] = &[("RobloxStudio.zip", "ReflectionMetadata.xml")];

const VERSION_URL: &str = "https://clientsettingscdn.roblox.com/v2/client-version/WindowsStudio64";
const CDN: &str = "https://setup.rbxcdn.com";
const MARKER: &str = ".rustudio-studio-version";
// Bump when the installed content layout changes, even within one CDN version.
const CONTENT_REVISION: &str = "legacy-textures-1";

/// Where the downloaded content lives.
pub fn install_dir() -> PathBuf {
    if let Ok(p) = std::env::var("RUSTUDIO_STUDIO_DOWNLOAD_DIR") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    crate::paths::data().join("rustudio/studio")
}

/// The downloaded Studio root, when a complete download is there.
pub fn installed() -> Option<PathBuf> {
    let d = install_dir();
    d.join(MARKER).is_file().then_some(d)
}

/// The packages an installed download lacks (one made before they were
/// listed): fetched into it as they are, nothing else touched.
fn missing_packages(root: &Path) -> Vec<(&'static str, &'static str)> {
    PACKAGES.iter().copied().filter(|(_, sub)| !root.join(sub).is_dir()).collect()
}

/// Fetches `missing_packages` into an installed `root` of `version`.
fn install_missing_packages(agent: &ureq::Agent, version: &str, root: &Path, progress: &(dyn Fn(&str) + Sync)) -> Result<(), String> {
    for (pkg, sub) in missing_packages(root) {
        progress(&format!("downloading {pkg}"));
        let done = root.join(format!(".done-{pkg}"));
        install_package(agent, version, pkg, &root.join(sub), &done)?;
        let _ = std::fs::remove_file(done);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadUiStatus {
    Idle,
    Downloading(String),
    Ready,
    OfflineError(String),
}

static DOWNLOAD_STATUS: Mutex<DownloadUiStatus> = Mutex::new(DownloadUiStatus::Idle);

pub fn download_ui_status() -> DownloadUiStatus {
    DOWNLOAD_STATUS.lock().map(|g| g.clone()).unwrap_or(DownloadUiStatus::Idle)
}

fn set_download_status(st: DownloadUiStatus) {
    if let Ok(mut g) = DOWNLOAD_STATUS.lock() {
        *g = st;
    }
}

/// True when Roblox Studio's official BuiltInPlugins folder is present on disk.
pub fn builtin_plugins_installed() -> bool {
    crate::default_scripts::studio_dir()
        .map(|r| r.join("BuiltInPlugins").is_dir())
        .unwrap_or(false)
}

/// Retry downloading Studio content & official BuiltInPlugins in the background.
pub fn retry_background_install() {
    set_download_status(DownloadUiStatus::Downloading("Checking Roblox Studio packages…".into()));
    spawn_background_install_if_missing();
}

static MESSAGES: Mutex<Vec<(bool, String)>> = Mutex::new(Vec::new());

/// Messages for Output from the background download: (is_error, text).
pub fn take_messages() -> Vec<(bool, String)> {
    MESSAGES.lock().map(|mut m| std::mem::take(&mut *m)).unwrap_or_default()
}
fn post(err: bool, t: String) {
    if let Ok(mut m) = MESSAGES.lock() {
        m.push((err, t));
    }
}
/// A line for Output from another start-up check (the Android entry
/// point's network check), shown with the download's messages.
pub fn note(err: bool, text: String) {
    post(err, text);
}

/// On a normal launch with no Studio content anywhere, download it in the
/// background and report to Output.
pub fn spawn_background_install_if_missing() {
    fetch_classic_oof_once();
    if std::env::var_os("RUSTUDIO_NO_STUDIO_FETCH").is_some() {
        return;
    }
    if let Some(root) = crate::default_scripts::studio_dir() {
        // Upgrade older rustudio downloads without replacing the rest of
        // their content: packages listed since, the View Selector, the
        // LuaPackages assets and the Studio UI textures.
        let packages = if root == install_dir() { missing_packages(&root) } else { Vec::new() };
        let need_vs = !has_view_selector(&root);
        let need_lp = !has_lua_package_assets(&root);
        let need_ui = !has_studio_ui_textures(&root);
        if root == install_dir() && (need_vs || need_lp || need_ui || !packages.is_empty()) {
            if let Some(marker) = std::fs::read_to_string(root.join(MARKER)).ok() {
                if let Some(version) = marker.split_whitespace().next().and_then(studio_version) {
                    let version = version.to_owned();
                    let _ = rt_thread::Builder::new().name("studio-content-upgrade".into()).spawn(move || {
                        let a = agent();
                        if !packages.is_empty() {
                            let names: Vec<&str> = packages.iter().map(|(p, _)| *p).collect();
                            set_download_status(DownloadUiStatus::Downloading(format!("Downloading {}…", names.join(", "))));
                            post(false, format!("Studio content: downloading {} (new in this version of rustudio)…", names.join(", ")));
                            match install_missing_packages(&a, &version, &root, &|msg| {
                                set_download_status(DownloadUiStatus::Downloading(msg.to_string()));
                            }) {
                                Ok(()) => {
                                    set_download_status(DownloadUiStatus::Ready);
                                    post(false, "Studio content: downloaded".into());
                                }
                                Err(e) => post(true, format!("Couldn't download Studio content: {e}")),
                            }
                        }
                        if need_vs {
                            if let Err(e) = install_view_selector(&a, &version, &root) {
                                post(true, format!("Couldn't download Studio's View Selector: {e}"));
                            }
                        }
                        if need_lp {
                            if let Err(e) = install_lua_package_assets(&a, &version, &root) {
                                post(true, format!("Couldn't download Studio's LuaPackages assets: {e}"));
                            }
                        }
                        if need_ui {
                            if let Err(e) = install_studio_ui_textures(&a, &version, &root) {
                                post(true, format!("Couldn't download Studio's UI textures: {e}"));
                            }
                        }
                    });
                }
            }
        }
        return;
    }
    let _ = rt_thread::Builder::new().name("studio-content".into()).spawn(|| {
        post(false, "No Roblox Studio install found: downloading Studio's built-in content (default character, player scripts, fonts, textures, sounds, the CoreScripts' icon font; about 70 MB) from Roblox's CDN…".into());
        // The packages as they come, so a slow connection is seen to get
        // somewhere (the retired textures are a hundred small fetches: one
        // line for all of them).
        let step = |s: &str| {
            if s.starts_with("downloading ") || (s.starts_with("fetching ") && !s.starts_with("fetching legacy ")) {
                post(false, format!("Studio content: {s}…"));
            } else if s.starts_with("ready") || s.starts_with("couldn't ") {
                post(false, format!("Studio content: {s}"));
            }
        };
        // A connection that drops gets a few more tries here (the download
        // goes on where it stopped), and again at the next start.
        const TRIES: u32 = 3;
        for attempt in 1..=TRIES {
            match install(&step) {
                Ok(v) => {
                    post(false, format!(
                        "Studio content {v} installed in {}. Play now uses Roblox's default character and player scripts; restart rustudio to use Studio's fonts everywhere.",
                        install_dir().display()
                    ));
                    break;
                }
                Err(e) if attempt < TRIES => {
                    post(false, format!("Studio content: {e}; trying again in 20 s"));
                    rt_thread::sleep(std::time::Duration::from_secs(20));
                }
                Err(e) => post(true, format!(
                    "Couldn't download Studio's built-in content: {e}. It continues where it stopped at the next start; `rustudio --install-studio-content` retries now, or point RUSTUDIO_STUDIO_DIR at a Studio install."
                )),
            }
        }
    });
}

fn agent() -> ureq::Agent {
    use std::time::Duration;
    // A slow connection may take what it takes for an 18 MB package; a
    // dead one (a phone that changed networks, a process thawed after a
    // while in the background) must fail within the minute so the next
    // try can go on where this one stopped, not sit on a socket for as
    // long as the kernel retransmits.
    // One agent (connection pool) for the whole install.
    static AGENT: std::sync::LazyLock<ureq::Agent> = std::sync::LazyLock::new(|| {
        let config = ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_secs(15)))
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .timeout_per_read(Some(Duration::from_secs(60)))
            .max_idle_connections_per_host(4)
            .user_agent(concat!("rustudio/", env!("CARGO_PKG_VERSION")))
            .build();
        rustudio_account::http::agent(config)
    });
    AGENT.clone()
}

fn get(agent: &ureq::Agent, url: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>, String> {
    let mut req = agent.get(url);
    if let Some((a, b)) = range {
        req = req.header("Range", &format!("bytes={a}-{b}"));
    }
    let mut resp = req.call().map_err(|e| format!("{url}: {e}"))?;
    if range.is_some() && resp.status().as_u16() != 206 {
        return Err(format!("{url}: no range support (HTTP {})", resp.status()));
    }
    resp.body_mut()
        .with_config()
        .limit(1 << 30)
        .read_to_vec()
        .map_err(|e| format!("{url}: {e}"))
}

fn content_length(agent: &ureq::Agent, url: &str) -> Result<u64, String> {
    let resp = agent.head(url).call().map_err(|e| format!("{url}: {e}"))?;
    resp.headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("{url}: no Content-Length"))
}

/// Download and unpack everything. Returns the Studio version installed.
/// `progress` gets one line per step.
/// Studio 0.533 (June 2022), the last build whose sounds package has the
/// classic death sound `uuhhh.mp3` ("Oof"); later Studios play a replacement.
const CLASSIC_OOF_PKG: &str = "version-58ed123c874e44c1-content-sounds.zip";

/// Where rustudio keeps the classic Oof it fetched from Roblox's CDN (its own
/// data folder: never inside a Studio install, never in the repo).
pub fn classic_oof() -> PathBuf {
    crate::paths::data().join("rustudio/classic-sounds/uuhhh.mp3")
}

/// Fetch the classic Oof in the background, once, unless it is already here.
pub fn fetch_classic_oof_once() {
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    // (RUSTUDIO_NO_STUDIO_FETCH or the rustudio option turns it off)
    if cfg!(test)
        || !crate::rustudio_settings::classic_death_sound_fetch()
        || classic_oof().is_file()
        || STARTED.swap(true, std::sync::atomic::Ordering::AcqRel)
    {
        return;
    }
    let _ = rt_thread::Builder::new().name("classic-oof".into()).spawn(|| {
        let dest = classic_oof();
        let got = fetch_one(&agent(), &format!("{CDN}/{CLASSIC_OOF_PKG}"), "uuhhh.mp3").and_then(|data| {
            let dir = dest.parent().ok_or("bad data dir")?;
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            let tmp = dir.join(".uuhhh.mp3.part");
            std::fs::write(&tmp, &data).map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())
        });
        if let Err(e) = got {
            post(true, format!("Couldn't fetch the classic death sound (Studio plays its replacement meanwhile): {e}"));
        }
    });
}

fn studio_version(version: &str) -> Option<&str> {
    version.strip_prefix("version-")
        .filter(|suffix| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_alphanumeric()))
        .map(|_| version)
}

fn has_view_selector(root: &Path) -> bool {
    root.join("StudioContent/models/ViewSelector/Basic.mesh").is_file()
        && ["back", "bottom", "front", "left", "right", "top"].iter().all(|face| {
            ["", "_hover"].iter().all(|suffix| {
                root.join(format!("StudioContent/textures/ViewSelector/{face}{suffix}.png")).is_file()
            })
        })
}

fn install_view_selector(agent: &ureq::Agent, version: &str, root: &Path) -> Result<(), String> {
    // Publish the body last so the renderer can pick up the complete cube.
    for kind in ["textures", "models"] {
        let url = format!("{CDN}/{version}-studiocontent-{kind}.zip");
        let bytes = get(agent, &url, None)?;
        let zip = Zip::parse(&bytes).map_err(|e| format!("{url}: {e}"))?;
        for entry in &zip.entries {
            let Some((dir, file)) = entry.name.split_once(['\\', '/']) else { continue };
            if dir != "ViewSelector" || file.is_empty() || file.contains(['\\', '/']) {
                continue;
            }
            let data = zip.read(&bytes, entry).map_err(|e| format!("{url} {}: {e}", entry.name))?;
            write_file(&root.join("StudioContent").join(kind), &entry.name, &data)?;
        }
    }
    has_view_selector(root).then_some(()).ok_or_else(|| "Studio packages are missing View Selector files".into())
}

const LUA_PACKAGE_REQUIRED_FILES: &[&str] = &[
    "Packages/_Index/BuilderIcons/BuilderIcons/BuilderIcons.json",
    "Packages/_Index/BuilderIcons/BuilderIcons/Font/BuilderIcons-Regular.ttf",
    "Packages/_Index/BuilderIcons/BuilderIcons/Font/BuilderIcons-Filled.ttf",
    "Packages/_Index/FoundationImages/FoundationImages/SpriteSheets/img_set_1x_4.png",
];

fn is_lua_package_asset(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".png")
        || lower.ends_with(".ttf")
        || lower.ends_with(".otf")
        || lower.ends_with(".dds")
        || lower.ends_with(".webp")
        || lower.ends_with("buildericons.json")
}

fn has_lua_package_assets(root: &Path) -> bool {
    LUA_PACKAGE_REQUIRED_FILES
        .iter()
        .all(|rel| root.join("ExtraContent/LuaPackages").join(rel).is_file())
}

fn install_lua_package_assets(agent: &ureq::Agent, version: &str, root: &Path) -> Result<(), String> {
    let url = format!("{CDN}/{version}-extracontent-luapackages.zip");
    let bytes = get(agent, &url, None)?;
    let zip = Zip::parse(&bytes).map_err(|e| format!("{url}: {e}"))?;
    let base = root.join("ExtraContent/LuaPackages");
    for e in &zip.entries {
        if e.name.ends_with(['/', '\\']) || !is_lua_package_asset(&e.name) {
            continue;
        }
        let data = zip.read(&bytes, e).map_err(|err| format!("{url} {}: {err}", e.name))?;
        write_file(&base, &e.name, &data)?;
    }
    has_lua_package_assets(root)
        .then_some(())
        .ok_or_else(|| "Studio packages are missing LuaPackages UI assets".into())
}

fn has_studio_ui_textures(root: &Path) -> bool {
    root.join("content/studio_svg_textures/Shared/Utility/Dark/Standard/RadioOn.png").is_file()
        && root.join("StudioContent/textures/StudioToolbox/RoundedBackground.png").is_file()
}

fn install_studio_ui_textures(agent: &ureq::Agent, version: &str, root: &Path) -> Result<(), String> {
    for (pkg, sub) in [
        ("content-studio_svg_textures.zip", "content/studio_svg_textures"),
        ("studiocontent-textures.zip", "StudioContent/textures"),
    ] {
        let url = format!("{CDN}/{version}-{pkg}");
        let bytes = get(agent, &url, None)?;
        let zip = Zip::parse(&bytes).map_err(|e| format!("{url}: {e}"))?;
        let base = root.join(sub);
        for e in &zip.entries {
            if e.name.ends_with(['/', '\\']) {
                continue;
            }
            let data = zip.read(&bytes, e).map_err(|err| format!("{url} {}: {err}", e.name))?;
            write_file(&base, &e.name, &data)?;
        }
    }
    has_studio_ui_textures(root)
        .then_some(())
        .ok_or_else(|| "Studio packages are missing Studio UI textures".into())
}

pub fn install(progress: &(dyn Fn(&str) + Sync)) -> Result<String, String> {
    let agent = agent();
    let v: serde_json::Value = serde_json::from_slice(&get(&agent, VERSION_URL, None)?)
        .map_err(|e| format!("version answer: {e}"))?;
    let version = v["clientVersionUpload"]
        .as_str()
        .and_then(studio_version)
        .ok_or("version answer has no clientVersionUpload")?
        .to_string();
    progress(&format!("Studio {} ({version})", v["version"].as_str().unwrap_or("?")));

    let dest = install_dir();
    if std::fs::read_to_string(dest.join(MARKER)).is_ok_and(|s| s.trim() == format!("{version} {CONTENT_REVISION}")) {
        install_missing_packages(&agent, &version, &dest, progress)?;
        if !has_view_selector(&dest) {
            progress("downloading Studio View Selector");
            install_view_selector(&agent, &version, &dest)?;
        }
        if !has_lua_package_assets(&dest) {
            progress("downloading Studio LuaPackages UI assets");
            install_lua_package_assets(&agent, &version, &dest)?;
        }
        if !has_studio_ui_textures(&dest) {
            progress("downloading Studio UI textures");
            install_studio_ui_textures(&agent, &version, &dest)?;
        }
        progress("already installed");
        return Ok(version);
    }
    let parent = dest.parent().ok_or("bad install dir")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    // The download goes on where an earlier run stopped (a phone's app is
    // killed when it leaves the screen for a while; a flaky connection): the
    // unpacked packages stay in one partial folder, each marked when whole,
    // until the whole content is there. Another Studio version starts over.
    let partial = parent.join(".studio-partial");
    let stamp = format!("{version} {CONTENT_REVISION}");
    if !std::fs::read_to_string(partial.join(".version")).is_ok_and(|s| s.trim() == stamp) {
        let _ = std::fs::remove_dir_all(&partial);
    }
    std::fs::create_dir_all(&partial).map_err(|e| e.to_string())?;
    std::fs::write(partial.join(".version"), format!("{stamp}\n")).map_err(|e| e.to_string())?;

    // The packages a few at a time (each is one request; on a phone's
    // connection the wait for each is mostly latency, and the biggest is
    // 18 MB, so three in flight stay well within memory).
    let result = (|| -> Result<(), String> {
        let total = PACKAGES.len();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let failed: Mutex<Option<String>> = Mutex::new(None);
        rt_thread::scope(|scope| {
            for _ in 0..PARALLEL_PACKAGES.min(total) {
                scope.spawn(|| loop {
                    let n = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let Some((pkg, sub)) = PACKAGES.get(n) else { return };
                    if failed.lock().map(|f| f.is_some()).unwrap_or(true) {
                        return;
                    }
                    let done = partial.join(format!(".done-{pkg}"));
                    if done.is_file() {
                        continue;
                    }
                    progress(&format!("downloading {pkg} ({}/{total})", n + 1));
                    if let Err(e) = install_package(&agent, &version, pkg, &partial.join(sub), &done) {
                        if let Ok(mut f) = failed.lock() {
                            f.get_or_insert(e);
                        }
                        return;
                    }
                });
            }
        });
        if let Some(e) = failed.lock().ok().and_then(|mut f| f.take()) {
            return Err(e);
        }
        // (the retired textures and the View Selector come after the
        // install: `install_extras`)
        if !has_lua_package_assets(&partial) {
            progress("downloading Studio LuaPackages UI assets");
            install_lua_package_assets(&agent, &version, &partial)?;
        }
        if !has_studio_ui_textures(&partial) {
            progress("downloading Studio UI textures");
            install_studio_ui_textures(&agent, &version, &partial)?;
        }
        for (pkg, file) in SINGLE_FILES {
            if partial.join(file).is_file() {
                continue;
            }
            progress(&format!("fetching {file} from {pkg}"));
            let url = format!("{CDN}/{version}-{pkg}");
            let data = fetch_one(&agent, &url, file)?;
            write_file(&partial, file, &data)?;
        }
        std::fs::write(partial.join(MARKER), format!("{stamp}\n")).map_err(|e| e.to_string())
    })();
    // (An error leaves the partial folder for the next attempt.)
    result?;
    if let Ok(rd) = std::fs::read_dir(&partial) {
        for e in rd.flatten() {
            let n = e.file_name();
            if n == ".version" || n.to_string_lossy().starts_with(".done-") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dest);
    std::fs::rename(&partial, &dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    progress(&format!("installed in {}", dest.display()));
    // Play has what it needs from here (character, player scripts, fonts,
    // textures). What follows is the editor's finish, which may take a
    // while on a slow connection and a hundred requests: it mustn't keep
    // Play waiting, and a failure in it is made good at the next start
    // (`spawn_background_install_if_missing` for the View Selector; a
    // retired texture is fetched by id when a place asks for it,
    // assets.rs `legacy_content`).
    progress("ready: Play now uses Roblox's default character and player scripts; the rest (retired textures, the View Selector) is fetched in the background");
    install_extras(&agent, &version, &dest, progress);
    Ok(version)
}

/// How many packages are downloaded at once.
const PARALLEL_PACKAGES: usize = 3;
/// How many retired textures are fetched at once.
const PARALLEL_LEGACY: usize = 2;

/// One package: downloaded, unpacked under `into`, and `done` written.
fn install_package(agent: &ureq::Agent, version: &str, pkg: &str, into: &Path, done: &Path) -> Result<(), String> {
    let url = format!("{CDN}/{version}-{pkg}");
    let bytes = get(agent, &url, None)?;
    let zip = Zip::parse(&bytes).map_err(|e| format!("{pkg}: {e}"))?;
    for e in &zip.entries {
        if e.name.ends_with(['/', '\\']) {
            continue;
        }
        let data = zip.read(&bytes, e).map_err(|err| format!("{pkg} {}: {err}", e.name))?;
        write_file(into, &e.name, &data)?;
    }
    std::fs::write(done, "").map_err(|e| e.to_string())
}

/// The editor's extras, into an installed `root`: the retired textures
/// Studio resolves through LegacyContentTable (materialized so file-based
/// consumers find them), and the View Selector cube. Best effort: what
/// fails is said and left for the next start.
fn install_extras(agent: &ureq::Agent, version: &str, root: &Path, progress: &(dyn Fn(&str) + Sync)) {
    let legacy: Vec<(&str, &str)> = include_str!("../../../assets/legacy-content.tsv")
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .filter(|(path, _)| path.starts_with("textures/") && !root.join("content").join(path).is_file())
        .collect();
    if !legacy.is_empty() {
        progress(&format!("fetching {} retired textures", legacy.len()));
        let next = std::sync::atomic::AtomicUsize::new(0);
        rt_thread::scope(|scope| {
            for _ in 0..PARALLEL_LEGACY.min(legacy.len()) {
                scope.spawn(|| loop {
                    let n = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let Some((path, id)) = legacy.get(n) else { return };
                    progress(&format!("fetching legacy {path}"));
                    let Ok(id) = id.trim().parse::<u64>() else {
                        progress(&format!("legacy texture {path}: bad id"));
                        continue;
                    };
                    match crate::assets::remote_cached(id) {
                        Ok(bytes) => {
                            if let Err(e) = write_file(&root.join("content"), path, &bytes) {
                                progress(&format!("legacy {path}: {e}"));
                            }
                        }
                        // Some retired entries have also been removed from
                        // asset delivery; a place that wants one asks by id.
                        Err(e) => progress(&format!("legacy {path} unavailable: {e}")),
                    }
                });
            }
        });
    }
    if !has_view_selector(root) {
        progress("downloading Studio View Selector");
        if let Err(e) = install_view_selector(agent, version, root) {
            progress(&format!("couldn't download Studio's View Selector ({e}); the next start tries again"));
        }
    }
}

/// Write `name` (a zip path, `\` or `/` separated) under `root`, refusing
/// anything that would land outside it.
fn write_file(root: &Path, name: &str, data: &[u8]) -> Result<(), String> {
    let mut path = root.to_path_buf();
    for part in name.split(['\\', '/']) {
        match part {
            "" | "." => {}
            ".." => return Err(format!("unsafe path in package: {name}")),
            p if p.contains(':') => return Err(format!("unsafe path in package: {name}")),
            p => path.push(p),
        }
    }
    if path == root {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&path, data).map_err(|e| format!("{}: {e}", path.display()))
}

/// One file out of a remote zip: the end record, the central directory and
/// that one entry, by range requests.
fn fetch_one(agent: &ureq::Agent, url: &str, file: &str) -> Result<Vec<u8>, String> {
    let mut out = fetch_many(agent, url, &[file])?;
    Ok(out.remove(0).1)
}

/// Several files out of a remote zip, reading the central directory once.
fn fetch_many<'a>(agent: &ureq::Agent, url: &str, files: &[&'a str]) -> Result<Vec<(&'a str, Vec<u8>)>, String> {
    let size = content_length(agent, url)?;
    let tail_len = size.min(65_536 + 22);
    let tail = get(agent, url, Some((size - tail_len, size - 1)))?;
    let (cd_off, cd_len) = Zip::end_record(&tail).map_err(|e| format!("{url}: {e}"))?;
    let cd = get(agent, url, Some((cd_off, cd_off + cd_len - 1)))?;
    let entries = Zip::central(&cd).map_err(|e| format!("{url}: {e}"))?;
    let mut out = Vec::with_capacity(files.len());
    for &file in files {
        let want = file.trim_start_matches(['\\', '/']).replace('\\', "/");
        let e = entries
            .iter()
            .find(|e| {
                e.name
                    .trim_start_matches(['\\', '/'])
                    .replace('\\', "/")
                    .eq_ignore_ascii_case(&want)
            })
            .ok_or_else(|| format!("{file} not in {url}"))?;
        // Local header (30 bytes + name + extra) then the data; read generously.
        let span = 30 + 1024 + e.comp_size;
        let chunk = get(agent, url, Some((e.offset, (e.offset + span).min(size) - 1)))?;
        let mut local = e.clone();
        local.offset = 0;
        out.push((file, Zip::read_entry(&chunk, &local)?));
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct Entry {
    name: String,
    method: u16,
    crc: u32,
    comp_size: u64,
    size: u64,
    offset: u64,
}

/// A minimal zip reader (stored + deflate, no zip64): enough for Roblox's
/// deployment packages.
struct Zip {
    entries: Vec<Entry>,
}

fn u16_at(b: &[u8], i: usize) -> Result<u16, String> {
    b.get(i..i + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or_else(|| "truncated zip".into())
}
fn u32_at(b: &[u8], i: usize) -> Result<u32, String> {
    b.get(i..i + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| "truncated zip".into())
}

impl Zip {
    fn parse(bytes: &[u8]) -> Result<Zip, String> {
        let (off, len) = Self::end_record(bytes)?;
        let cd = bytes
            .get(off as usize..(off + len) as usize)
            .ok_or("central directory out of range")?;
        Ok(Zip { entries: Self::central(cd)? })
    }

    /// (central directory offset, length) from the tail of a zip.
    fn end_record(tail: &[u8]) -> Result<(u64, u64), String> {
        let pos = tail
            .windows(4)
            .rposition(|w| w == b"PK\x05\x06")
            .ok_or("no zip end record")?;
        let len = u32_at(tail, pos + 12)? as u64;
        let off = u32_at(tail, pos + 16)? as u64;
        if len == 0xFFFF_FFFF || off == 0xFFFF_FFFF {
            return Err("zip64 not supported".into());
        }
        Ok((off, len))
    }

    fn central(cd: &[u8]) -> Result<Vec<Entry>, String> {
        let mut out = Vec::new();
        let mut p = 0;
        while p + 46 <= cd.len() && &cd[p..p + 4] == b"PK\x01\x02" {
            let method = u16_at(cd, p + 10)?;
            let crc = u32_at(cd, p + 16)?;
            let comp_size = u32_at(cd, p + 20)? as u64;
            let size = u32_at(cd, p + 24)? as u64;
            let nl = u16_at(cd, p + 28)? as usize;
            let el = u16_at(cd, p + 30)? as usize;
            let cl = u16_at(cd, p + 32)? as usize;
            let offset = u32_at(cd, p + 42)? as u64;
            let name = cd.get(p + 46..p + 46 + nl).ok_or("truncated zip")?;
            out.push(Entry {
                name: String::from_utf8_lossy(name).into_owned(),
                method,
                crc,
                comp_size,
                size,
                offset,
            });
            p += 46 + nl + el + cl;
        }
        Ok(out)
    }

    fn read(&self, bytes: &[u8], e: &Entry) -> Result<Vec<u8>, String> {
        Self::read_entry(bytes, e)
    }

    /// The entry's data, given bytes where its local header starts at `e.offset`.
    fn read_entry(bytes: &[u8], e: &Entry) -> Result<Vec<u8>, String> {
        let h = e.offset as usize;
        if bytes.get(h..h + 4) != Some(b"PK\x03\x04") {
            return Err("bad local header".into());
        }
        let nl = u16_at(bytes, h + 26)? as usize;
        let el = u16_at(bytes, h + 28)? as usize;
        let start = h + 30 + nl + el;
        let comp = bytes
            .get(start..start + e.comp_size as usize)
            .ok_or("entry data out of range")?;
        let data = match e.method {
            0 => comp.to_vec(),
            8 => {
                let mut out = Vec::with_capacity(e.size as usize);
                flate2::read::DeflateDecoder::new(comp)
                    .read_to_end(&mut out)
                    .map_err(|err| format!("inflate: {err}"))?;
                out
            }
            m => return Err(format!("compression method {m} not supported")),
        };
        let mut crc = flate2::Crc::new();
        crc.update(&data);
        if data.len() as u64 != e.size || crc.sum() != e.crc {
            return Err("checksum mismatch".into());
        }
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_escaping_paths() {
        let d = (if cfg!(target_arch = "wasm32") { std::path::PathBuf::from("/tmp") } else { std::env::temp_dir() }).join(format!("rs-sc-test-{}", (if cfg!(target_arch = "wasm32") { 1u32 } else { std::process::id() })));
        assert!(write_file(&d, "..\\evil.txt", b"x").is_err());
        assert!(write_file(&d, "a/../../evil.txt", b"x").is_err());
        assert!(write_file(&d, "C:\\evil.txt", b"x").is_err());
        assert!(write_file(&d, "\\sub\\ok.txt", b"x").is_ok());
        assert!(d.join("sub/ok.txt").is_file());
        let _ = std::fs::remove_dir_all(&d);
    }
}
