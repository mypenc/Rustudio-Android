//! Per-user folders. Linux (and the other XDG Unixes) keeps the layout
//! rustudio has always used (`~/.local/share`, `~/.config`, `~/.local/state`,
//! `~/.cache`); Windows and macOS use their own folders through `dirs` (on
//! Windows `%LOCALAPPDATA%` for data, state and cache, `%APPDATA%` for
//! config; on macOS `~/Library/Application Support` for data, config and
//! state, `~/Library/Caches` for caches), the same config folder
//! rustudio-account uses. Android has no home folder: the app's own
//! folders stand in (`Context.getFilesDir()` for config and state,
//! `getExternalFilesDir()` for data, `getCacheDir()` for caches;
//! src/android/mod.rs).

use std::path::PathBuf;

/// The user's home folder ("." when it can't be found).
pub fn home() -> PathBuf {
    #[cfg(target_os = "android")]
    return crate::android_dirs::files_dir();
    #[cfg(all(unix, not(target_os = "android")))]
    return std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    #[cfg(not(unix))]
    return dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
}

/// Linux and the other XDG Unixes (not macOS, not Android).
#[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
fn xdg(var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(fallback))
}

/// Shared data: `~/.local/share` on Linux. Not `$XDG_DATA_HOME`, so files
/// rustudio already put there (the content link, the Studio download) keep
/// being found.
pub fn data() -> PathBuf {
    #[cfg(target_os = "android")]
    return crate::android_dirs::data_dir();
    #[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
    return home().join(".local/share");
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    return dirs::data_local_dir().unwrap_or_else(home);
}

/// Settings: `$XDG_CONFIG_HOME` or `~/.config` on Linux.
pub fn config() -> PathBuf {
    #[cfg(target_os = "android")]
    return crate::android_dirs::config_dir();
    #[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
    return xdg("XDG_CONFIG_HOME", ".config");
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    return dirs::config_dir().unwrap_or_else(home);
}

/// State that outlives a run: `$XDG_STATE_HOME` or `~/.local/state` on Linux.
pub fn state() -> PathBuf {
    #[cfg(target_os = "android")]
    return crate::android_dirs::state_dir();
    #[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
    return xdg("XDG_STATE_HOME", ".local/state");
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    return dirs::data_local_dir().unwrap_or_else(home);
}

/// Caches: `$XDG_CACHE_HOME` or `~/.cache` on Linux.
pub fn cache() -> PathBuf {
    #[cfg(target_os = "android")]
    return crate::android_dirs::cache_dir();
    #[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
    return xdg("XDG_CACHE_HOME", ".cache");
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    return dirs::cache_dir().unwrap_or_else(|| if cfg!(target_arch = "wasm32") { std::path::PathBuf::from("/tmp") } else { std::env::temp_dir() });
}

/// Where screenshots go: `~/Pictures` on Linux, the Pictures folder elsewhere.
pub fn pictures() -> PathBuf {
    #[cfg(target_os = "android")]
    return crate::android_dirs::pictures_dir();
    #[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
    return home().join("Pictures");
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    return dirs::picture_dir().unwrap_or_else(|| home().join("Pictures"));
}

/// Roblox Studio installed on this Mac (`RobloxStudio.app` in /Applications
/// or ~/Applications): the folder holding `content/`, `ExtraContent/` and
/// `PlatformContent/`, laid out like the Windows install and the CDN
/// packages. None on other systems.
pub fn installed_studio() -> Option<PathBuf> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    [PathBuf::from("/Applications"), home().join("Applications")]
        .into_iter()
        .map(|a| a.join("RobloxStudio.app/Contents/Resources"))
        .find(|r| r.join("ExtraContent/scripts/PlayerScripts").is_dir() && r.join("content").is_dir())
}

/// Newest local Windows Roblox Studio install under `%LOCALAPPDATA%\Roblox\Versions`
/// that contains `BuiltInPlugins/Optimized_Embedded_Signature`. Kept separate
/// from [`installed_studio`] so `default_scripts::studio_dir()` continues to
/// prefer rustudio's up-to-date CDN download over an older local install.
pub fn windows_studio_install() -> Option<PathBuf> {
    if !cfg!(target_os = "windows") {
        return None;
    }
    let versions = data().join("Roblox/Versions");
    let rd = std::fs::read_dir(versions).ok()?;
    rd.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|r| r.join("BuiltInPlugins/Optimized_Embedded_Signature").is_dir())
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
}

/// A data file that ships with rustudio (`api-dump.json`), wherever this
/// build keeps it: next to the executable, in a macOS app bundle's
/// `Contents/Resources`, in `../share/rustudio` (an install prefix), the
/// Android app's unpacked assets, else the source tree it was built from
/// (`cargo run`).
pub fn resource(name: &str) -> PathBuf {
    #[cfg(target_os = "android")]
    {
        let p = crate::android_dirs::resource_dir().join(name);
        if p.is_file() {
            return p;
        }
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.canonicalize().ok())
        .and_then(|e| e.parent().map(std::path::Path::to_path_buf));
    if let Some(dir) = exe_dir {
        for p in [dir.join(name), dir.join("../Resources").join(name), dir.join("../share/rustudio").join(name)] {
            if p.is_file() {
                return p;
            }
        }
    }
    // (crates/rustudio-core: the root crate's folder, where api-dump.json is)
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folders_per_platform() {
        if cfg!(target_os = "macos") {
            let lib = home().join("Library");
            assert_eq!(data(), lib.join("Application Support"));
            assert_eq!(config(), lib.join("Application Support"));
            assert_eq!(state(), lib.join("Application Support"));
            assert_eq!(cache(), lib.join("Caches"));
        } else if cfg!(all(unix, not(target_os = "android"))) {
            assert_eq!(data(), home().join(".local/share"));
        }
    }

    #[test]
    fn api_dump_found() {
        assert!(resource("api-dump.json").is_file());
    }
}
