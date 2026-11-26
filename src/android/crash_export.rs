//! Crash report `.txt` export via Android Storage Access Framework (SAF)
//! `ACTION_CREATE_DOCUMENT`, with automatic `Download/` MediaStore persistence
//! and desktop `rfd` fallback.
//!
//! Because `NativeActivity` does not forward `onActivityResult` to native code
//! unless subclassed, on Android this module:
//! 1. Persists a guaranteed complete copy into `Downloads/` (via
//!    `MediaStore.Downloads` on API 29+, or filesystem `Download/` / app
//!    crashes folder fallback) so the crash text is never lost.
//! 2. Launches Android SAF `Intent(ACTION_CREATE_DOCUMENT)` with MIME type
//!    `text/plain` and `EXTRA_TITLE` so the user can also pick a custom
//!    destination. If a custom Activity forwards the created `content://` URI to
//!    [`on_saf_document_created`], the content is written directly to that URI.
//!
//! On desktop (`#[cfg(not(target_os = "android"))]`), spawns a non-blocking
//! `rfd::AsyncFileDialog` save dialog for `.txt`.

use std::sync::Mutex;

/// Result / status update from a crash report export attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportStatus {
    /// Export succeeded (message describes where the file was saved or picker state).
    Success(String),
    /// User cancelled the save dialog.
    Cancelled,
    /// Export failed with a user-readable error message.
    Error(String),
}

/// Request code used for `startActivityForResult(ACTION_CREATE_DOCUMENT, ...)`.
pub const SAF_CREATE_DOC_REQUEST_CODE: i32 = 0x5253; // "RS"

static LAST_STATUS: Mutex<Option<ExportStatus>> = Mutex::new(None);
static PENDING_SAF_CONTENT: Mutex<Option<String>> = Mutex::new(None);

fn set_status(status: ExportStatus) {
    if let Ok(mut g) = LAST_STATUS.lock() {
        *g = Some(status);
    }
}

/// Polls and clears the latest export status (for the crash dialog status banner).
pub fn poll_export_status() -> Option<ExportStatus> {
    LAST_STATUS.lock().ok().and_then(|mut g| g.take())
}

/// Ensures the suggested filename is safe and ends with `.txt`.
fn sanitize_txt_name(suggested_name: &str) -> String {
    let mut clean: String = suggested_name
        .trim()
        .chars()
        .map(|c| {
            if c == '/' || c == '\\' || c == ':' || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    if clean.is_empty() {
        clean = format!(
            "rustudio-crash-{}.txt",
            chrono::Local::now().format("%Y%m%d-%H%M%S")
        );
    }
    if !clean.to_ascii_lowercase().ends_with(".txt") {
        clean.push_str(".txt");
    }
    clean
}

/// Exports `content` as a `.txt` file using Android SAF `ACTION_CREATE_DOCUMENT`
/// (`text/plain`), with `Download/` fallback on Android and `rfd` save dialog on desktop.
///
/// Updates [`poll_export_status`] and also returns the immediate [`ExportStatus`]
/// (on desktop, returns `Success("Opening save dialog...")` immediately and updates
/// [`poll_export_status`] once the user finishes choosing a file).
pub fn export_crash_txt_saf(suggested_name: &str, content: String) -> ExportStatus {
    let file_name = sanitize_txt_name(suggested_name);

    #[cfg(target_os = "android")]
    {
        export_android(&file_name, content)
    }

    #[cfg(not(target_os = "android"))]
    {
        export_desktop(file_name, content)
    }
}

#[cfg(target_os = "android")]
fn export_android(file_name: &str, content: String) -> ExportStatus {
    // Store content in case a Java Activity subclass forwards onActivityResult
    // to `on_saf_document_created(uri)`.
    if let Ok(mut g) = PENDING_SAF_CONTENT.lock() {
        *g = Some(content.clone());
    }

    // 1. Write a guaranteed copy to Downloads (MediaStore on API 29+, or direct path).
    let saved_copy = save_to_downloads_android(file_name, content.as_bytes());

    // 2. Launch Android SAF ACTION_CREATE_DOCUMENT ("text/plain").
    let saf_launch = launch_saf_create_document(file_name);

    let status = match (saf_launch, saved_copy) {
        (Ok(()), Ok(location)) => ExportStatus::Success(format!(
            "Opened system Save picker for {file_name} (copy saved to {location})."
        )),
        (Ok(()), Err(copy_err)) => {
            log::warn!("crash_export: fallback Download copy failed: {copy_err}");
            ExportStatus::Success(format!("Opened system Save picker for {file_name}."))
        }
        (Err(saf_err), Ok(location)) => {
            log::warn!("crash_export: SAF ACTION_CREATE_DOCUMENT failed ({saf_err}); using Download copy");
            ExportStatus::Success(format!("Saved crash report to {location}."))
        }
        (Err(saf_err), Err(copy_err)) => {
            // Last-resort fallback inside app filesDir/crashes/exports/
            let fallback_dir = super::data_dir().join("crashes");
            let fallback_path = fallback_dir.join(file_name);
            if std::fs::create_dir_all(&fallback_dir).is_ok()
                && std::fs::write(&fallback_path, content.as_bytes()).is_ok()
            {
                ExportStatus::Success(format!(
                    "Saved crash report to {} (SAF unavailable: {saf_err}).",
                    fallback_path.display()
                ))
            } else {
                ExportStatus::Error(format!(
                    "Export failed — SAF error: {saf_err}; Downloads error: {copy_err}"
                ))
            }
        }
    };

    set_status(status.clone());
    status
}

/// Launches `Intent(Intent.ACTION_CREATE_DOCUMENT)` with `CATEGORY_OPENABLE`,
/// `setType("text/plain")`, and `EXTRA_TITLE = file_name`.
#[cfg(target_os = "android")]
fn launch_saf_create_document(file_name: &str) -> super::jni::Result<()> {
    use jni::objects::JValue;
    use jni::{jni_sig, jni_str};

    super::jni::with_activity(|env, activity| {
        let action = env.new_string("android.intent.action.CREATE_DOCUMENT")?;
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
        let mime = env.new_string("text/plain")?;
        env.call_method(
            &intent,
            jni_str!("setType"),
            jni_sig!("(Ljava/lang/String;)Landroid/content/Intent;"),
            &[JValue::Object(&mime)],
        )?;
        let extra_title_key = env.new_string("android.intent.extra.TITLE")?;
        let extra_title_val = env.new_string(file_name)?;
        env.call_method(
            &intent,
            jni_str!("putExtra"),
            jni_sig!("(Ljava/lang/String;Ljava/lang/String;)Landroid/content/Intent;"),
            &[JValue::Object(&extra_title_key), JValue::Object(&extra_title_val)],
        )?;
        env.call_method(
            activity,
            jni_str!("startActivityForResult"),
            jni_sig!("(Landroid/content/Intent;I)V"),
            &[
                JValue::Object(&intent),
                JValue::Int(SAF_CREATE_DOC_REQUEST_CODE),
            ],
        )?;
        Ok(())
    })
}

/// Optional callback hook for a custom Java Activity's `onActivityResult` to
/// write the pending crash text directly to the SAF `content://` URI.
#[cfg(target_os = "android")]
pub fn on_saf_document_created(uri_str: &str) -> super::jni::Result<String> {
    let content = PENDING_SAF_CONTENT
        .lock()
        .ok()
        .and_then(|mut g| g.take())
        .ok_or_else(|| "no pending crash export content".to_string())?;
    write_content_uri(uri_str, content.as_bytes())?;
    let msg = format!("Exported crash report to {uri_str}");
    set_status(ExportStatus::Success(msg.clone()));
    Ok(msg)
}

#[cfg(target_os = "android")]
fn write_content_uri(uri_str: &str, bytes: &[u8]) -> super::jni::Result<()> {
    use jni::objects::JValue;
    use jni::{jni_sig, jni_str};

    super::jni::with_activity(|env, activity| {
        let juri = env.new_string(uri_str)?;
        let uri = env
            .call_static_method(
                jni_str!("android/net/Uri"),
                jni_str!("parse"),
                jni_sig!("(Ljava/lang/String;)Landroid/net/Uri;"),
                &[JValue::Object(&juri)],
            )?
            .l()?;
        let resolver = env
            .call_method(
                activity,
                jni_str!("getContentResolver"),
                jni_sig!("()Landroid/content/ContentResolver;"),
                &[],
            )?
            .l()?;
        let stream = env
            .call_method(
                &resolver,
                jni_str!("openOutputStream"),
                jni_sig!("(Landroid/net/Uri;)Ljava/io/OutputStream;"),
                &[JValue::Object(&uri)],
            )?
            .l()?;
        if stream.is_null() {
            return Err(jni::errors::Error::NullPtr("openOutputStream returned null"));
        }
        let signed: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        let jbytes = env.new_byte_array(signed.len())?;
        jbytes.set_region(env, 0, &signed)?;
        env.call_method(
            &stream,
            jni_str!("write"),
            jni_sig!("([B)V"),
            &[JValue::Object(&jbytes)],
        )?;
        env.call_method(&stream, jni_str!("flush"), jni_sig!("()V"), &[])?;
        env.call_method(&stream, jni_str!("close"), jni_sig!("()V"), &[])?;
        Ok(())
    })
}

/// Writes `bytes` into the public `Download/` folder via `MediaStore.Downloads`
/// (API 29+, no storage permission required), falling back to `/storage/emulated/0/Download`.
#[cfg(target_os = "android")]
pub fn save_to_downloads_android(file_name: &str, bytes: &[u8]) -> super::jni::Result<String> {
    if super::jni::sdk_int() >= 29 {
        if let Ok(desc) = write_via_mediastore_downloads(file_name, bytes) {
            return Ok(desc);
        }
    }
    let dl_path = std::path::PathBuf::from("/storage/emulated/0/Download").join(file_name);
    std::fs::write(&dl_path, bytes).map_err(|e| format!("{}: {e}", dl_path.display()))?;
    Ok(format!("Download/{file_name}"))
}

#[cfg(target_os = "android")]
fn write_via_mediastore_downloads(file_name: &str, bytes: &[u8]) -> super::jni::Result<String> {
    use jni::objects::JValue;
    use jni::{jni_sig, jni_str};

    super::jni::with_activity(|env, activity| {
        let cv = env.new_object(jni_str!("android/content/ContentValues"), jni_sig!("()V"), &[])?;
        let put_str = |env: &mut jni::Env, k: &str, v: &str| -> jni::errors::Result<()> {
            let jk = env.new_string(k)?;
            let jv = env.new_string(v)?;
            env.call_method(
                &cv,
                jni_str!("put"),
                jni_sig!("(Ljava/lang/String;Ljava/lang/String;)V"),
                &[JValue::Object(&jk), JValue::Object(&jv)],
            )?;
            Ok(())
        };
        put_str(env, "_display_name", file_name)?;
        put_str(env, "mime_type", "text/plain")?;
        put_str(env, "relative_path", "Download/Rustudio/")?;

        let downloads_uri = env
            .get_static_field(
                jni_str!("android/provider/MediaStore$Downloads"),
                jni_str!("EXTERNAL_CONTENT_URI"),
                jni_sig!("Landroid/net/Uri;"),
            )?
            .l()?;
        let resolver = env
            .call_method(
                activity,
                jni_str!("getContentResolver"),
                jni_sig!("()Landroid/content/ContentResolver;"),
                &[],
            )?
            .l()?;
        let inserted = env
            .call_method(
                &resolver,
                jni_str!("insert"),
                jni_sig!("(Landroid/net/Uri;Landroid/content/ContentValues;)Landroid/net/Uri;"),
                &[JValue::Object(&downloads_uri), JValue::Object(&cv)],
            )?
            .l()?;
        if inserted.is_null() {
            return Err(jni::errors::Error::NullPtr("MediaStore.Downloads.insert returned null"));
        }
        let stream = env
            .call_method(
                &resolver,
                jni_str!("openOutputStream"),
                jni_sig!("(Landroid/net/Uri;)Ljava/io/OutputStream;"),
                &[JValue::Object(&inserted)],
            )?
            .l()?;
        if stream.is_null() {
            return Err(jni::errors::Error::NullPtr("openOutputStream returned null"));
        }
        let signed: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        let jbytes = env.new_byte_array(signed.len())?;
        jbytes.set_region(env, 0, &signed)?;
        env.call_method(
            &stream,
            jni_str!("write"),
            jni_sig!("([B)V"),
            &[JValue::Object(&jbytes)],
        )?;
        env.call_method(&stream, jni_str!("flush"), jni_sig!("()V"), &[])?;
        env.call_method(&stream, jni_str!("close"), jni_sig!("()V"), &[])?;
        Ok(format!("Download/Rustudio/{file_name}"))
    })
}

#[cfg(not(target_os = "android"))]
fn export_desktop(file_name: String, content: String) -> ExportStatus {
    let dialog = rfd::AsyncFileDialog::new()
        .set_title("Export Crash Report")
        .set_file_name(&file_name)
        .add_filter("Text files", &["txt"]);

    let spawned = rt_thread::Builder::new()
        .name("crash-export".into())
        .spawn(move || {
            let handle = pollster::block_on(dialog.save_file());
            match handle {
                Some(h) => {
                    let path = h.path().to_path_buf();
                    match std::fs::write(&path, content.as_bytes()) {
                        Ok(()) => set_status(ExportStatus::Success(format!(
                            "Saved crash report to {}",
                            path.display()
                        ))),
                        Err(e) => set_status(ExportStatus::Error(format!(
                            "Failed to write {}: {e}",
                            path.display()
                        ))),
                    }
                }
                None => set_status(ExportStatus::Cancelled),
            }
        });

    match spawned {
        Ok(_) => ExportStatus::Success("Choose a location to save the crash report...".to_string()),
        Err(e) => {
            let status = ExportStatus::Error(format!("Failed to open save dialog: {e}"));
            set_status(status.clone());
            status
        }
    }
}
