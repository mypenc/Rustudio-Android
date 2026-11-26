//! The system clipboard. egui-winit has no clipboard backend on Android
//! (its copies stay in a string inside the process), so copies are
//! mirrored to `ClipboardManager` after each frame and pastes take its
//! text ([`jni::clipboard_set`](super::jni::clipboard_set) /
//! [`jni::clipboard_get_later`](super::jni::clipboard_get_later)).
//!
//! Both go through the Java main thread without the editor's thread
//! waiting: a copy is sent and forgotten; the clipboard's text is asked
//! for ahead of the paste ([`refresh`] — when a text field gets the
//! keyboard, when the window comes back into focus, when the text menu or
//! bar opens) and kept here ([`text`]) for the frame it is wanted in. The
//! text the editor copied is kept too, so Paste follows a Copy at once.
//! A paste reads the clipboard at that moment as well ([`text`] →
//! `jni::clipboard_get_now`, on the editor's thread: any thread may, see
//! there), so what was copied in another app a second ago is what pastes;
//! the asked-ahead answer is the fallback when that read fails.
//!
//! What never reaches the editor: the soft keyboard's own paste key. A
//! NativeActivity has no `InputConnection`, so the keyboard is in the
//! framework's fallback mode, where a committed string longer than one
//! character becomes a `KeyEvent.ACTION_MULTIPLE` whose characters only
//! the Java object carries — nothing of them reaches native code (that
//! is also why the keyboard offers no word suggestions here). Hence the
//! editor's own Paste: the bar above the keyboard (text_bar.rs) and the
//! text fields' long-press menu (text_menu.rs).
//!
//! egui-winit swallows the paste shortcut while its own string is empty
//! (no `Event::Paste` at all), so the first frame gives it a marker text:
//! every paste then arrives as `Paste(marker)` and is swapped for the
//! system clipboard's text here (`before_frame` runs before the menu's and
//! the bar's, whose pastes come with their text). Every read and write is
//! logged (`adb logcat -s rustudio`: "clipboard: ..."); Android 10+ lets
//! only the focused app read.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use web_time::Instant;

use super::jni::ClipboardAnswer;

/// egui-winit's stand-in clipboard text (never shown, never mirrored).
const SEED: &str = "\u{200b}rustudio-clipboard-seed\u{200b}";
/// A question the main thread hasn't answered in this long is given up
/// on (it is busy with a pause or a configuration change) and asked again.
const GIVE_UP: Duration = Duration::from_secs(2);

struct Cache {
    /// The clipboard's text as last read, or as last put there by the
    /// editor (`None`: none, or not read yet).
    text: Option<String>,
    /// The question out to the main thread, and when it was asked.
    asked: Option<(Instant, ClipboardAnswer)>,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache { text: None, asked: None });
static SEEDED: AtomicBool = AtomicBool::new(false);

fn cache() -> std::sync::MutexGuard<'static, Cache> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Asks the main thread for the clipboard's text, unless a question is
/// already out. The answer shows in [`text`] once [`poll`] has taken it
/// (`before_frame` does, every frame).
pub fn refresh() {
    let mut c = cache();
    if let Some((when, _)) = &c.asked {
        if when.elapsed() < GIVE_UP {
            return;
        }
    }
    c.asked = Some((Instant::now(), super::jni::clipboard_get_later()));
}

/// Takes the last question's answer, once it has come.
pub fn poll() {
    let mut c = cache();
    let Some((_, slot)) = &c.asked else { return };
    let answer = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
    match answer {
        Some(Ok(text)) => {
            c.asked = None;
            c.text = text;
        }
        Some(Err(e)) => {
            c.asked = None;
            log::warn!("clipboard: {e}");
        }
        None => {}
    }
}

/// The clipboard's text as last known (a read's answer, or the editor's
/// own last copy), if any. Cheap: for a Paste button's enabled state,
/// every frame. The paste itself takes [`text_now`].
pub fn text() -> Option<String> {
    cache().text.clone()
}

/// The clipboard's text at this moment: read from the system clipboard
/// now (what the user copied last, in whichever app), else — when the
/// read itself fails, not when the clipboard is empty — as last known. One
/// JNI call: for the moment of a paste, and [`prime`], not every frame.
pub fn text_now() -> Option<String> {
    match super::jni::clipboard_get_now() {
        Ok(now) => {
            let mut c = cache();
            if now.is_some() {
                c.text = now.clone();
            }
            // (empty, or not ours to read: the editor's own last copy
            // still pastes within the editor)
            now.or_else(|| c.text.clone())
        }
        Err(_) => cache().text.clone(),
    }
}

/// When a text menu or bar opens: the clipboard's text read now, so Paste
/// is enabled at once; the main thread is asked instead if the read here
/// fails.
pub fn prime() {
    match super::jni::clipboard_get_now() {
        Ok(Some(s)) => cache().text = Some(s),
        Ok(None) => {}
        Err(_) => refresh(),
    }
}

/// Puts `text` on the system clipboard and remembers it as the
/// clipboard's text.
pub fn set(text: String) {
    if text.is_empty() {
        return;
    }
    if let Err(e) = super::jni::clipboard_set(&text) {
        log::warn!("clipboard: {e}");
    }
    cache().text = Some(text);
}

/// In `App::raw_input_hook`: takes a read's answer; a paste of
/// egui-winit's (a hardware keyboard's Ctrl+V) takes the system
/// clipboard's text as read at that moment, when it has one — the marker
/// alone (nothing ever copied) pastes nothing; the window coming back into
/// focus (the user may have copied in another app) asks for the text
/// again.
pub fn before_frame(raw: &mut egui::RawInput) {
    poll();
    if raw.events.iter().any(|e| matches!(e, egui::Event::WindowFocused(true))) {
        refresh();
    }
    if !raw.events.iter().any(|e| matches!(e, egui::Event::Paste(_))) {
        return;
    }
    let system = text_now();
    raw.events.retain_mut(|e| {
        let egui::Event::Paste(text) = e else { return true };
        match &system {
            Some(s) => {
                *text = s.clone();
                true
            }
            None => text.as_str() != SEED,
        }
    });
    // (for the next one: what is there now)
    refresh();
}

/// At the end of `App::update`: text egui was asked to copy this frame
/// (`Context::copy_text`, Ctrl+C in a text field, Explorer copies) goes to
/// the system clipboard as well.
pub fn after_frame(ctx: &egui::Context) {
    if !SEEDED.swap(true, Ordering::SeqCst) {
        ctx.copy_text(SEED.to_string());
        return;
    }
    let copied: Option<String> = ctx.output(|o| {
        o.commands.iter().rev().find_map(|c| match c {
            egui::OutputCommand::CopyText(s) if !s.is_empty() && s != SEED => Some(s.clone()),
            _ => None,
        })
    });
    let Some(text) = copied else { return };
    if cache().text.as_deref() == Some(text.as_str()) {
        return;
    }
    set(text);
}
