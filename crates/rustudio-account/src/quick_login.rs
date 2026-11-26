//! Sign in through the default browser with Roblox's Quick Sign-in (the
//! "cross-device login" the Roblox app uses): rustudio asks for a short code,
//! the person enters it at [`CONFIRM_URL`] in any browser signed in to
//! Roblox, and once they confirm, the code and its private key are traded for
//! a `.ROBLOSECURITY` session of rustudio's own. The browser's own cookie is
//! never read.
//!
//! The editor may not live through the browser visit: on Android the system
//! can stop the process while the browser is in front. [`QuickLogin::remember`]
//! keeps the pending code in a file next to `account.toml` and
//! [`QuickLogin::pending`] hands it back to the next start, which goes on
//! polling as if nothing happened.

use std::path::PathBuf;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{session::session_agent, AccountConfig, Result};

/// Where the person types the code (roblox.com asks them to log in first if
/// that browser isn't signed in).
pub const CONFIRM_URL: &str = "https://www.roblox.com/crossdevicelogin/ConfirmCode";

const CREATE: &str = "https://apis.roblox.com/auth-token-service/v1/login/create";
const STATUS: &str = "https://apis.roblox.com/auth-token-service/v1/login/status";
const LOGIN: &str = "https://auth.roblox.com/v2/login";
const COOKIE: &str = ".ROBLOSECURITY";

/// A pending Quick Sign-in. `private_key` proves this client asked for the
/// code; it goes nowhere but to Roblox and (while the code is pending) the
/// private file [`QuickLogin::remember`] writes.
pub struct QuickLogin {
    pub code: String,
    private_key: String,
}

/// Where a pending code waits for the next start, next to `account.toml`.
const PENDING_FILE: &str = "quick-login.json";

/// A pending code older than this isn't resumed; Roblox's codes don't last
/// that long anyway (an unknown code reads as expired).
const PENDING_FOR: Duration = Duration::from_secs(10 * 60);

impl std::fmt::Debug for QuickLogin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuickLogin").field("code", &self.code).finish_non_exhaustive()
    }
}

/// Where the code is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuickStatus {
    /// Waiting for someone to enter it.
    Created,
    /// Entered; the browser is asking that account to confirm.
    UserLinked(Option<String>),
    /// Confirmed: [`QuickLogin::finish`] gets the session.
    Validated,
    /// Turned down in the browser.
    Cancelled,
    /// Ran out of time (codes last a few minutes).
    Expired,
}

fn err(msg: impl Into<String>) -> crate::Error {
    crate::Error(msg.into())
}

fn post(agent: &ureq::Agent, url: &str, body: &serde_json::Value, csrf: Option<&str>) -> Result<ureq::http::Response<ureq::Body>> {
    let mut req = agent.post(url).header("Content-Type", "application/json").header("Accept", "application/json");
    if let Some(t) = csrf {
        req = req.header("x-csrf-token", t);
    }
    Ok(req.send(body.to_string().as_bytes())?)
}

/// POSTs `body`, retrying once with the `x-csrf-token` a 403 hands back
/// (every one of these endpoints wants one, signed in or not).
fn call(agent: &ureq::Agent, url: &str, body: &serde_json::Value) -> Result<(u16, String)> {
    let r = post(agent, url, body, None)?;
    let token = r.headers().get("x-csrf-token").and_then(|v| v.to_str().ok()).map(str::to_owned);
    match token {
        Some(t) if r.status().as_u16() == 403 => Ok(read(post(agent, url, body, Some(&t))?)),
        _ => Ok(read(r)),
    }
}

fn read(mut r: ureq::http::Response<ureq::Body>) -> (u16, String) {
    let status = r.status().as_u16();
    let text = r.body_mut().with_config().limit(1 << 20).read_to_string().unwrap_or_default();
    (status, text)
}

/// Roblox's first error message in a reply, if any.
fn roblox_error(text: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    v.get("errors")?.get(0)?.get("message")?.as_str().map(str::to_owned)
}

/// Writes `text` to `path` readable by this user only.
fn write_private(path: &std::path::Path, text: &str) -> Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut f = options.open(path)?;
    f.write_all(text.as_bytes())?;
    Ok(())
}

impl QuickLogin {
    /// Asks Roblox for a new code.
    pub fn start() -> Result<QuickLogin> {
        let (status, text) = call(&session_agent(), CREATE, &serde_json::json!({}))?;
        if status != 200 {
            return Err(err(format!(
                "Roblox didn't give a sign-in code (HTTP {status}{})",
                roblox_error(&text).map(|m| format!(": {m}")).unwrap_or_default()
            )));
        }
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let field = |k: &str| v.get(k).and_then(|x| x.as_str()).filter(|s| !s.is_empty()).map(str::to_owned);
        match (field("code"), field("privateKey")) {
            (Some(code), Some(private_key)) => Ok(QuickLogin { code, private_key }),
            _ => Err(err("Roblox's sign-in code reply had no code")),
        }
    }

    /// A code from an earlier [`QuickLogin::start`] (resuming after a
    /// restart; see [`QuickLogin::pending`]).
    pub fn from_parts(code: String, private_key: String) -> QuickLogin {
        QuickLogin { code, private_key }
    }

    fn body(&self) -> serde_json::Value {
        serde_json::json!({ "code": self.code, "privateKey": self.private_key })
    }

    /// The pending-code file (`RUSTUDIO_ACCOUNT_CONFIG` moves it with the
    /// config).
    pub fn pending_path() -> PathBuf {
        AccountConfig::path().with_file_name(PENDING_FILE)
    }

    /// Keeps this code for a later start ([`QuickLogin::pending`]): the
    /// editor may be stopped while the browser is in front. The file is
    /// private to the user and [`QuickLogin::forget`] removes it once the
    /// sign-in ends either way.
    pub fn remember(&self) -> Result<()> {
        let path = Self::pending_path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let started = (std::time::UNIX_EPOCH + web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).unwrap_or_default()).duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let text = serde_json::json!({ "code": self.code, "privateKey": self.private_key, "started": started }).to_string();
        write_private(&path, &text)
    }

    /// Removes the pending-code file, if any.
    pub fn forget() {
        let _ = std::fs::remove_file(Self::pending_path());
    }

    /// The code an earlier run was waiting on, if it is recent enough to
    /// still be valid. A stale or unreadable file is removed.
    pub fn pending() -> Option<QuickLogin> {
        let text = std::fs::read_to_string(Self::pending_path()).ok()?;
        let parsed = serde_json::from_str::<serde_json::Value>(&text).ok().and_then(|v| {
            let field = |k: &str| v.get(k).and_then(|x| x.as_str()).filter(|s| !s.is_empty()).map(str::to_owned);
            let started = v.get("started").and_then(|x| x.as_u64())?;
            Some((field("code")?, field("privateKey")?, started))
        });
        let now = (std::time::UNIX_EPOCH + web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).unwrap_or_default()).duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        match parsed {
            Some((code, private_key, started)) if now.saturating_sub(started) <= PENDING_FOR.as_secs() => {
                Some(QuickLogin { code, private_key })
            }
            _ => {
                Self::forget();
                None
            }
        }
    }

    /// Where the code is now.
    pub fn status(&self) -> Result<QuickStatus> {
        let (status, text) = call(&session_agent(), STATUS, &self.body())?;
        if status == 400 || status == 404 {
            // An unknown code is an expired one.
            return Ok(QuickStatus::Expired);
        }
        if status != 200 {
            return Err(err(format!("Roblox's sign-in check failed (HTTP {status})")));
        }
        let v: serde_json::Value = serde_json::from_str(&text)?;
        Ok(match v.get("status").and_then(|s| s.as_str()).unwrap_or("") {
            "UserLinked" => QuickStatus::UserLinked(
                v.get("accountName").and_then(|s| s.as_str()).filter(|s| !s.is_empty()).map(str::to_owned),
            ),
            "Validated" => QuickStatus::Validated,
            "Cancelled" => QuickStatus::Cancelled,
            "Created" => QuickStatus::Created,
            _ => QuickStatus::Expired,
        })
    }

    /// Trades a confirmed code for a session: the `.ROBLOSECURITY` value.
    pub fn finish(&self) -> Result<String> {
        let agent = session_agent();
        let body = serde_json::json!({ "ctype": "AuthToken", "cvalue": self.code, "password": self.private_key });
        let mut csrf: Option<String> = None;
        for _ in 0..2 {
            let r = post(&agent, LOGIN, &body, csrf.as_deref())?;
            let fresh = r.headers().get("x-csrf-token").and_then(|v| v.to_str().ok()).map(str::to_owned);
            let cookie = r
                .headers()
                .get_all("set-cookie")
                .iter()
                .filter_map(|v| v.to_str().ok())
                .find_map(|c| {
                    let (name, rest) = c.split_once('=')?;
                    let value = rest.split(';').next()?.trim();
                    (name.trim() == COOKIE && !value.is_empty()).then(|| value.to_owned())
                });
            let challenged = r.headers().contains_key("rblx-challenge-id");
            let (status, text) = read(r);
            if let Some(c) = cookie {
                return Ok(c);
            }
            if status == 403 && fresh.is_some() && fresh != csrf && !challenged {
                csrf = fresh;
                continue;
            }
            if challenged {
                return Err(err("Roblox wants an extra verification step for this sign-in; paste the cookie instead"));
            }
            return Err(err(format!(
                "Roblox didn't sign in (HTTP {status}{})",
                roblox_error(&text).map(|m| format!(": {m}")).unwrap_or_default()
            )));
        }
        Err(err("Roblox kept refusing the sign-in token"))
    }

    /// Checks every couple of seconds until the code is confirmed, turned
    /// down, expired, or `stop` says to give up; `on_status` hears each
    /// change. The session value on success, None if it ended without one
    /// (the reason went to `on_status`).
    pub fn wait(&self, stop: &dyn Fn() -> bool, on_status: &mut dyn FnMut(&QuickStatus)) -> Result<Option<String>> {
        let mut last = QuickStatus::Created;
        // A check that fails to reach Roblox (the phone's connection
        // changing while the browser is in front, a moment without
        // network) is no verdict on the code: a few are let pass, the
        // code being good for minutes; only failing on end on gives up.
        const PATIENCE: u32 = 6;
        let mut failures = 0u32;
        loop {
            if stop() {
                return Ok(None);
            }
            let now = match self.status() {
                Ok(s) => {
                    failures = 0;
                    s
                }
                Err(e) if failures + 1 < PATIENCE => {
                    failures += 1;
                    eprintln!("[Info] Roblox: browser sign-in: check {failures} of {PATIENCE} failed ({e}); trying again");
                    std::thread::sleep(Duration::from_secs(3));
                    continue;
                }
                Err(e) => return Err(e),
            };
            if now != last {
                on_status(&now);
                last = now.clone();
            }
            match now {
                QuickStatus::Validated => {
                    // The exchange is one POST; a connection that fails
                    // under it gets two more goes (the token is good
                    // until it is used).
                    let mut tries = 0u32;
                    loop {
                        match self.finish() {
                            Ok(c) => return Ok(Some(c)),
                            Err(e) if tries < 2 && !e.0.starts_with("Roblox") => {
                                tries += 1;
                                eprintln!("[Info] Roblox: browser sign-in: the token exchange failed ({e}); trying again");
                                std::thread::sleep(Duration::from_secs(3));
                            }
                            Err(e) => return Err(e),
                        }
                    }
                }
                QuickStatus::Cancelled | QuickStatus::Expired => return Ok(None),
                _ => {}
            }
            // Short sleeps so Cancel takes effect quickly.
            for _ in 0..8 {
                if stop() {
                    return Ok(None);
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RUSTUDIO_ACCOUNT_CONFIG` is process-wide: one test owns it.
    #[test]
    fn pending_code_round_trip() {
        let dir = std::env::temp_dir().join(format!("rustudio-quick-login-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("RUSTUDIO_ACCOUNT_CONFIG", dir.join("account.toml"));
        assert!(QuickLogin::pending().is_none());
        let q = QuickLogin::from_parts("ABC123".into(), "secret".into());
        q.remember().unwrap();
        assert_eq!(QuickLogin::pending_path(), dir.join("quick-login.json"));
        let back = QuickLogin::pending().expect("fresh code resumes");
        assert_eq!((back.code.as_str(), back.private_key.as_str()), ("ABC123", "secret"));
        // The file survives being read (the resumed run removes it when the sign-in ends).
        assert!(QuickLogin::pending_path().exists());
        // A stale code is dropped, file and all.
        let old = serde_json::json!({ "code": "OLD", "privateKey": "k", "started": 1_000 }).to_string();
        std::fs::write(QuickLogin::pending_path(), old).unwrap();
        assert!(QuickLogin::pending().is_none());
        assert!(!QuickLogin::pending_path().exists());
        // So is one that doesn't parse.
        std::fs::write(QuickLogin::pending_path(), "{").unwrap();
        assert!(QuickLogin::pending().is_none());
        assert!(!QuickLogin::pending_path().exists());
        QuickLogin::forget();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
