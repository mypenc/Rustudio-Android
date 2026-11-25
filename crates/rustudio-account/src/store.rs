//! System keyring entries for the signed-in session and saved accounts.
//! Legacy publishing-key entries remain accessible for older clients.

use crate::{Error, Result};

/// The keyring's error type (keyring-core's on every platform).
pub(crate) type KeyringError = keyring::Error;

/// Android: the `keyring` crate has no store here (its `Entry::new` answers
/// NoDefaultStore), so entries go through keyring-core with
/// android-native-keyring-store: secrets in a SharedPreferences file of the
/// app's own, encrypted with a key the Android Keystore holds. The store is
/// installed as keyring-core's default on first use; it needs the
/// application context ndk-context carries, which android-activity sets
/// before `android_main`.
#[cfg(target_os = "android")]
mod keyring {
    pub use keyring_core::Error;

    pub struct Entry(keyring_core::Entry);

    fn init() -> keyring_core::Result<()> {
        static STORE: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
        let r = STORE.get_or_init(|| {
            android_native_keyring_store::Store::new()
                .map(|store| keyring_core::set_default_store(store))
                .map_err(|e| e.to_string())
        });
        match r {
            Ok(()) => Ok(()),
            Err(e) => Err(Error::Invalid("Android keystore".to_string(), e.clone())),
        }
    }

    impl Entry {
        pub fn new(service: &str, user: &str) -> keyring_core::Result<Self> {
            init()?;
            keyring_core::Entry::new(service, user).map(Self)
        }

        pub fn get_secret(&self) -> keyring_core::Result<Vec<u8>> {
            self.0.get_secret()
        }

        pub fn set_secret(&self, secret: &[u8]) -> keyring_core::Result<()> {
            self.0.set_secret(secret)
        }

        pub fn get_password(&self) -> keyring_core::Result<String> {
            self.0.get_password()
        }

        pub fn set_password(&self, password: &str) -> keyring_core::Result<()> {
            self.0.set_password(password)
        }

        pub fn delete_credential(&self) -> keyring_core::Result<()> {
            self.0.delete_credential()
        }
    }
}

/// Browser build: there is no system keyring. Entries live in this page's
/// memory for its lifetime (nothing secret is written to disk or to web
/// storage).
#[cfg(target_arch = "wasm32")]
mod keyring {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    #[derive(Debug)]
    pub enum Error {
        NoEntry,
        Other(String),
    }

    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Error::NoEntry => f.write_str("no matching entry"),
                Error::Other(s) => f.write_str(s),
            }
        }
    }

    impl std::error::Error for Error {}

    fn map() -> &'static Mutex<HashMap<String, Vec<u8>>> {
        static MAP: OnceLock<Mutex<HashMap<String, Vec<u8>>>> = OnceLock::new();
        MAP.get_or_init(Default::default)
    }

    pub struct Entry(String);

    impl Entry {
        pub fn new(service: &str, user: &str) -> Result<Entry, Error> {
            Ok(Entry(format!("{service}\u{0}{user}")))
        }
        pub fn get_secret(&self) -> Result<Vec<u8>, Error> {
            map().lock().unwrap().get(&self.0).cloned().ok_or(Error::NoEntry)
        }
        pub fn set_secret(&self, secret: &[u8]) -> Result<(), Error> {
            map().lock().unwrap().insert(self.0.clone(), secret.to_vec());
            Ok(())
        }
        pub fn get_password(&self) -> Result<String, Error> {
            let v = self.get_secret()?;
            String::from_utf8(v).map_err(|e| Error::Other(e.to_string()))
        }
        pub fn set_password(&self, password: &str) -> Result<(), Error> {
            self.set_secret(password.as_bytes())
        }
        pub fn delete_credential(&self) -> Result<(), Error> {
            map().lock().unwrap().remove(&self.0).map(|_| ()).ok_or(Error::NoEntry)
        }
    }
}

const SERVICE: &str = "rustudio";
const API_KEY_USER: &str = "roblox-open-cloud-api-key";
/// Left behind by OAuth sign-in before the cookie session; removed on sight.
const OLD_OAUTH_USER: &str = "roblox-oauth-tokens";

fn entry(user: &str) -> Result<keyring::Entry> {
    keyring::Entry::new(SERVICE, user)
        .map_err(|e| Error(format!("system keyring unavailable: {e}")))
}

/// The secret stored for `user` under service `rustudio`, if any. Raw
/// bytes, so Windows (which stores a password as UTF-16) fits twice as much.
#[cfg(not(test))]
pub(crate) fn load(user: &str) -> Result<Option<Vec<u8>>> {
    match entry(user)?.get_secret() {
        Ok(v) => Ok(Some(v)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(not(test))]
pub(crate) fn save(user: &str, secret: &[u8]) -> Result<()> {
    entry(user)?.set_secret(secret)?;
    Ok(())
}

#[cfg(not(test))]
pub(crate) fn delete(user: &str) -> Result<()> {
    match entry(user)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
thread_local! {
    /// Tests keep the entries in memory (per test thread) instead of the
    /// person's keyring.
    static MEMORY: std::cell::RefCell<std::collections::HashMap<String, Vec<u8>>> =
        Default::default();
}

#[cfg(test)]
pub(crate) fn load(user: &str) -> Result<Option<Vec<u8>>> {
    Ok(MEMORY.with(|m| m.borrow().get(user).cloned()))
}

#[cfg(test)]
pub(crate) fn save(user: &str, secret: &[u8]) -> Result<()> {
    MEMORY.with(|m| m.borrow_mut().insert(user.to_string(), secret.to_vec()));
    Ok(())
}

#[cfg(test)]
pub(crate) fn delete(user: &str) -> Result<()> {
    MEMORY.with(|m| m.borrow_mut().remove(user));
    Ok(())
}

pub struct PublishKeyStore;

impl PublishKeyStore {
    fn entry(user: &str) -> Result<keyring::Entry> {
        entry(user)
    }

    pub fn save(key: &str) -> Result<()> {
        Self::entry(API_KEY_USER)?.set_password(key.trim())?;
        Ok(())
    }

    /// The saved publishing key, if any.
    pub fn load() -> Result<Option<String>> {
        match Self::entry(API_KEY_USER)?.get_password() {
            Ok(s) => Ok(Some(s).filter(|k| !k.trim().is_empty())),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn clear() -> Result<()> {
        match Self::entry(API_KEY_USER)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// Deletes the OAuth tokens an older rustudio saved.
    pub fn forget_oauth_tokens() {
        if let Ok(e) = Self::entry(OLD_OAUTH_USER) {
            let _ = e.delete_credential();
        }
    }
}
