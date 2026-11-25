//! A stand-in for the parts of `ureq` 3 that rustudio uses, for the browser
//! build (wasm32). The page has no sockets, and Roblox's web APIs send no CORS
//! headers, so no request can succeed from here: every call fails with
//! [`Error::ConnectionFailed`] and the callers' existing offline paths run.
//! The types keep ureq's shape so the rest of the code compiles unchanged;
//! native targets use the real crate (see the `package = "ureq-wasm"`
//! renames in the Cargo manifests).

use std::fmt;
use std::io::{self, Read};
use std::time::Duration;

pub use http;

// ---------------------------------------------------------------- errors

#[derive(Debug)]
pub enum Timeout {
    Global,
    Connect,
    Resolve,
    SendRequest,
    RecvResponse,
    RecvBody,
    PerCall,
    Other,
}

/// Stand-in for `rustls::Error`.
#[derive(Debug)]
pub struct RustlsError(pub String);

impl fmt::Display for RustlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    StatusCode(u16),
    Http(http::Error),
    BadUri(String),
    Protocol(String),
    Io(io::Error),
    Timeout(Timeout),
    HostNotFound,
    TooManyRedirects,
    RedirectFailed,
    InvalidProxyUrl,
    ConnectionFailed,
    BodyExceedsLimit(u64),
    Tls(&'static str),
    Rustls(RustlsError),
    Decompress(String, io::Error),
    Json(serde_json::Error),
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::StatusCode(c) => write!(f, "http status: {c}"),
            Error::Http(e) => write!(f, "http: {e}"),
            Error::BadUri(u) => write!(f, "bad uri: {u}"),
            Error::Protocol(p) => write!(f, "protocol: {p}"),
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Timeout(t) => write!(f, "timeout: {t:?}"),
            Error::HostNotFound => f.write_str("host not found"),
            Error::TooManyRedirects => f.write_str("too many redirects"),
            Error::RedirectFailed => f.write_str("redirect failed"),
            Error::InvalidProxyUrl => f.write_str("invalid proxy url"),
            Error::ConnectionFailed => f.write_str("connection failed: the browser build has no network access to this host"),
            Error::BodyExceedsLimit(n) => write!(f, "response body is larger than {n} bytes"),
            Error::Tls(s) => write!(f, "tls: {s}"),
            Error::Rustls(e) => write!(f, "rustls: {e}"),
            Error::Decompress(k, e) => write!(f, "{k} decompress: {e}"),
            Error::Json(e) => write!(f, "json: {e}"),
            Error::Other(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<http::Error> for Error {
    fn from(e: http::Error) -> Self {
        Error::Http(e)
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

// ---------------------------------------------------------------- config

pub mod config {
    use super::*;

    #[derive(Clone, Debug, Default)]
    pub struct Config {
        pub(crate) user_agent: Option<String>,
        pub(crate) http_status_as_error: bool,
        pub(crate) max_redirects: u32,
    }

    impl Config {
        pub fn builder() -> ConfigBuilder<AgentScope> {
            ConfigBuilder::new()
        }
        pub fn http_status_as_error(&self) -> bool {
            self.http_status_as_error
        }
        pub fn user_agent(&self) -> Option<&str> {
            self.user_agent.as_deref()
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct AgentScope;
    #[derive(Debug, Clone, Copy)]
    pub struct RequestScope;

    #[derive(Debug)]
    pub struct ConfigBuilder<S> {
        pub(crate) config: Config,
        pub(crate) _scope: std::marker::PhantomData<S>,
    }

    impl<S> ConfigBuilder<S> {
        pub(crate) fn new() -> Self {
            ConfigBuilder {
                config: Config { http_status_as_error: true, max_redirects: 10, ..Config::default() },
                _scope: std::marker::PhantomData,
            }
        }
        pub fn http_status_as_error(mut self, v: bool) -> Self {
            self.config.http_status_as_error = v;
            self
        }
        pub fn https_only(self, _v: bool) -> Self {
            self
        }
        pub fn max_redirects(mut self, v: u32) -> Self {
            self.config.max_redirects = v;
            self
        }
        pub fn user_agent(mut self, v: impl Into<String>) -> Self {
            self.config.user_agent = Some(v.into());
            self
        }
        pub fn accept(self, _v: impl Into<String>) -> Self {
            self
        }
        pub fn accept_encoding(self, _v: impl Into<String>) -> Self {
            self
        }
        pub fn timeout_global(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_per_call(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_resolve(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_connect(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_send_request(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_await_100(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_send_body(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_recv_response(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_recv_body(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_per_read(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn timeout_per_write(self, _v: Option<Duration>) -> Self {
            self
        }
        pub fn max_idle_connections(self, _v: usize) -> Self {
            self
        }
        pub fn max_idle_connections_per_host(self, _v: usize) -> Self {
            self
        }
        pub fn max_idle_age(self, _v: Duration) -> Self {
            self
        }
        pub fn no_delay(self, _v: bool) -> Self {
            self
        }
        pub fn save_redirect_history(self, _v: bool) -> Self {
            self
        }
        pub fn input_buffer_size(self, _v: usize) -> Self {
            self
        }
        pub fn output_buffer_size(self, _v: usize) -> Self {
            self
        }
        pub fn max_response_header_size(self, _v: usize) -> Self {
            self
        }
    }

    impl ConfigBuilder<AgentScope> {
        pub fn build(self) -> Config {
            self.config
        }
    }
}

pub use config::Config;

// ---------------------------------------------------------------- body

/// A response body. Never produced by a successful call here; kept so the
/// types in signatures resolve.
#[derive(Debug, Default)]
pub struct Body {
    data: io::Cursor<Vec<u8>>,
    limit: Option<u64>,
}

pub struct BodyBuilder;

impl BodyBuilder {
    pub fn data(self, data: impl Into<Vec<u8>>) -> Body {
        Body { data: io::Cursor::new(data.into()), limit: None }
    }
}

impl Body {
    pub fn builder() -> BodyBuilder {
        BodyBuilder
    }
    pub fn mime_type(&self) -> Option<&str> {
        None
    }
    pub fn charset(&self) -> Option<&str> {
        None
    }
    pub fn content_length(&self) -> Option<u64> {
        Some(self.data.get_ref().len() as u64)
    }
    pub fn as_reader(&mut self) -> BodyReader<'_> {
        BodyReader { inner: &mut self.data }
    }
    pub fn into_reader(self) -> BodyOwnedReader {
        BodyOwnedReader { inner: self.data }
    }
    pub fn with_config(&mut self) -> BodyWithConfig<'_> {
        BodyWithConfig { body: self, limit: 10 * 1024 * 1024 }
    }
    pub fn read_to_string(&mut self) -> Result<String, Error> {
        self.with_config().read_to_string()
    }
    pub fn read_to_vec(&mut self) -> Result<Vec<u8>, Error> {
        self.with_config().read_to_vec()
    }
    pub fn read_json<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, Error> {
        let v = self.read_to_vec()?;
        Ok(serde_json::from_slice(&v)?)
    }
}

pub struct BodyReader<'a> {
    inner: &'a mut io::Cursor<Vec<u8>>,
}
impl Read for BodyReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}
pub struct BodyOwnedReader {
    inner: io::Cursor<Vec<u8>>,
}
impl Read for BodyOwnedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

pub struct BodyWithConfig<'a> {
    body: &'a mut Body,
    limit: u64,
}
impl<'a> BodyWithConfig<'a> {
    pub fn limit(mut self, v: u64) -> Self {
        self.limit = v;
        self
    }
    pub fn lossy_utf8(self, _v: bool) -> Self {
        self
    }
    pub fn reader(self) -> BodyReader<'a> {
        BodyReader { inner: &mut self.body.data }
    }
    pub fn read_to_vec(self) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();
        self.body.data.by_ref().take(self.limit + 1).read_to_end(&mut out)?;
        if out.len() as u64 > self.limit {
            return Err(Error::BodyExceedsLimit(self.limit));
        }
        Ok(out)
    }
    pub fn read_to_string(self) -> Result<String, Error> {
        let v = self.read_to_vec()?;
        String::from_utf8(v).map_err(|e| Error::Other(e.to_string()))
    }
}

// ---------------------------------------------------------------- send body

/// What `RequestBuilder::send` / `Agent::run` accept as a request body.
pub trait AsSendBody {
    fn into_send_bytes(self) -> Vec<u8>;
}
impl AsSendBody for () {
    fn into_send_bytes(self) -> Vec<u8> {
        Vec::new()
    }
}
impl AsSendBody for Vec<u8> {
    fn into_send_bytes(self) -> Vec<u8> {
        self
    }
}
impl AsSendBody for &Vec<u8> {
    fn into_send_bytes(self) -> Vec<u8> {
        self.clone()
    }
}
impl AsSendBody for &[u8] {
    fn into_send_bytes(self) -> Vec<u8> {
        self.to_vec()
    }
}
impl<const N: usize> AsSendBody for &[u8; N] {
    fn into_send_bytes(self) -> Vec<u8> {
        self.to_vec()
    }
}
impl AsSendBody for &str {
    fn into_send_bytes(self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}
impl AsSendBody for String {
    fn into_send_bytes(self) -> Vec<u8> {
        self.into_bytes()
    }
}
impl AsSendBody for &String {
    fn into_send_bytes(self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

// ---------------------------------------------------------------- agent

#[derive(Clone, Debug)]
pub struct Agent {
    config: Config,
}

impl Default for Agent {
    fn default() -> Self {
        Agent { config: config::ConfigBuilder::<config::AgentScope>::new().build() }
    }
}

impl From<Config> for Agent {
    fn from(config: Config) -> Self {
        Agent { config }
    }
}

pub struct WithoutBody;
pub struct WithBody;

pub struct RequestBuilder<B> {
    agent: Agent,
    builder: http::request::Builder,
    config: config::ConfigBuilder<config::RequestScope>,
    _b: std::marker::PhantomData<B>,
}

impl Agent {
    pub fn new_with_defaults() -> Self {
        Self::default()
    }
    pub fn new_with_config(config: Config) -> Self {
        Agent { config }
    }
    pub fn config_builder() -> config::ConfigBuilder<config::AgentScope> {
        config::ConfigBuilder::new()
    }
    pub fn config(&self) -> &Config {
        &self.config
    }

    fn req<B>(&self, method: http::Method, uri: impl AsRef<str>) -> RequestBuilder<B> {
        RequestBuilder {
            agent: self.clone(),
            builder: http::Request::builder().method(method).uri(uri.as_ref()),
            config: config::ConfigBuilder::new(),
            _b: std::marker::PhantomData,
        }
    }
    pub fn get(&self, uri: impl AsRef<str>) -> RequestBuilder<WithoutBody> {
        self.req(http::Method::GET, uri)
    }
    pub fn head(&self, uri: impl AsRef<str>) -> RequestBuilder<WithoutBody> {
        self.req(http::Method::HEAD, uri)
    }
    pub fn delete(&self, uri: impl AsRef<str>) -> RequestBuilder<WithoutBody> {
        self.req(http::Method::DELETE, uri)
    }
    pub fn post(&self, uri: impl AsRef<str>) -> RequestBuilder<WithBody> {
        self.req(http::Method::POST, uri)
    }
    pub fn put(&self, uri: impl AsRef<str>) -> RequestBuilder<WithBody> {
        self.req(http::Method::PUT, uri)
    }
    pub fn patch(&self, uri: impl AsRef<str>) -> RequestBuilder<WithBody> {
        self.req(http::Method::PATCH, uri)
    }

    /// Runs a prepared request. Always fails here (see the crate docs).
    pub fn run<B: AsSendBody>(&self, request: http::Request<B>) -> Result<http::Response<Body>, Error> {
        let _ = request.into_body().into_send_bytes();
        Err(Error::ConnectionFailed)
    }
}

impl<B> RequestBuilder<B> {
    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        http::HeaderName: TryFrom<K>,
        <http::HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        http::HeaderValue: TryFrom<V>,
        <http::HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.builder = self.builder.header(key, value);
        self
    }
    pub fn query(self, _key: impl AsRef<str>, _value: impl AsRef<str>) -> Self {
        self
    }
    pub fn config(self) -> RequestConfig<B> {
        RequestConfig { inner: self }
    }
}

/// `request.config().timeout_global(..).build()` returns the request again.
pub struct RequestConfig<B> {
    inner: RequestBuilder<B>,
}
macro_rules! forward {
    ($($name:ident($($arg:ident: $ty:ty),*)),* $(,)?) => {$(
        pub fn $name(mut self, $($arg: $ty),*) -> Self {
            self.inner.config = self.inner.config.$name($($arg),*);
            self
        }
    )*};
}
impl<B> RequestConfig<B> {
    forward!(
        http_status_as_error(v: bool),
        max_redirects(v: u32),
        timeout_global(v: Option<Duration>),
        timeout_per_call(v: Option<Duration>),
        timeout_connect(v: Option<Duration>),
        timeout_resolve(v: Option<Duration>),
        timeout_recv_response(v: Option<Duration>),
        timeout_recv_body(v: Option<Duration>),
    );
    pub fn build(self) -> RequestBuilder<B> {
        self.inner
    }
}

impl RequestBuilder<WithoutBody> {
    pub fn call(self) -> Result<http::Response<Body>, Error> {
        let req = self.builder.body(())?;
        self.agent.run(req)
    }
}

impl RequestBuilder<WithBody> {
    pub fn send(self, data: impl AsSendBody) -> Result<http::Response<Body>, Error> {
        let req = self.builder.body(data.into_send_bytes())?;
        self.agent.run(req)
    }
    pub fn send_empty(self) -> Result<http::Response<Body>, Error> {
        self.send(())
    }
    pub fn send_json(self, data: impl serde::Serialize) -> Result<http::Response<Body>, Error> {
        let v = serde_json::to_vec(&data)?;
        self.send(v)
    }
}

/// `ureq::get(url)`: a request on a default agent.
pub fn get(uri: impl AsRef<str>) -> RequestBuilder<WithoutBody> {
    Agent::default().get(uri)
}
pub fn post(uri: impl AsRef<str>) -> RequestBuilder<WithBody> {
    Agent::default().post(uri)
}

// ---------------------------------------------------------------- response ext

pub trait ResponseExt {
    fn get_uri(&self) -> &http::Uri;
}
impl<T> ResponseExt for http::Response<T> {
    fn get_uri(&self) -> &http::Uri {
        static EMPTY: std::sync::OnceLock<http::Uri> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| http::Uri::from_static("/"))
    }
}
