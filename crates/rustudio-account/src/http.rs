//! The HTTP agents rustudio makes, built one way: ureq with a resolver
//! that puts a host's IPv4 addresses before its IPv6 ones.
//!
//! ureq tries a host's addresses one after the other, each with a share of
//! the connect timeout (none set: the kernel's, about two minutes). A
//! resolver lists IPv6 first, and a network whose IPv6 is advertised but
//! doesn't route — some phones' Wi-Fi and mobile networks — makes every
//! new connection wait that share out before IPv4 is tried: an app that
//! "works, but is very slow". With IPv4 first such a network costs nothing;
//! an IPv6-only network (its DNS64 giving no A records) is served by the
//! IPv6 addresses that remain. (`ureq::unversioned` is ureq's "may change
//! in a minor release" API; Cargo.lock pins the version.)

#[cfg(not(target_arch = "wasm32"))]
use ureq::config::Config;
#[cfg(not(target_arch = "wasm32"))]
use ureq::unversioned::resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver};
#[cfg(not(target_arch = "wasm32"))]
use ureq::unversioned::transport::{DefaultConnector, NextTimeout};
#[cfg(not(target_arch = "wasm32"))]
use ureq::Agent;
#[cfg(target_arch = "wasm32")]
use ureq::{config::Config, Agent};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Default)]
#[cfg(not(target_arch = "wasm32"))]
struct V4First(DefaultResolver);

#[cfg(not(target_arch = "wasm32"))]
impl Resolver for V4First {
    fn resolve(&self, uri: &ureq::http::Uri, config: &Config, timeout: NextTimeout) -> Result<ResolvedSocketAddrs, ureq::Error> {
        let mut addrs = self.0.resolve(uri, config, timeout)?;
        // (stable: each family keeps the resolver's order)
        addrs.sort_by_key(|a| a.is_ipv6());
        Ok(addrs)
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// An agent for `config` (`ureq::Agent::config_builder()…build()`), with
/// the resolver above. In place of `Agent::from(config)`.
pub fn agent(config: Config) -> Agent {
    Agent::with_parts(config, DefaultConnector::default(), V4First::default())
}

#[cfg(not(target_arch = "wasm32"))]
/// One line on how `url`'s host is reached, phase by phase — DNS, a TCP
/// connect to its first IPv4 and its first IPv6 address (5 s each), then
/// an HTTPS GET with the agent above — with the time each took. For the
/// log of a device where the network "works, but slowly": it tells a slow
/// DNS from an address family that doesn't route from a slow server.
/// Takes up to ~40 s; call it on a thread of its own.
pub fn probe(url: &str) -> Result<String, String> {
    use std::fmt::Write as _;
    use std::net::{TcpStream, ToSocketAddrs};
    use std::time::Duration;
use web_time::Instant;

    let uri: ureq::http::Uri = url.parse().map_err(|e| format!("{url}: {e}"))?;
    let host = uri.host().ok_or_else(|| format!("{url}: no host"))?.to_string();
    let port = uri.port_u16().unwrap_or(443);
    let mut out = format!("{host}:");

    let t = Instant::now();
    let addrs: Vec<_> = match (host.as_str(), port).to_socket_addrs() {
        Ok(a) => a.collect(),
        Err(e) => return Err(format!("{host}: DNS failed after {} ms: {e}", t.elapsed().as_millis())),
    };
    let v4 = addrs.iter().find(|a| a.is_ipv4()).copied();
    let v6 = addrs.iter().find(|a| a.is_ipv6()).copied();
    let _ = write!(
        out,
        " DNS {} ms ({} IPv4, {} IPv6{})",
        t.elapsed().as_millis(),
        addrs.iter().filter(|a| a.is_ipv4()).count(),
        addrs.iter().filter(|a| a.is_ipv6()).count(),
        if addrs.first().map_or(false, |a| a.is_ipv6()) { ", IPv6 listed first" } else { "" }
    );

    for (name, addr) in [("IPv4", v4), ("IPv6", v6)] {
        let Some(addr) = addr else { continue };
        let t = Instant::now();
        match TcpStream::connect_timeout(&addr, Duration::from_secs(5)) {
            Ok(_) => { let _ = write!(out, "; {name} TCP connect {} ms", t.elapsed().as_millis()); }
            Err(e) => { let _ = write!(out, "; {name} TCP connect FAILED after {} ms ({e})", t.elapsed().as_millis()); }
        }
    }

    let config = Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(30)))
        .timeout_resolve(Some(Duration::from_secs(15)))
        .timeout_connect(Some(Duration::from_secs(10)))
        .user_agent(concat!("rustudio/", env!("CARGO_PKG_VERSION")))
        .build();
    let t = Instant::now();
    match agent(config).get(url).call() {
        Ok(mut r) => {
            let headers_ms = t.elapsed().as_millis();
            let status = r.status().as_u16();
            let bytes = r.body_mut().read_to_vec().map(|v| v.len()).unwrap_or(0);
            let _ = write!(out, "; HTTPS GET {status}: headers in {headers_ms} ms, {bytes} bytes in {} ms", t.elapsed().as_millis());
            Ok(out)
        }
        Err(e) => Err(format!("{out}; HTTPS GET FAILED after {} ms: {e}", t.elapsed().as_millis())),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_before_v6() {
        // A name every resolver answers with both families, in its own
        // order; localhost's are "::1" and "127.0.0.1".
        let r = V4First::default();
        let config = Agent::config_builder().build();
        let uri: ureq::http::Uri = "http://localhost:80".parse().unwrap();
        let addrs = r.resolve(&uri, &config, NextTimeout::default()).unwrap();
        let first_v6 = addrs.iter().position(|a| a.is_ipv6());
        let last_v4 = addrs.iter().rposition(|a| a.is_ipv4());
        if let (Some(v6), Some(v4)) = (first_v6, last_v4) {
            assert!(v4 < v6, "{addrs:?}");
        }
    }

    #[test]
    #[ignore = "needs the network"]
    fn probe_line() {
        let line = probe("https://clientsettingscdn.roblox.com/v2/client-version/WindowsStudio64");
        eprintln!("{line:?}");
        let line = line.unwrap();
        assert!(line.contains("DNS") && line.contains("HTTPS GET 200"), "{line}");
    }
}

/// The browser build has no connector or resolver to configure.
#[cfg(target_arch = "wasm32")]
pub fn agent(config: Config) -> Agent {
    Agent::from(config)
}

/// There is no socket-level network probe in the browser.
#[cfg(target_arch = "wasm32")]
pub fn probe(_url: &str) -> Result<String, String> {
    Err("network probe is not available in the browser build".into())
}
