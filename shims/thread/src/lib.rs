//! `std::thread` everywhere except wasm32, where the browser build has no
//! threads: `spawn` runs the closure inline on the caller and returns an
//! already-finished handle, and `sleep` spins on the web clock.

#[cfg(not(target_arch = "wasm32"))]
pub use std::thread::*;

#[cfg(target_arch = "wasm32")]
pub use std::thread::{current, park, park_timeout, yield_now, Thread, ThreadId};

#[cfg(target_arch = "wasm32")]
pub use self::wasm::*;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::any::Any;
    use std::io;
    use std::marker::PhantomData;
    use std::num::NonZeroUsize;
    use std::time::Duration;

    pub type Result<T> = std::thread::Result<T>;

    pub struct JoinHandle<T> {
        result: Option<T>,
        thread: std::thread::Thread,
    }
    impl<T> JoinHandle<T> {
        pub fn join(mut self) -> Result<T> {
            match self.result.take() {
                Some(v) => Ok(v),
                None => Err(Box::new("worker threads are not run in the browser build") as Box<dyn Any + Send>),
            }
        }
        pub fn thread(&self) -> &std::thread::Thread {
            &self.thread
        }
        pub fn is_finished(&self) -> bool {
            true
        }
    }

    pub fn spawn<F, T>(f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        JoinHandle { result: Some(f()), thread: std::thread::current() }
    }

    #[derive(Debug, Default)]
    pub struct Builder {
        name: Option<String>,
    }
    impl Builder {
        pub fn new() -> Self {
            Builder { name: None }
        }
        pub fn name(mut self, name: String) -> Self {
            self.name = Some(name);
            self
        }
        pub fn stack_size(self, _size: usize) -> Self {
            self
        }
        pub fn spawn<F, T>(self, f: F) -> io::Result<JoinHandle<T>>
        where
            F: FnOnce() -> T + Send + 'static,
            T: Send + 'static,
        {
            if self.name.is_some() {
                // Named threads are long-lived workers (`loop { rx.recv() }`,
                // watchdogs): running them inline would never return. Park the
                // closure forever instead; leaking it keeps its channels open so
                // senders do not start failing.
                std::mem::forget(f);
                return Ok(JoinHandle { result: None, thread: std::thread::current() });
            }
            Ok(spawn(f))
        }
        pub fn spawn_scoped<'scope, 'env, F, T>(self, scope: &'scope Scope<'scope, 'env>, f: F) -> io::Result<ScopedJoinHandle<'scope, T>>
        where
            F: FnOnce() -> T + Send + 'scope,
            T: Send + 'scope,
        {
            Ok(scope.spawn(f))
        }
    }

    pub struct Scope<'scope, 'env: 'scope> {
        _scope: PhantomData<&'scope mut &'scope ()>,
        _env: PhantomData<&'env mut &'env ()>,
    }
    pub struct ScopedJoinHandle<'scope, T> {
        result: Option<T>,
        _scope: PhantomData<&'scope ()>,
    }
    impl<'scope, T> ScopedJoinHandle<'scope, T> {
        pub fn join(mut self) -> Result<T> {
            Ok(self.result.take().expect("joined twice"))
        }
        pub fn is_finished(&self) -> bool {
            true
        }
    }
    impl<'scope, 'env> Scope<'scope, 'env> {
        pub fn spawn<F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
        where
            F: FnOnce() -> T + Send + 'scope,
            T: Send + 'scope,
        {
            ScopedJoinHandle { result: Some(f()), _scope: PhantomData }
        }
    }
    pub fn scope<'env, F, T>(f: F) -> T
    where
        F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
    {
        let s = Scope { _scope: PhantomData, _env: PhantomData };
        f(&s)
    }

    pub fn sleep(dur: Duration) {
        let end = web_time::Instant::now() + dur;
        while web_time::Instant::now() < end {
            std::hint::spin_loop();
        }
    }

    pub fn available_parallelism() -> io::Result<NonZeroUsize> {
        Ok(NonZeroUsize::new(1).unwrap())
    }

    #[allow(dead_code)]
    fn _assert(_: &dyn Any) {}
}
