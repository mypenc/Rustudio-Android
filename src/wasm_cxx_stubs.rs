//! wasm32-unknown-unknown has no C++ exception runtime (wasi-sdk's libc++abi
//! is built without one), but Luau's parser and compiler `throw` on a syntax
//! or compile error. These two symbols let the module link. Reaching either
//! traps: in the browser build a script with a syntax error stops the page
//! instead of reporting the error. (Runtime errors never get here: Luau uses
//! setjmp/longjmp on wasm32, see vendor/mlua-sys/build/find_vendored.rs.)
//! A real exception runtime on wasm exception handling would replace this.

#[no_mangle]
pub extern "C" fn __cxa_allocate_exception(_size: usize) -> *mut u8 {
    core::arch::wasm32::unreachable()
}

#[no_mangle]
pub extern "C" fn __cxa_throw(_obj: *mut u8, _tinfo: *mut u8, _dest: *mut u8) -> ! {
    core::arch::wasm32::unreachable()
}
