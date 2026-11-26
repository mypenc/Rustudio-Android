#![allow(dead_code)]

pub fn probe_lua() {
    #[cfg(feature = "lua55")]
    let artifacts = lua_src::Build::new().build(lua_src::Lua55);

    #[cfg(feature = "lua54")]
    let artifacts = lua_src::Build::new().build(lua_src::Lua54);

    #[cfg(feature = "lua53")]
    let artifacts = lua_src::Build::new().build(lua_src::Lua53);

    #[cfg(feature = "lua52")]
    let artifacts = lua_src::Build::new().build(lua_src::Lua52);

    #[cfg(feature = "lua51")]
    let artifacts = lua_src::Build::new().build(lua_src::Lua51);

    #[cfg(feature = "luajit")]
    let artifacts = luajit_src::Build::new()
        .lua52compat(cfg!(feature = "luajit52"))
        .build();

    #[cfg(feature = "luau")]
    let artifacts = {
        let mut build = luau0_src::Build::new();
        build
            .enable_codegen(cfg!(feature = "luau-codegen"))
            .set_max_cstack_size(1000000)
            .set_vector_size(if cfg!(feature = "luau-vector4") { 4 } else { 3 });
        // rustudio (web port): wasm32-unknown-unknown has no C++ exception
        // runtime (wasi-sdk's libc++abi is built without one), so Luau
        // reports errors with setjmp/longjmp there (LUA_USE_LONGJMP).
        if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
            build.use_longjmp(true);
        }
        build.build()
    };

    artifacts.print_cargo_metadata();
}
