//! Script editor tabs: Luau syntax highlighting, line numbers, live compile
//! check, fullscreen maximize, Portrait/Landscape orientation toggle,
//! touch-friendly Luau autocomplete, and smart bracket/quote/block auto-closing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use web_time::Instant;

use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, FontId, RichText, Stroke};
use rbx_dom_weak::types::{Ref, Variant};

use crate::app::Editor;
use crate::icons;
use rustudio_engine::runtime;
use rustudio_engine::theme;

const KEYWORDS: &[&str] = &[
    "and", "break", "continue", "do", "else", "elseif", "end", "export", "false", "for",
    "function", "if", "in", "local", "nil", "not", "or", "repeat", "return", "then", "true",
    "type", "until", "while",
];

const GLOBALS: &[&str] = &[
    "game",
    "workspace",
    "Workspace",
    "script",
    "Instance",
    "Vector3",
    "Vector2",
    "CFrame",
    "Color3",
    "BrickColor",
    "UDim",
    "UDim2",
    "TweenInfo",
    "RaycastParams",
    "Enum",
    "task",
    "print",
    "warn",
    "error",
    "require",
    "typeof",
    "type",
    "math",
    "string",
    "table",
    "pcall",
    "xpcall",
    "pairs",
    "ipairs",
    "next",
    "select",
    "tostring",
    "tonumber",
    "setmetatable",
    "getmetatable",
    "assert",
    "wait",
    "spawn",
    "delay",
    "tick",
    "time",
    "coroutine",
    "buffer",
    "bit32",
    "utf8",
    "os",
    "debug",
    "self",
    "shared",
    "_G",
    "rawget",
    "rawset",
    "rawequal",
    "unpack",
];

// ------------------------------------------------------------------ autocomplete catalog

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AcKind {
    Keyword,
    Stdlib,
    Global,
    Method,
    Property,
    Local,
    Plugin,
}

impl AcKind {
    fn badge(self) -> (&'static str, Color32) {
        match self {
            AcKind::Keyword => ("kw", Color32::from_rgb(198, 120, 221)),
            AcKind::Stdlib => ("fn", Color32::from_rgb(97, 175, 239)),
            AcKind::Global => ("cls", Color32::from_rgb(229, 192, 123)),
            AcKind::Method => ("mth", Color32::from_rgb(86, 182, 194)),
            AcKind::Property => ("prp", Color32::from_rgb(152, 195, 121)),
            AcKind::Local => ("var", Color32::from_rgb(224, 108, 117)),
            AcKind::Plugin => ("plg", Color32::from_rgb(180, 180, 195)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AcItem {
    label: String,
    insert: String,
    kind: AcKind,
    detail: &'static str,
}

const STDLIB_FUNCTIONS: &[(&str, &str)] = &[
    ("task.wait", "(seconds?) -> number"),
    ("task.spawn", "(f, ...) -> thread"),
    ("task.defer", "(f, ...) -> thread"),
    ("task.delay", "(duration, f, ...) -> thread"),
    ("math.abs", "(x) -> number"),
    ("math.clamp", "(x, min, max) -> number"),
    ("math.floor", "(x) -> number"),
    ("math.ceil", "(x) -> number"),
    ("math.round", "(x) -> number"),
    ("math.rad", "(deg) -> number"),
    ("math.deg", "(rad) -> number"),
    ("math.random", "(m?, n?) -> number"),
    ("math.min", "(x, ...) -> number"),
    ("math.max", "(x, ...) -> number"),
    ("table.insert", "(t, pos?, value)"),
    ("table.remove", "(t, pos?) -> any"),
    ("table.find", "(t, value, init?) -> number?"),
    ("table.clone", "(t) -> table"),
    ("table.clear", "(t)"),
    ("string.format", "(fmt, ...) -> string"),
    ("string.sub", "(s, i, j?) -> string"),
    ("string.find", "(s, pattern, init?, plain?)"),
    ("string.gsub", "(s, pattern, repl, n?)"),
    ("string.split", "(s, sep?) -> {string}"),
    ("string.lower", "(s) -> string"),
    ("string.upper", "(s) -> string"),
    ("print", "(...)"),
    ("warn", "(...)"),
    ("error", "(message, level?)"),
    ("pcall", "(f, ...) -> (boolean, ...)"),
    ("xpcall", "(f, err, ...) -> (boolean, ...)"),
    ("pairs", "(t)"),
    ("ipairs", "(t)"),
    ("next", "(t, index?)"),
    ("typeof", "(value) -> string"),
    ("tonumber", "(e, base?) -> number?"),
    ("tostring", "(e) -> string"),
    ("require", "(module) -> any"),
    ("task", "Luau task library"),
    ("math", "Luau math library"),
    ("table", "Luau table library"),
    ("string", "Luau string library"),
];

const ROBLOX_GLOBALS: &[(&str, &str)] = &[
    ("game", "DataModel"),
    ("workspace", "Workspace"),
    ("script", "LuaSourceContainer"),
    ("Instance.new", "(className, parent?) -> Instance"),
    ("Vector3.new", "(x?, y?, z?) -> Vector3"),
    ("Vector2.new", "(x?, y?) -> Vector2"),
    ("CFrame.new", "(...) -> CFrame"),
    ("CFrame.Angles", "(rx, ry, rz) -> CFrame"),
    ("CFrame.lookAt", "(at, lookAt, up?) -> CFrame"),
    ("Color3.fromRGB", "(r, g, b) -> Color3"),
    ("Color3.new", "(r?, g?, b?) -> Color3"),
    ("UDim2.new", "(xS, xO, yS, yO) -> UDim2"),
    ("UDim2.fromScale", "(x, y) -> UDim2"),
    ("UDim2.fromOffset", "(x, y) -> UDim2"),
    ("UDim.new", "(scale?, offset?) -> UDim"),
    ("BrickColor.new", "(val) -> BrickColor"),
    ("TweenInfo.new", "(time?, style?, dir?, ...)"),
    ("RaycastParams.new", "() -> RaycastParams"),
    ("Enum", "Enums"),
    ("Players", "Players service"),
    ("RunService", "RunService"),
    ("UserInputService", "UserInputService"),
    ("ReplicatedStorage", "ReplicatedStorage"),
    ("TweenService", "TweenService"),
    ("CollectionService", "CollectionService"),
    ("Debris", "Debris service"),
    ("Lighting", "Lighting service"),
    ("SoundService", "SoundService"),
    ("StarterGui", "StarterGui"),
    ("ServerScriptService", "ServerScriptService"),
    ("ServerStorage", "ServerStorage"),
    ("Instance", "Roblox Instance"),
    ("Vector3", "3D vector"),
    ("Vector2", "2D vector"),
    ("CFrame", "Coordinate frame"),
    ("Color3", "RGB color"),
    ("UDim2", "2D UI dimension"),
    ("UDim", "1D UI dimension"),
    ("BrickColor", "Named brick color"),
    ("TweenInfo", "Tween configuration"),
    ("RaycastParams", "Raycast parameters"),
];

const MEMBER_METHODS: &[(&str, &str)] = &[
    ("GetService", "(className) -> Instance"),
    ("FindFirstChild", "(name, recursive?) -> Instance?"),
    ("FindFirstChildOfClass", "(className) -> Instance?"),
    ("WaitForChild", "(name, timeOut?) -> Instance"),
    ("Clone", "() -> Instance"),
    ("Destroy", "()"),
    ("GetChildren", "() -> {Instance}"),
    ("GetDescendants", "() -> {Instance}"),
    ("IsA", "(className) -> boolean"),
    ("Connect", "(func) -> RBXScriptConnection"),
    ("Once", "(func) -> RBXScriptConnection"),
    ("Wait", "() -> ..."),
    ("FireServer", "(...)"),
    ("FireClient", "(player, ...)"),
    ("FireAllClients", "(...)"),
    ("PivotTo", "(targetCFrame)"),
    ("GetPivot", "() -> CFrame"),
    ("MoveTo", "(position, part?)"),
    ("GetAttribute", "(attribute) -> any"),
    ("SetAttribute", "(attribute, value)"),
];

const MEMBER_PROPERTIES: &[(&str, &str)] = &[
    ("OnServerEvent", "RBXScriptSignal"),
    ("OnClientEvent", "RBXScriptSignal"),
    ("Parent", "Instance?"),
    ("Name", "string"),
    ("Position", "Vector3"),
    ("CFrame", "CFrame"),
    ("Size", "Vector3"),
    ("Color", "Color3"),
    ("Transparency", "number"),
    ("Anchored", "boolean"),
    ("CanCollide", "boolean"),
    ("Humanoid", "Humanoid"),
    ("Character", "Model?"),
    ("LocalPlayer", "Player"),
    ("Touched", "RBXScriptSignal"),
    ("Heartbeat", "RBXScriptSignal"),
    ("RenderStepped", "RBXScriptSignal"),
    ("Stepped", "RBXScriptSignal"),
];

/// Receiver-specific members for `foo.` completions (e.g. `task.`, `math.`, `CFrame.`).
fn receiver_members(receiver: &str) -> Option<&'static [(&'static str, &'static str, AcKind)]> {
    const TASK_M: &[(&str, &str, AcKind)] = &[
        ("wait", "(seconds?) -> number", AcKind::Stdlib),
        ("spawn", "(f, ...) -> thread", AcKind::Stdlib),
        ("defer", "(f, ...) -> thread", AcKind::Stdlib),
        ("delay", "(duration, f, ...) -> thread", AcKind::Stdlib),
    ];
    const MATH_M: &[(&str, &str, AcKind)] = &[
        ("abs", "(x) -> number", AcKind::Stdlib),
        ("clamp", "(x, min, max) -> number", AcKind::Stdlib),
        ("floor", "(x) -> number", AcKind::Stdlib),
        ("ceil", "(x) -> number", AcKind::Stdlib),
        ("round", "(x) -> number", AcKind::Stdlib),
        ("rad", "(deg) -> number", AcKind::Stdlib),
        ("deg", "(rad) -> number", AcKind::Stdlib),
        ("random", "(m?, n?) -> number", AcKind::Stdlib),
        ("min", "(x, ...) -> number", AcKind::Stdlib),
        ("max", "(x, ...) -> number", AcKind::Stdlib),
    ];
    const TABLE_M: &[(&str, &str, AcKind)] = &[
        ("insert", "(t, pos?, value)", AcKind::Stdlib),
        ("remove", "(t, pos?) -> any", AcKind::Stdlib),
        ("find", "(t, value, init?) -> number?", AcKind::Stdlib),
        ("clone", "(t) -> table", AcKind::Stdlib),
        ("clear", "(t)", AcKind::Stdlib),
    ];
    const STRING_M: &[(&str, &str, AcKind)] = &[
        ("format", "(fmt, ...) -> string", AcKind::Stdlib),
        ("sub", "(s, i, j?) -> string", AcKind::Stdlib),
        ("find", "(s, pattern, init?, plain?)", AcKind::Stdlib),
        ("gsub", "(s, pattern, repl, n?)", AcKind::Stdlib),
        ("split", "(s, sep?) -> {string}", AcKind::Stdlib),
        ("lower", "(s) -> string", AcKind::Stdlib),
        ("upper", "(s) -> string", AcKind::Stdlib),
    ];
    const INSTANCE_M: &[(&str, &str, AcKind)] = &[
        ("new", "(className, parent?) -> Instance", AcKind::Global),
    ];
    const VECTOR3_M: &[(&str, &str, AcKind)] = &[
        ("new", "(x?, y?, z?) -> Vector3", AcKind::Global),
        ("zero", "Vector3(0, 0, 0)", AcKind::Property),
        ("one", "Vector3(1, 1, 1)", AcKind::Property),
    ];
    const VECTOR2_M: &[(&str, &str, AcKind)] = &[
        ("new", "(x?, y?) -> Vector2", AcKind::Global),
        ("zero", "Vector2(0, 0)", AcKind::Property),
        ("one", "Vector2(1, 1)", AcKind::Property),
    ];
    const CFRAME_M: &[(&str, &str, AcKind)] = &[
        ("new", "(...) -> CFrame", AcKind::Global),
        ("Angles", "(rx, ry, rz) -> CFrame", AcKind::Global),
        ("lookAt", "(at, lookAt, up?) -> CFrame", AcKind::Global),
        ("identity", "CFrame", AcKind::Property),
    ];
    const COLOR3_M: &[(&str, &str, AcKind)] = &[
        ("fromRGB", "(r, g, b) -> Color3", AcKind::Global),
        ("new", "(r?, g?, b?) -> Color3", AcKind::Global),
    ];
    const UDIM2_M: &[(&str, &str, AcKind)] = &[
        ("new", "(xS, xO, yS, yO) -> UDim2", AcKind::Global),
        ("fromScale", "(x, y) -> UDim2", AcKind::Global),
        ("fromOffset", "(x, y) -> UDim2", AcKind::Global),
    ];
    const UDIM_M: &[(&str, &str, AcKind)] = &[
        ("new", "(scale?, offset?) -> UDim", AcKind::Global),
    ];
    const BRICKCOLOR_M: &[(&str, &str, AcKind)] = &[
        ("new", "(val) -> BrickColor", AcKind::Global),
    ];
    const TWEENINFO_M: &[(&str, &str, AcKind)] = &[
        ("new", "(time?, style?, dir?, ...)", AcKind::Global),
    ];
    const RAYCASTPARAMS_M: &[(&str, &str, AcKind)] = &[
        ("new", "() -> RaycastParams", AcKind::Global),
    ];

    match receiver {
        "task" => Some(TASK_M),
        "math" => Some(MATH_M),
        "table" => Some(TABLE_M),
        "string" => Some(STRING_M),
        "Instance" => Some(INSTANCE_M),
        "Vector3" => Some(VECTOR3_M),
        "Vector2" => Some(VECTOR2_M),
        "CFrame" => Some(CFRAME_M),
        "Color3" => Some(COLOR3_M),
        "UDim2" => Some(UDIM2_M),
        "UDim" => Some(UDIM_M),
        "BrickColor" => Some(BRICKCOLOR_M),
        "TweenInfo" => Some(TWEENINFO_M),
        "RaycastParams" => Some(RAYCASTPARAMS_M),
        _ => None,
    }
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Scrapes declared local variables, function names, and loop/parameter
/// identifiers from Luau source text.
fn scrape_dynamic_identifiers(src: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut push = |name: &str| {
        if name.len() >= 2
            && !KEYWORDS.contains(&name)
            && !GLOBALS.contains(&name)
            && seen.insert(name.to_string())
        {
            found.push(name.to_string());
        }
    };

    for raw_line in src.lines() {
        let line = strip_line_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        // `local function foo(a, b)` or `function foo(a, b)` or `function Mod:Bar(a)`
        if let Some(rest) = line
            .strip_prefix("local function ")
            .or_else(|| line.strip_prefix("function "))
        {
            let rest = rest.trim_start();
            let name_end = rest
                .find(|c: char| !is_ident_char(c) && c != '.' && c != ':')
                .unwrap_or(rest.len());
            let full = &rest[..name_end];
            // Record the final identifier segment (and the base if plain).
            for seg in full.split(|c| c == '.' || c == ':') {
                if seg.chars().next().is_some_and(is_ident_start) {
                    push(seg);
                }
            }
            if let Some(open) = rest.find('(') {
                if let Some(close) = rest[open + 1..].find(')') {
                    let params = &rest[open + 1..open + 1 + close];
                    for p in params.split(',') {
                        let p = p.split(':').next().unwrap_or("").trim();
                        if p.chars().next().is_some_and(is_ident_start)
                            && p.chars().all(is_ident_char)
                        {
                            push(p);
                        }
                    }
                }
            }
            continue;
        }
        // `local a, b = ...`
        if let Some(rest) = line.strip_prefix("local ") {
            let lhs = rest.split('=').next().unwrap_or(rest);
            for var in lhs.split(',') {
                let name = var.split(':').next().unwrap_or("").trim();
                if name.chars().next().is_some_and(is_ident_start)
                    && name.chars().all(is_ident_char)
                {
                    push(name);
                }
            }
            continue;
        }
        // `for i, v in ...` or `for i = 1, 10 do`
        if let Some(rest) = line.strip_prefix("for ") {
            let lhs = rest
                .split(" in ")
                .next()
                .and_then(|s| s.split('=').next())
                .unwrap_or("");
            for var in lhs.split(',') {
                let name = var.split(':').next().unwrap_or("").trim();
                if name.chars().next().is_some_and(is_ident_start)
                    && name.chars().all(is_ident_char)
                {
                    push(name);
                }
            }
        }
    }
    found
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AcContext {
    /// Char index where the replacement token starts.
    replace_start: usize,
    /// Typed prefix to match against (may be empty right after `.` or `:`).
    prefix: String,
    /// Optional member trigger (`'.'` or `':'`) and optional receiver identifier before it.
    member_trigger: Option<(char, Option<String>)>,
}

/// Computes the autocomplete context at `cursor` (char index) in `src`.
/// Triggers when typing an identifier (>= 1 char) or immediately after `.` or `:`.
fn autocomplete_context(src: &str, cursor: usize) -> Option<AcContext> {
    let chars: Vec<char> = src.chars().collect();
    if cursor > chars.len() {
        return None;
    }
    let mut start = cursor;
    while start > 0 && is_ident_char(chars[start - 1]) {
        start -= 1;
    }
    let prefix: String = chars[start..cursor].iter().collect();

    // Check if preceded by `.` or `:`.
    if start > 0 && (chars[start - 1] == '.' || chars[start - 1] == ':') {
        let trigger = chars[start - 1];
        // Don't trigger on `..` (string concatenation) or numeric literals like `1.5`.
        if trigger == '.' && start >= 2 && (chars[start - 2] == '.' || chars[start - 2].is_ascii_digit()) {
            return None;
        }
        let mut recv_end = start - 1;
        let mut recv_start = recv_end;
        while recv_start > 0 && is_ident_char(chars[recv_start - 1]) {
            recv_start -= 1;
        }
        // Only treat as a clean identifier receiver if it starts with a letter/underscore.
        let receiver = if recv_start < recv_end && is_ident_start(chars[recv_start]) {
            Some(chars[recv_start..recv_end].iter().collect())
        } else {
            None
        };
        let _ = &mut recv_end;
        return Some(AcContext {
            replace_start: start,
            prefix,
            member_trigger: Some((trigger, receiver)),
        });
    }

    // Standard identifier trigger: at least 1 char starting with ident start.
    if !prefix.is_empty() && prefix.chars().next().is_some_and(is_ident_start) {
        return Some(AcContext {
            replace_start: start,
            prefix,
            member_trigger: None,
        });
    }

    None
}

/// Builds ranked autocomplete suggestions for `src` at `cursor`, merging any
/// plugin-provided completions (`plugin_items`).
fn build_completions(
    src: &str,
    cursor: usize,
    plugin_items: &[(String, String)],
) -> Option<(AcContext, Vec<AcItem>)> {
    let ctx = autocomplete_context(src, cursor)?;
    let prefix_lower = ctx.prefix.to_ascii_lowercase();
    let mut candidates: Vec<AcItem> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let mut add = |label: &str, insert: &str, kind: AcKind, detail: &'static str| {
        if !prefix_lower.is_empty() {
            let ll = label.to_ascii_lowercase();
            if !ll.contains(&prefix_lower) {
                return;
            }
            // Don't suggest an exact no-op completion unless there are other items.
        }
        if seen.insert(insert.to_string()) {
            candidates.push(AcItem {
                label: label.to_string(),
                insert: insert.to_string(),
                kind,
                detail,
            });
        }
    };

    // Plugin completions first if available.
    for (label, insert) in plugin_items {
        add(label, insert, AcKind::Plugin, "plugin");
    }

    match &ctx.member_trigger {
        Some((trigger, receiver)) => {
            if *trigger == '.' {
                if let Some(recv) = receiver.as_deref() {
                    if let Some(members) = receiver_members(recv) {
                        for &(name, detail, kind) in members {
                            add(name, name, kind, detail);
                        }
                    }
                }
            }
            if *trigger == ':' {
                for &(name, detail) in MEMBER_METHODS {
                    add(name, name, AcKind::Method, detail);
                }
                for &(name, detail) in MEMBER_PROPERTIES {
                    add(name, name, AcKind::Property, detail);
                }
            } else {
                for &(name, detail) in MEMBER_PROPERTIES {
                    add(name, name, AcKind::Property, detail);
                }
                for &(name, detail) in MEMBER_METHODS {
                    add(name, name, AcKind::Method, detail);
                }
            }
        }
        None => {
            for &kw in KEYWORDS {
                add(kw, kw, AcKind::Keyword, "keyword");
            }
            for &(name, detail) in STDLIB_FUNCTIONS {
                add(name, name, AcKind::Stdlib, detail);
            }
            for &(name, detail) in ROBLOX_GLOBALS {
                add(name, name, AcKind::Global, detail);
            }
            for ident in scrape_dynamic_identifiers(src) {
                // Skip if it's solely the exact prefix being typed right now.
                if ident == ctx.prefix {
                    continue;
                }
                add(&ident, &ident, AcKind::Local, "local");
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // If the only match is identical to what's already typed, don't pop up.
    if candidates.len() == 1 && candidates[0].insert == ctx.prefix {
        return None;
    }

    // Rank: exact prefix match (case-sensitive) < case-insensitive prefix match < substring match,
    // then prefer dynamic locals / keywords / stdlib / globals, then shorter / alphabetical.
    candidates.sort_by(|a, b| {
        let score = |item: &AcItem| -> (u8, u8, usize) {
            let match_tier = if ctx.prefix.is_empty() {
                0
            } else if item.label.starts_with(&ctx.prefix) {
                0
            } else if item.label.to_ascii_lowercase().starts_with(&prefix_lower) {
                1
            } else {
                2
            };
            let kind_tier = match item.kind {
                AcKind::Plugin => 0,
                AcKind::Local => 1,
                AcKind::Method if matches!(ctx.member_trigger, Some((':', _))) => 1,
                AcKind::Property if matches!(ctx.member_trigger, Some(('.', _))) => 1,
                AcKind::Keyword => 2,
                AcKind::Stdlib => 3,
                AcKind::Global => 4,
                AcKind::Method => 5,
                AcKind::Property => 6,
            };
            (match_tier, kind_tier, item.label.len())
        };
        score(a).cmp(&score(b)).then_with(|| a.label.cmp(&b.label))
    });

    candidates.truncate(24);
    Some((ctx, candidates))
}

// ------------------------------------------------------------------ auto-closing & smart blocks

/// Strips a trailing single-line `-- ...` comment outside of string literals.
fn strip_line_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut i = 0;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = quote {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        if b == b'"' || b == b'\'' || b == b'`' {
            quote = Some(b);
            i += 1;
            continue;
        }
        if b == b'-' && bytes.get(i + 1) == Some(&b'-') {
            return &line[..i];
        }
        i += 1;
    }
    line
}

fn ends_with_keyword(s: &str, kw: &str) -> bool {
    if !s.ends_with(kw) {
        return false;
    }
    let prefix_len = s.len() - kw.len();
    if prefix_len == 0 {
        return true;
    }
    s[..prefix_len]
        .chars()
        .next_back()
        .is_none_or(|c| !is_ident_char(c))
}

/// Checks if `line` (a single line before Enter) opens a Luau block and returns
/// the corresponding closing keyword (`"end"` or `"until "`).
fn line_block_closer(raw_line: &str) -> Option<&'static str> {
    let trimmed = strip_line_comment(raw_line).trim();
    if trimmed.is_empty() {
        return None;
    }
    // `elseif ... then` continues an existing `if` block; do not insert another `end`.
    if trimmed.starts_with("elseif")
        && trimmed
            .as_bytes()
            .get(6)
            .is_none_or(|b| !b.is_ascii_alphanumeric() && *b != b'_')
    {
        return None;
    }
    if ends_with_keyword(trimmed, "repeat") {
        return Some("until ");
    }
    if ends_with_keyword(trimmed, "then") || ends_with_keyword(trimmed, "do") {
        return Some("end");
    }
    // Function header ending with `)` (or return type annotation like `): number`) without `end` on the same line.
    if !ends_with_keyword(trimmed, "end") {
        let tokens = tokenize(trimmed);
        let has_fn = tokens
            .iter()
            .any(|(r, t)| *t == Tok::Keyword && &trimmed[r.clone()] == "function");
        let has_end = tokens
            .iter()
            .any(|(r, t)| *t == Tok::Keyword && &trimmed[r.clone()] == "end");
        if has_fn && !has_end && trimmed.contains('(') && trimmed.contains(')') {
            return Some("end");
        }
    }
    None
}

/// Computes `(end_balance, repeat_balance)` across the whole Luau script.
/// Positive means there are more unclosed openers than closers.
fn block_balance(src: &str) -> (isize, isize) {
    let mut end_bal: isize = 0;
    let mut repeat_bal: isize = 0;
    for (range, tok) in tokenize(src) {
        if tok != Tok::Keyword {
            continue;
        }
        match &src[range] {
            // In Luau: `if ... then`, `for ... do`, `while ... do`, bare `do`,
            // and `function ...` each close with `end`. Counting `do` covers
            // `for`, `while`, and bare `do` without double-counting.
            "if" | "do" | "function" => end_bal += 1,
            "end" => end_bal -= 1,
            "repeat" => repeat_bal += 1,
            "until" => repeat_bal -= 1,
            _ => {}
        }
    }
    (end_bal, repeat_bal)
}

fn matching_closer(open: char) -> Option<char> {
    match open {
        '(' => Some(')'),
        '[' => Some(']'),
        '{' => Some('}'),
        '"' => Some('"'),
        '\'' => Some('\''),
        _ => None,
    }
}

fn is_closing_pair_char(c: char) -> bool {
    matches!(c, ')' | ']' | '}' | '"' | '\'')
}

/// Applies auto-closing brackets/quotes, smart skip-over, smart backspace on
/// empty pairs, and smart Luau block auto-close on Enter by diffing the text
/// before and after a single `TextEdit` frame.
///
/// Returns `Some((updated_text, new_cursor_char_index))` if a transform applied.
fn apply_auto_edit(
    prev_text: &str,
    prev_cursor: usize,
    new_text: &str,
    new_cursor: usize,
) -> Option<(String, usize)> {
    let prev_chars: Vec<char> = prev_text.chars().collect();
    let mut new_chars: Vec<char> = new_text.chars().collect();

    // Case 1: Single character inserted at `prev_cursor` (`new_cursor == prev_cursor + 1`).
    if new_chars.len() == prev_chars.len() + 1 && new_cursor == prev_cursor + 1 && new_cursor >= 1 {
        let ch = new_chars[new_cursor - 1];
        let next_ch = new_chars.get(new_cursor).copied();
        let prev_ch = if new_cursor >= 2 {
            Some(new_chars[new_cursor - 2])
        } else {
            None
        };

        // 1a. Smart skip-over: typing a closing char right before an identical closing char.
        if is_closing_pair_char(ch) && next_ch == Some(ch) {
            // For quotes, avoid skipping if escaped (`\"`).
            if !(matches!(ch, '"' | '\'') && prev_ch == Some('\\')) {
                new_chars.remove(new_cursor - 1);
                return Some((new_chars.into_iter().collect(), new_cursor));
            }
        }

        // 1b. Auto-closing opening bracket or quote.
        if let Some(close) = matching_closer(ch) {
            let is_quote = matches!(ch, '"' | '\'');
            let skip_quote = is_quote
                && (prev_ch.is_some_and(|c| c == '\\' || is_ident_char(c))
                    || next_ch.is_some_and(is_ident_char));
            let skip_bracket = !is_quote && next_ch.is_some_and(is_ident_char);
            if !skip_quote && !skip_bracket {
                new_chars.insert(new_cursor, close);
                return Some((new_chars.into_iter().collect(), new_cursor));
            }
        }

        // 1c. Smart Luau block auto-close on Enter (`\n`).
        if ch == '\n' {
            let line_end = new_cursor - 1;
            let mut line_start = line_end;
            while line_start > 0 && new_chars[line_start - 1] != '\n' {
                line_start -= 1;
            }
            let prev_line: String = new_chars[line_start..line_end].iter().collect();
            let parent_indent: String = prev_line
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect();
            let indent_unit = if parent_indent.contains('\t') { "\t" } else { "    " };

            if let Some(closer) = line_block_closer(&prev_line) {
                let (end_bal, repeat_bal) = block_balance(new_text);
                let unbalanced = match closer {
                    "until " => repeat_bal > 0,
                    _ => end_bal > 0,
                };
                if unbalanced {
                    let middle_indent = format!("{parent_indent}{indent_unit}");
                    let insert_str = format!("{middle_indent}\n{parent_indent}{closer}");
                    let cursor_offset = middle_indent.chars().count();
                    let insert_chars: Vec<char> = insert_str.chars().collect();
                    new_chars.splice(new_cursor..new_cursor, insert_chars);
                    return Some((new_chars.into_iter().collect(), new_cursor + cursor_offset));
                }
            }

            // Preserve parent indentation on normal Enter if indented.
            if !parent_indent.is_empty() {
                let indent_chars: Vec<char> = parent_indent.chars().collect();
                let len = indent_chars.len();
                new_chars.splice(new_cursor..new_cursor, indent_chars);
                return Some((new_chars.into_iter().collect(), new_cursor + len));
            }
        }
    }

    // Case 2: Smart backspace deleting an empty pair `()`, `[]`, `{}`, `""`, `''`.
    if new_chars.len() + 1 == prev_chars.len()
        && prev_cursor == new_cursor + 1
        && new_cursor < prev_chars.len()
    {
        let deleted = prev_chars[new_cursor];
        if let Some(expected_close) = matching_closer(deleted) {
            if new_chars.get(new_cursor).copied() == Some(expected_close) {
                new_chars.remove(new_cursor);
                return Some((new_chars.into_iter().collect(), new_cursor));
            }
        }
    }

    None
}

// ------------------------------------------------------------------ syntax highlighting

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tok {
    Plain,
    Keyword,
    Global,
    Str,
    Num,
    Comment,
    Op,
}

/// Splits Luau source into coloured spans (byte ranges).
fn tokenize(src: &str) -> Vec<(std::ops::Range<usize>, Tok)> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    // `[[ ... ]]` / `[==[ ... ]==]` long brackets; returns the end index.
    let long_bracket = |at: usize| -> Option<usize> {
        if b.get(at) != Some(&b'[') {
            return None;
        }
        let mut j = at + 1;
        let mut level = 0;
        while b.get(j) == Some(&b'=') {
            level += 1;
            j += 1;
        }
        if b.get(j) != Some(&b'[') {
            return None;
        }
        let close: String = format!("]{}]", "=".repeat(level));
        Some(
            src[j + 1..]
                .find(&close)
                .map(|p| j + 1 + p + close.len())
                .unwrap_or(src.len()),
        )
    };
    while i < b.len() {
        let c = b[i];
        let start = i;
        if c == b'-' && b.get(i + 1) == Some(&b'-') {
            i = long_bracket(i + 2)
                .unwrap_or_else(|| src[i..].find('\n').map(|p| i + p).unwrap_or(src.len()));
            out.push((start..i, Tok::Comment));
        } else if c == b'"' || c == b'\'' || c == b'`' {
            i += 1;
            while i < b.len() && b[i] != c && b[i] != b'\n' {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i = (i + 1).min(b.len());
            out.push((start..i, Tok::Str));
        } else if let Some(end) = (c == b'[').then(|| long_bracket(i)).flatten() {
            i = end;
            out.push((start..i, Tok::Str));
        } else if c.is_ascii_digit()
            || (c == b'.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.' || b[i] == b'_') {
                i += 1;
            }
            out.push((start..i, Tok::Num));
        } else if c.is_ascii_alphabetic() || c == b'_' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let word = &src[start..i];
            let after_dot = start > 0 && b[start - 1] == b'.' || start > 0 && b[start - 1] == b':';
            let t = if KEYWORDS.contains(&word) {
                Tok::Keyword
            } else if GLOBALS.contains(&word) && !after_dot {
                Tok::Global
            } else {
                Tok::Plain
            };
            out.push((start..i, t));
        } else if c.is_ascii_punctuation() {
            i += 1;
            out.push((start..i, Tok::Op));
        } else {
            // Whitespace and anything else (including multi-byte UTF-8).
            i += 1;
            while i < b.len() && !src.is_char_boundary(i) {
                i += 1;
            }
            out.push((start..i, Tok::Plain));
        }
    }
    out
}

/// Byte range of a 1-based line.
fn line_range(src: &str, line: usize) -> Option<std::ops::Range<usize>> {
    let mut start = 0;
    for (n, l) in src.split('\n').enumerate() {
        if n + 1 == line {
            return Some(start..start + l.len());
        }
        start += l.len() + 1;
    }
    None
}

/// A highlighted layout job; the error line gets a red wavy-looking underline.
pub fn highlight(src: &str, error_line: Option<usize>, font: FontId) -> LayoutJob {
    let mut job = LayoutJob::default();
    let err = error_line.and_then(|l| line_range(src, l));
    // Studio Settings > Script Editor > Theme.
    let sc = crate::studio_settings::script_colors();
    for (range, tok) in tokenize(src) {
        let color = match tok {
            Tok::Plain => sc.text,
            Tok::Keyword => sc.keyword,
            Tok::Global => sc.builtin,
            Tok::Str => sc.string,
            Tok::Num => sc.number,
            Tok::Comment => sc.comment,
            Tok::Op => sc.operator,
        };
        // Split the token where it crosses into / out of the error line.
        let mut cuts = vec![range.start, range.end];
        if let Some(e) = &err {
            for p in [e.start, e.end] {
                if p > range.start && p < range.end {
                    cuts.push(p);
                }
            }
        }
        cuts.sort_unstable();
        for w in cuts.windows(2) {
            let (a, z) = (w[0], w[1]);
            if a == z {
                continue;
            }
            let on_err = err
                .as_ref()
                .is_some_and(|e| a >= e.start && z <= e.end && e.start != e.end);
            let fmt = TextFormat {
                font_id: font.clone(),
                color,
                italics: tok == Tok::Comment,
                underline: if on_err {
                    Stroke::new(1.5, theme::ERROR)
                } else {
                    Stroke::NONE
                },
                background: if on_err {
                    Color32::from_rgba_unmultiplied(240, 96, 86, 28)
                } else {
                    Color32::TRANSPARENT
                },
                ..Default::default()
            };
            job.append(&src[a..z], 0.0, fmt);
        }
    }
    job
}

// ------------------------------------------------------------------ state structs

#[derive(Default)]
struct AutocompleteState {
    open: bool,
    items: Vec<AcItem>,
    selected: usize,
    replace_start: usize,
    /// Cursor index at which the popup was opened/updated.
    cursor: usize,
    /// Screen position for the popup anchor (bottom-left and top-left of caret).
    caret_bottom: Option<egui::Pos2>,
    caret_top: Option<egui::Pos2>,
}

pub struct ScriptBuf {
    pub text: String,
    pub error: Option<(usize, String)>,
    checked: Option<String>,
    edited: Instant,
    /// Plugins' ScriptEditorService analysis results (line, message) and
    /// autocomplete items (label, text) for the text they were made for.
    plugin_diags: Vec<(i64, String)>,
    plugin_checked: Option<String>,
    completions: Vec<(String, String)>,
    /// Built-in + merged Luau autocomplete popup state.
    ac: AutocompleteState,
}

#[derive(Default)]
pub struct ScriptTabs {
    pub bufs: HashMap<Ref, ScriptBuf>,
    /// Lines (1-based) to put the cursor on and scroll to once the tab
    /// draws (the profiler's "Open script at line N").
    pub goto: HashMap<Ref, usize>,
    /// Script currently maximized to fill the whole CentralPanel/screen.
    pub maximized: Option<Ref>,
    /// User's chosen orientation while actively editing a script
    /// (`true` = Portrait, `false` = Landscape).
    pub script_portrait: bool,
    /// Last orientation pushed to Android via `sync_orientation`.
    last_applied_portrait: Option<bool>,
}

fn apply_android_orientation(portrait: bool) {
    #[cfg(target_os = "android")]
    crate::android::orientation::set_portrait(portrait);
    #[cfg(not(target_os = "android"))]
    let _ = portrait;
}

impl Editor {
    /// Returns `true` when a Script Editor tab is currently maximized to
    /// fullscreen (and not suppressed by an active Playtest).
    pub fn is_script_maximized(&self) -> bool {
        self.maximized_script().is_some()
    }

    /// Returns the maximized script `Ref` if fullscreen Script Editor mode is
    /// active and the script still exists in the current DataModel.
    pub fn maximized_script(&self) -> Option<Ref> {
        if self.game.play.is_some() {
            return None;
        }
        self.scripts
            .maximized
            .filter(|&r| self.game.doc.dom.get_by_ref(r).is_some())
    }

    /// Explicitly sets or clears fullscreen Script Editor mode.
    pub fn set_script_maximized(&mut self, script: Option<Ref>) {
        self.scripts.maximized = script;
    }

    /// Renders the maximized Script Editor filling the provided `ui` (typically
    /// a fullscreen `CentralPanel` in `src/app.rs`). Returns `true` if a
    /// maximized script was rendered.
    pub fn maximized_script_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let Some(r) = self.maximized_script() else {
            if self.game.play.is_none() {
                self.scripts.maximized = None;
            }
            return false;
        };
        self.script_tab_ui(ui, r);
        true
    }

    /// Returns whether the user prefers Portrait orientation while scripting.
    pub fn is_script_portrait(&self) -> bool {
        self.scripts.script_portrait
    }

    /// Updates the desired Script Editor orientation preference.
    pub fn set_script_portrait(&mut self, portrait: bool) {
        self.scripts.script_portrait = portrait;
    }

    /// Synchronizes Android screen orientation with Script Editor activity:
    /// automatically locks back to Landscape (`set_portrait(false)`) whenever
    /// the user is NOT actively in the Script Editor (e.g. switches to the 3D
    /// viewport, closes/minimizes the script tab, or starts Playtest), and
    /// restores the user's chosen orientation when actively scripting.
    pub fn sync_orientation(&mut self, script_editor_active: bool) {
        let want_portrait =
            script_editor_active && self.game.play.is_none() && self.scripts.script_portrait;
        if self.scripts.last_applied_portrait != Some(want_portrait) {
            self.scripts.last_applied_portrait = Some(want_portrait);
            apply_android_orientation(want_portrait);
        }
    }

    pub fn script_title(&self, r: Ref) -> String {
        match self.game.doc.dom.get_by_ref(r) {
            Some(i) => i.name.clone(),
            None => "Script".to_string(),
        }
    }

    pub fn script_tab_ui(&mut self, ui: &mut egui::Ui, r: Ref) {
        self.live_draft_toolbar(ui, r);
        let Some(inst) = self.game.doc.dom.get_by_ref(r) else {
            ui.add_space(12.0);
            ui.label(RichText::new("This script is not in the DataModel being shown (it may belong to the edit copy while playing).").color(theme::TEXT_DIM));
            return;
        };
        let class = inst.class.to_string();
        let source = match self.game.doc.prop(r, "Source") {
            Some(Variant::String(s)) => s,
            _ => String::new(),
        };
        let full_name = self.full_name(r);
        let follow_live_source = self.live_script_has_draft(r) == Some(false);
        // A tab on screen is an open ScriptEditorService document.
        if !crate::plugin_host::script_docs::is_open(r) {
            crate::plugin_host::script_docs::open(r, None, true);
        }
        let buf = self.scripts.bufs.entry(r).or_insert_with(|| ScriptBuf {
            text: source.clone(),
            error: None,
            checked: None,
            edited: Instant::now(),
            plugin_diags: Vec::new(),
            plugin_checked: None,
            completions: Vec::new(),
            ac: AutocompleteState::default(),
        });
        let editor_id = egui::Id::new(("script_text", r));
        let goto = self.scripts.goto.get(&r).copied();
        let focused = ui.ctx().memory(|m| m.has_focus(editor_id));
        // Undo/redo or a script change elsewhere: follow the document.
        if (!focused || follow_live_source) && buf.text != source {
            buf.text = source.clone();
            buf.checked = None;
            buf.ac.open = false;
        }
        // Compile check once typing pauses.
        if buf.checked.as_deref() != Some(buf.text.as_str())
            && buf.edited.elapsed() > Duration::from_millis(250)
        {
            buf.error = runtime::check_syntax(&buf.text, &full_name);
            buf.checked = Some(buf.text.clone());
        }
        // Plugins' script analysis callbacks (ScriptEditorService), once
        // typing pauses too.
        let analyze = buf.plugin_checked.as_deref() != Some(buf.text.as_str())
            && buf.edited.elapsed() > Duration::from_millis(250);
        if analyze {
            buf.plugin_checked = Some(buf.text.clone());
            let diags = self
                .plugins
                .with_edit(&mut self.game.doc, |rt| {
                    let lua = &rt.lua;
                    if crate::plugin_host::api::script_editor::has_callbacks(lua).1 {
                        crate::plugin_host::api::script_editor::analyze(lua, r).unwrap_or_default()
                    } else {
                        Vec::new()
                    }
                })
                .unwrap_or_default();
            if let Some(b) = self.scripts.bufs.get_mut(&r) {
                b.plugin_diags = diags;
            }
        }

        let is_maximized = self.scripts.maximized == Some(r);
        let is_portrait = self.scripts.script_portrait;
        let mut toggle_maximize = false;
        let mut toggle_portrait = false;

        let Some(buf) = self.scripts.bufs.get_mut(&r) else { return };

        // Top toolbar: script info + syntax status + Portrait/Landscape toggle + Maximize/Exit Fullscreen.
        ui.horizontal(|ui| {
            icons::class_icon(ui, &class, 16.0);
            ui.label(RichText::new(&full_name).strong());
            ui.label(
                RichText::new(format!("{} lines", buf.text.lines().count().max(1)))
                    .small()
                    .color(theme::TEXT_DIM),
            );
            match &buf.error {
                Some((_, msg)) => {
                    ui.label(RichText::new(format!("⚠ {msg}")).color(theme::ERROR));
                }
                None => {
                    let color = Color32::from_rgb(130, 200, 120);
                    ui.label(rustudio_engine::ti::text("check", 12.0).color(color));
                    ui.label(RichText::new("no syntax errors").small().color(color));
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Keep buttons clear of rounded phone corners / nav bar edge.
                if cfg!(target_os = "android") {
                    ui.add_space(16.0);
                }
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.spacing_mut().button_padding = egui::vec2(12.0, 6.0);
                let btn_h = if cfg!(target_os = "android") { 34.0 } else { 26.0 };
                let max_btn = if is_maximized {
                    egui::Button::new(RichText::new("⤡ Exit").size(13.0).strong().color(Color32::WHITE))
                        .fill(theme::ACCENT)
                        .min_size(egui::vec2(72.0, btn_h))
                } else {
                    egui::Button::new(RichText::new("⛶ Full").size(13.0))
                        .min_size(egui::vec2(72.0, btn_h))
                };
                if ui
                    .add(max_btn)
                    .on_hover_text(if is_maximized { "Exit fullscreen" } else { "Maximize Script Editor" })
                    .clicked()
                {
                    toggle_maximize = true;
                }

                let orient_btn = if is_portrait {
                    egui::Button::new(RichText::new("↻ Port").size(13.0).strong().color(Color32::WHITE))
                        .fill(theme::ACCENT)
                        .min_size(egui::vec2(74.0, btn_h))
                } else {
                    egui::Button::new(RichText::new("↻ Land").size(13.0))
                        .min_size(egui::vec2(74.0, btn_h))
                };
                if ui
                    .add(orient_btn)
                    .on_hover_text(if is_portrait { "Switch to Landscape" } else { "Switch to Portrait" })
                    .clicked()
                {
                    toggle_portrait = true;
                }
            });
        });

        for (line, msg) in &buf.plugin_diags {
            ui.label(RichText::new(format!("⚠ line {line}: {msg}")).small().color(theme::ERROR));
        }
        ui.separator();

        // Intercept keyboard navigation for the autocomplete popup BEFORE
        // TextEdit consumes Enter/Tab/Arrows/Escape.
        let mut accept_ac_idx: Option<usize> = None;
        if focused && buf.ac.open && !buf.ac.items.is_empty() {
            let count = buf.ac.items.len();
            if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown)) {
                buf.ac.selected = (buf.ac.selected + 1) % count;
            } else if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp)) {
                buf.ac.selected = (buf.ac.selected + count - 1) % count;
            } else if ui.input_mut(|i| {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Tab)
                    || i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
            }) {
                accept_ac_idx = Some(buf.ac.selected.min(count - 1));
            } else if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
                buf.ac.open = false;
            }
        }

        // Snapshot text and collapsed cursor before TextEdit runs so we can
        // apply auto-closing, smart skip-over, smart backspace, and block close.
        let prev_text = buf.text.clone();
        let prev_cursor = egui::TextEdit::load_state(ui.ctx(), editor_id)
            .and_then(|st| st.cursor.char_range())
            .and_then(|cr| {
                let (a, b) = (cr.primary.index.0, cr.secondary.index.0);
                (a == b).then_some(a)
            });

        // Studio Settings > Script Editor > Font (size), Background Color,
        // Text Wrapping.
        let font = FontId::monospace(crate::studio_settings::font_px("Studio", "Font"));
        let row_h = ui.fonts_mut(|f| f.row_height(&font));
        let wrap = crate::studio_settings::flag("Studio", "Text Wrapping", false);
        let bg = crate::studio_settings::script_colors().background;
        ui.painter().rect_filled(ui.available_rect_before_wrap(), 0.0, bg);
        let error_line = buf.error.as_ref().map(|e| e.0);
        let mut changed = false;
        let mut caret_pos: Option<(egui::Pos2, egui::Pos2)> = None;
        let mut post_edit_cursor: Option<usize> = None;

        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .id_salt(("script_scroll", r))
            .show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    // Line numbers.
                    let lines = buf.text.split('\n').count().max(1);
                    let digits = lines.to_string().len().max(2);
                    let mut gutter = LayoutJob::default();
                    for n in 1..=lines {
                        let c = if Some(n) == error_line {
                            theme::ERROR
                        } else {
                            theme::TEXT_DIM
                        };
                        let nl = if n == lines { "" } else { "\n" };
                        gutter.append(
                            &format!("{n:>digits$}{nl}"),
                            0.0,
                            TextFormat {
                                font_id: font.clone(),
                                color: c,
                                ..Default::default()
                            },
                        );
                    }
                    let galley = ui.fonts_mut(|f| f.layout_job(gutter));
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(galley.size().x + 10.0, galley.size().y.max(row_h)),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(rect, 0.0, theme::PANEL_DARK);
                    ui.painter()
                        .galley(rect.min + egui::vec2(4.0, 2.0), galley, theme::TEXT_DIM);

                    let mut layouter = |ui: &egui::Ui,
                                        text: &dyn egui::TextBuffer,
                                        wrap_w: f32|
                     -> Arc<egui::Galley> {
                        let mut job = highlight(text.as_str(), error_line, font.clone());
                        if wrap {
                            job.wrap.max_width = wrap_w;
                        }
                        ui.fonts_mut(|f| f.layout_job(job))
                    };
                    let out = egui::TextEdit::multiline(&mut buf.text)
                        .id(editor_id)
                        .code_editor()
                        .frame(egui::Frame::NONE)
                        .desired_width(f32::INFINITY)
                        .desired_rows(lines.max(24))
                        .lock_focus(true)
                        .layouter(&mut layouter)
                        .show(ui);
                    // The soft keyboard came up under the editor: the caret
                    // scrolls into the room left above it.
                    #[cfg(target_os = "android")]
                    crate::android::keyboard::reveal_cursor(ui, &out);

                    if let Some(cr) = out.cursor_range {
                        let mut cr_rect = out.galley.pos_from_cursor(cr.primary);
                        cr_rect.max.y = cr_rect.max.y.max(cr_rect.min.y + row_h);
                        let translated = cr_rect.translate(
                            out.galley_pos.to_vec2() - egui::vec2(out.galley.rect.left(), 0.0),
                        );
                        caret_pos = Some((translated.left_bottom(), translated.left_top()));
                        let (a, b) = (cr.primary.index.0, cr.secondary.index.0);
                        if a == b {
                            post_edit_cursor = Some(a);
                        }
                    }

                    let resp = out.response.response;
                    if resp.changed() {
                        changed = true;
                        buf.edited = Instant::now();
                    }
                    if let Some(line) = goto {
                        let y = resp.rect.top() + line.saturating_sub(1) as f32 * row_h;
                        ui.scroll_to_rect(
                            egui::Rect::from_min_size(
                                egui::pos2(resp.rect.left(), y),
                                egui::vec2(1.0, row_h),
                            ),
                            Some(egui::Align::Center),
                        );
                    }
                });
            });

        if let Some((bottom, top)) = caret_pos {
            buf.ac.caret_bottom = Some(bottom);
            buf.ac.caret_top = Some(top);
        }

        // Apply auto-closing brackets/quotes, smart skip-over, smart backspace,
        // and smart Luau block auto-close on Enter.
        let mut cursor_override: Option<usize> = None;
        if changed {
            if let (Some(pc), Some(nc)) = (prev_cursor, post_edit_cursor) {
                if let Some((updated, new_cur)) = apply_auto_edit(&prev_text, pc, &buf.text, nc) {
                    buf.text = updated;
                    post_edit_cursor = Some(new_cur);
                    cursor_override = Some(new_cur);
                }
            }
        }

        // Render touch-friendly Luau autocomplete popup when open.
        if buf.ac.open && !buf.ac.items.is_empty() {
            let popup_w = 270.0_f32;
            let row_height = 30.0_f32; // >= 28pt for Android touch ergonomics
            let visible_rows = buf.ac.items.len().min(7);
            let est_h = (visible_rows as f32 * row_height + 12.0).min(230.0);
            let screen = ui.ctx().content_rect();
            let base_bottom = buf.ac.caret_bottom.unwrap_or(screen.center());
            let base_top = buf.ac.caret_top.unwrap_or(base_bottom);
            let x = base_bottom.x.clamp(screen.left() + 6.0, (screen.right() - popup_w - 6.0).max(screen.left() + 6.0));
            let y = if base_bottom.y + 4.0 + est_h <= screen.bottom() - 8.0 {
                base_bottom.y + 4.0
            } else {
                (base_top.y - est_h - 4.0).max(screen.top() + 6.0)
            };

            let area_resp = egui::Area::new(egui::Id::new(("script_ac_popup", r)))
                .order(egui::Order::Foreground)
                .fixed_pos(egui::pos2(x, y))
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style())
                        .fill(theme::PANEL)
                        .stroke(Stroke::new(1.0, theme::DIVIDER))
                        .inner_margin(egui::Margin::same(4))
                        .show(ui, |ui| {
                            ui.set_min_width(popup_w);
                            ui.set_max_width(popup_w);
                            egui::ScrollArea::vertical()
                                .id_salt(("script_ac_scroll", r))
                                .max_height(220.0)
                                .auto_shrink([false, true])
                                .show(ui, |ui| {
                                    ui.spacing_mut().item_spacing.y = 2.0;
                                    for (idx, item) in buf.ac.items.iter().enumerate() {
                                        let is_sel = idx == buf.ac.selected;
                                        let (rect, resp) = ui.allocate_exact_size(
                                            egui::vec2(popup_w - 8.0, row_height),
                                            egui::Sense::click(),
                                        );
                                        if is_sel || resp.hovered() {
                                            let fill = if is_sel {
                                                theme::ACCENT.linear_multiply(0.32)
                                            } else {
                                                theme::PANEL_DARK
                                            };
                                            ui.painter().rect_filled(rect, 4.0, fill);
                                        }
                                        let (badge, badge_color) = item.kind.badge();
                                        let badge_rect = egui::Rect::from_min_size(
                                            egui::pos2(rect.left() + 4.0, rect.center().y - 9.0),
                                            egui::vec2(28.0, 18.0),
                                        );
                                        ui.painter().rect_filled(
                                            badge_rect,
                                            3.0,
                                            badge_color.linear_multiply(0.22),
                                        );
                                        ui.painter().text(
                                            badge_rect.center(),
                                            egui::Align2::CENTER_CENTER,
                                            badge,
                                            FontId::monospace(10.0),
                                            badge_color,
                                        );
                                        let label_x = badge_rect.right() + 6.0;
                                        ui.painter().text(
                                            egui::pos2(label_x, rect.center().y),
                                            egui::Align2::LEFT_CENTER,
                                            &item.label,
                                            FontId::monospace(13.0),
                                            if is_sel { Color32::WHITE } else { theme::TEXT },
                                        );
                                        if !item.detail.is_empty() {
                                            ui.painter().text(
                                                egui::pos2(rect.right() - 6.0, rect.center().y),
                                                egui::Align2::RIGHT_CENTER,
                                                item.detail,
                                                FontId::proportional(10.5),
                                                theme::TEXT_DIM,
                                            );
                                        }
                                        if resp.clicked() {
                                            accept_ac_idx = Some(idx);
                                        }
                                    }
                                });
                        });
                });

            // Dismiss popup if the user taps outside both the editor and the popup.
            if !focused
                && accept_ac_idx.is_none()
                && !area_resp.response.contains_pointer()
                && ui.input(|i| i.pointer.any_pressed())
            {
                buf.ac.open = false;
            }
        }

        // The cursor / selection as the plugins' ScriptDocument sees it
        // (Studio's 1-based line, character), and selections they set.
        let mut state = egui::TextEdit::load_state(ui.ctx(), editor_id);
        let mut word_end: Option<usize> = post_edit_cursor;
        if let Some(st) = state.as_mut() {
            if let Some(new_cur) = cursor_override {
                st.cursor.set_char_range(Some(egui::text::CCursorRange::one(
                    egui::text::CCursor::new(new_cur),
                )));
                st.clone().store(ui.ctx(), editor_id);
                word_end = Some(new_cur);
            }
            if let Some(cr) = st.cursor.char_range() {
                let (a, b) = (cr.primary.index.0, cr.secondary.index.0);
                let pa = crate::plugin_host::script_docs::pos_of(&buf.text, a);
                let pb = crate::plugin_host::script_docs::pos_of(&buf.text, b);
                crate::plugin_host::script_docs::set_selection(r, (pa.0, pa.1, pb.0, pb.1), true);
                if a == b {
                    word_end = Some(a);
                }
            }
        }

        // Apply accepted autocomplete suggestion (from tap or Tab/Enter).
        if let Some(idx) = accept_ac_idx {
            if let Some(item) = buf.ac.items.get(idx).cloned() {
                let chars: Vec<char> = buf.text.chars().collect();
                let end = word_end.unwrap_or(buf.ac.cursor).min(chars.len());
                let start = buf.ac.replace_start.min(end);
                let insert_chars: Vec<char> = item.insert.chars().collect();
                let new_cur = start + insert_chars.len();
                buf.text = chars[..start]
                    .iter()
                    .chain(insert_chars.iter())
                    .chain(chars[end..].iter())
                    .collect();
                buf.ac.open = false;
                buf.ac.items.clear();
                buf.completions.clear();
                buf.edited = Instant::now();
                changed = true;
                word_end = Some(new_cur);
                if let Some(st) = state.as_mut() {
                    st.cursor.set_char_range(Some(egui::text::CCursorRange::one(
                        egui::text::CCursor::new(new_cur),
                    )));
                    st.clone().store(ui.ctx(), editor_id);
                }
                ui.ctx().memory_mut(|m| m.request_focus(editor_id));
            }
        }

        // Update built-in autocomplete candidates whenever text changes.
        if changed && accept_ac_idx.is_none() {
            if let Some(cur) = word_end {
                if let Some((ac_ctx, items)) = build_completions(&buf.text, cur, &buf.completions) {
                    buf.ac.open = true;
                    buf.ac.replace_start = ac_ctx.replace_start;
                    buf.ac.cursor = cur;
                    buf.ac.items = items;
                    buf.ac.selected = buf.ac.selected.min(buf.ac.items.len().saturating_sub(1));
                } else {
                    buf.ac.open = false;
                    buf.ac.items.clear();
                }
            } else {
                buf.ac.open = false;
                buf.ac.items.clear();
            }
        }

        if let (Some(line), Some(mut st)) = (goto, state.clone()) {
            // (the profiler's jump: the cursor at the line's start, focused)
            let at = crate::plugin_host::script_docs::index_of(&buf.text, line as i64, 1);
            st.cursor.set_char_range(Some(egui::text::CCursorRange::one(
                egui::text::CCursor::new(at),
            )));
            st.store(ui.ctx(), editor_id);
            ui.ctx().memory_mut(|m| m.request_focus(editor_id));
            self.scripts.goto.remove(&r);
        }
        if let Some((sel, mut st)) =
            crate::plugin_host::script_docs::take_selection_request(r).zip(state)
        {
            let a = crate::plugin_host::script_docs::index_of(&buf.text, sel.0, sel.1);
            let b = crate::plugin_host::script_docs::index_of(&buf.text, sel.2, sel.3);
            st.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                egui::text::CCursor::new(b),
                egui::text::CCursor::new(a),
            )));
            st.store(ui.ctx(), editor_id);
        }

        if changed {
            let text = self
                .scripts
                .bufs
                .get(&r)
                .map(|b| b.text.clone())
                .unwrap_or_default();
            // ScriptEditorService's TextDocumentDidChange for the person's typing.
            if let Some(c) = crate::plugin_host::script_docs::diff(&source, &text) {
                crate::plugin_host::script_docs::changed(r, vec![c]);
            }
            // Plugins' autocomplete for the word being typed.
            let pos = word_end.map(|i| crate::plugin_host::script_docs::pos_of(&text, i));
            self.game.doc.set_props(
                "Edit script",
                vec![(r, rbx_dom_weak::ustr("Source"), Variant::String(text.clone()))],
                Some(format!("source:{r:?}")),
            );
            let typing_word = word_end.is_some_and(|i| {
                let c: Vec<char> = text.chars().collect();
                i > 0
                    && (c[i - 1].is_alphanumeric()
                        || c[i - 1] == '_'
                        || c[i - 1] == '.'
                        || c[i - 1] == ':')
            });
            let items = match (pos, typing_word) {
                (Some((line, ch)), true) => self
                    .plugins
                    .with_edit(&mut self.game.doc, |rt| {
                        let lua = &rt.lua;
                        if crate::plugin_host::api::script_editor::has_callbacks(lua).0 {
                            crate::plugin_host::api::script_editor::autocomplete(lua, r, line, ch)
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        }
                    })
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            if let Some(b) = self.scripts.bufs.get_mut(&r) {
                b.completions = items;
                // Re-merge plugin items into the open popup if any arrived.
                if !b.completions.is_empty() && accept_ac_idx.is_none() {
                    if let Some(cur) = word_end {
                        if let Some((ac_ctx, merged)) =
                            build_completions(&b.text, cur, &b.completions)
                        {
                            b.ac.open = true;
                            b.ac.replace_start = ac_ctx.replace_start;
                            b.ac.cursor = cur;
                            b.ac.items = merged;
                            b.ac.selected = b.ac.selected.min(b.ac.items.len().saturating_sub(1));
                        }
                    }
                }
            }
        }

        // Apply toolbar button actions after mutable borrows of `buf` are done.
        if toggle_maximize {
            self.scripts.maximized = if is_maximized { None } else { Some(r) };
        }
        if toggle_portrait {
            self.scripts.script_portrait = !is_portrait;
            self.sync_orientation(true);
        }

        if buf_needs_repaint(self.scripts.bufs.get(&r)) {
            ui.ctx().request_repaint_after(Duration::from_millis(300));
        }
    }

    /// Opens the script at instance path `path` (as the profiler names it),
    /// at `line` if given. False when no script has that path.
    pub fn open_script_named(&mut self, path: &str, line: Option<u32>) -> bool {
        let found = self
            .game
            .doc
            .dom
            .descendants()
            .filter(|i| matches!(i.class.as_str(), "Script" | "LocalScript" | "ModuleScript"))
            .map(|i| i.referent())
            .find(|&r| self.full_name(r) == path);
        match found {
            Some(r) => {
                self.open_script.push(r);
                if let Some(l) = line {
                    self.scripts.goto.insert(r, l as usize);
                }
                true
            }
            None => {
                self.warn(format!("MicroProfiler: no script named {path} in the DataModel being shown"));
                false
            }
        }
    }

    fn full_name(&self, r: Ref) -> String {
        let root = self.game.doc.dom.root_ref();
        let mut parts = Vec::new();
        let mut cur = r;
        while let Some(i) = self.game.doc.dom.get_by_ref(cur) {
            if cur == root {
                break;
            }
            parts.push(i.name.clone());
            cur = i.parent();
        }
        parts.reverse();
        parts.join(".")
    }
}

fn buf_needs_repaint(b: Option<&ScriptBuf>) -> bool {
    b.is_some_and(|b| b.checked.as_deref() != Some(b.text.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens() {
        let t = tokenize("local x = 5 -- hi\nprint(\"a\")");
        let kinds: Vec<Tok> = t
            .iter()
            .filter(|(_, k)| *k != Tok::Plain || false)
            .map(|(_, k)| *k)
            .collect();
        assert_eq!(kinds[0], Tok::Keyword);
        assert!(
            kinds.contains(&Tok::Num)
                && kinds.contains(&Tok::Comment)
                && kinds.contains(&Tok::Str)
                && kinds.contains(&Tok::Global)
        );
    }

    #[test]
    fn syntax_error_line() {
        let e = runtime::check_syntax("local a = 1\nlocal = 2\n", "Workspace.S").unwrap();
        assert_eq!(e.0, 2, "{e:?}");
    }

    #[test]
    fn autocomplete_keywords_and_stdlib() {
        let (ctx, items) = build_completions("loc", 3, &[]).unwrap();
        assert_eq!(ctx.replace_start, 0);
        assert!(items.iter().any(|i| i.label == "local" && i.kind == AcKind::Keyword));

        let (_, items) = build_completions("task.", 5, &[]).unwrap();
        assert!(items.iter().any(|i| i.label == "wait"));
        assert!(items.iter().any(|i| i.label == "spawn"));

        let (_, items) = build_completions("game:", 5, &[]).unwrap();
        assert!(items.iter().any(|i| i.label == "GetService"));
        assert!(items.iter().any(|i| i.label == "FindFirstChild"));
    }

    #[test]
    fn autocomplete_dynamic_identifiers() {
        let src = "local mySpeed = 16\nfunction jumpBoost(power)\nend\nmyS";
        let (_, items) = build_completions(src, src.len(), &[]).unwrap();
        assert!(items.iter().any(|i| i.label == "mySpeed" && i.kind == AcKind::Local));
    }

    #[test]
    fn auto_close_skip_and_backspace() {
        // Auto-close `(`
        let (t, c) = apply_auto_edit("print", 5, "print(", 6).unwrap();
        assert_eq!(t, "print()");
        assert_eq!(c, 6);

        // Smart skip-over `)`
        let (t, c) = apply_auto_edit("print()", 6, "print())", 7).unwrap();
        assert_eq!(t, "print()");
        assert_eq!(c, 7);

        // Smart backspace on empty `()`
        let (t, c) = apply_auto_edit("print()", 6, "print)", 5).unwrap();
        assert_eq!(t, "print");
        assert_eq!(c, 5);

        // Auto-close quotes & skip-over
        let (t, c) = apply_auto_edit("local s = ", 10, "local s = \"", 11).unwrap();
        assert_eq!(t, "local s = \"\"");
        assert_eq!(c, 11);
        let (t, c) = apply_auto_edit("local s = \"\"", 11, "local s = \"\"\"", 12).unwrap();
        assert_eq!(t, "local s = \"\"");
        assert_eq!(c, 12);
    }

    #[test]
    fn auto_close_luau_blocks() {
        // `if ... then`
        let (t, c) = apply_auto_edit("  if x then", 11, "  if x then\n", 12).unwrap();
        assert_eq!(t, "  if x then\n      \n  end");
        assert_eq!(c, 18);

        // `function foo()`
        let (t, c) = apply_auto_edit("local function foo()", 20, "local function foo()\n", 21).unwrap();
        assert_eq!(t, "local function foo()\n    \nend");
        assert_eq!(c, 25);

        // `repeat`
        let (t, c) = apply_auto_edit("repeat", 6, "repeat\n", 7).unwrap();
        assert_eq!(t, "repeat\n    \nuntil ");
        assert_eq!(c, 11);

        // Already balanced `if ... then ... end` should NOT insert another `end`
        let balanced = "if x then\nend";
        let res = apply_auto_edit("if x then\nend", 9, "if x then\n\nend", 10);
        assert!(res.is_none(), "expected no duplicate end for balanced block: {balanced}");
    }
}
