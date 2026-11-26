//! Native implementations of the Studio Window > Script / Tools panels:
//! - Command Bar
//! - Insert Object (panel)
//! - Activity History
//! - Drafts
//! - Find in Place
//! - Object Browser
//! - Analysis (Script Analysis)
//! - Script Activity
//! - Run Script
//! - Tag Editor
//! - Task Scheduler
//!
//! Each panel exposes an `Editor::*_panel_ui(&mut self, ui: &mut egui::Ui)` method
//! (ready to dock as a `Tab` if wired into `app.rs`) and is also hosted as a
//! resizable, collapsible Studio `egui::Window` via `Editor::studio_tools_windows_ui`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};

use egui::{Color32, FontId, RichText, Sense, Vec2};
use rbx_dom_weak::types::{Ref, Tags, Variant};
use rbx_dom_weak::ustr;
use rustudio_engine::runtime::Side;
use rustudio_engine::theme;

use crate::app::Editor;
use crate::cloud_ui::DraftState;
use crate::studio_panels::class_meta;

/// Identifier for each of the 11 Studio tool panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolPanel {
    CommandBar,
    InsertObject,
    ActivityHistory,
    Drafts,
    FindInPlace,
    ObjectBrowser,
    ScriptAnalysis,
    ScriptActivity,
    RunScript,
    TagEditor,
    TaskScheduler,
}

impl ToolPanel {
    pub fn title(self) -> &'static str {
        match self {
            ToolPanel::CommandBar => "Command Bar",
            ToolPanel::InsertObject => "Insert Object",
            ToolPanel::ActivityHistory => "Activity History",
            ToolPanel::Drafts => "Drafts",
            ToolPanel::FindInPlace => "Find in Place",
            ToolPanel::ObjectBrowser => "Object Browser",
            ToolPanel::ScriptAnalysis => "Script Analysis",
            ToolPanel::ScriptActivity => "Script Activity",
            ToolPanel::RunScript => "Run Script",
            ToolPanel::TagEditor => "Tag Editor",
            ToolPanel::TaskScheduler => "Task Scheduler",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            ToolPanel::CommandBar => "command_bar",
            ToolPanel::InsertObject => "insert_object",
            ToolPanel::ActivityHistory => "activity_history",
            ToolPanel::Drafts => "drafts",
            ToolPanel::FindInPlace => "find_in_place",
            ToolPanel::ObjectBrowser => "object_browser",
            ToolPanel::ScriptAnalysis => "script_analysis",
            ToolPanel::ScriptActivity => "script_activity",
            ToolPanel::RunScript => "run_script",
            ToolPanel::TagEditor => "tag_editor",
            ToolPanel::TaskScheduler => "task_scheduler",
        }
    }

    fn default_size(self) -> Vec2 {
        match self {
            ToolPanel::CommandBar => Vec2::new(620.0, 120.0),
            ToolPanel::InsertObject => Vec2::new(300.0, 440.0),
            ToolPanel::ActivityHistory => Vec2::new(320.0, 360.0),
            ToolPanel::Drafts => Vec2::new(380.0, 340.0),
            ToolPanel::FindInPlace => Vec2::new(460.0, 380.0),
            ToolPanel::ObjectBrowser => Vec2::new(660.0, 460.0),
            ToolPanel::ScriptAnalysis => Vec2::new(500.0, 320.0),
            ToolPanel::ScriptActivity => Vec2::new(460.0, 360.0),
            ToolPanel::RunScript => Vec2::new(480.0, 360.0),
            ToolPanel::TagEditor => Vec2::new(340.0, 400.0),
            ToolPanel::TaskScheduler => Vec2::new(400.0, 380.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalTarget {
    Edit,
    Server,
    Client,
}

impl EvalTarget {
    fn label(self) -> &'static str {
        match self {
            EvalTarget::Edit => "Edit",
            EvalTarget::Server => "Server",
            EvalTarget::Client => "Client",
        }
    }
}

struct PendingEval {
    id: u64,
    side: Option<Side>,
    source_label: &'static str,
}

#[derive(Clone)]
struct FindHit {
    script: Ref,
    class: String,
    path: String,
    line: usize,
    preview: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiagSeverity {
    Error,
    Warning,
}

#[derive(Clone)]
struct AnalysisDiag {
    script: Ref,
    path: String,
    line: usize,
    severity: DiagSeverity,
    message: String,
}

#[derive(Clone)]
struct HistoryRecord {
    label: String,
}

/// Persistent UI state for all 11 Studio tool panels, stored on `UiState`.
pub struct StudioToolsState {
    // Open flags
    pub command_bar_open: bool,
    pub insert_object_open: bool,
    pub activity_history_open: bool,
    pub drafts_open: bool,
    pub find_in_place_open: bool,
    pub object_browser_open: bool,
    pub script_analysis_open: bool,
    pub script_activity_open: bool,
    pub run_script_open: bool,
    pub tag_editor_open: bool,
    pub task_scheduler_open: bool,
    focus_next: Option<ToolPanel>,

    // Shared Luau eval state
    next_eval_id: u64,
    pending_evals: Vec<PendingEval>,

    // Command Bar
    cmd_input: String,
    cmd_multiline: bool,
    cmd_target: EvalTarget,
    cmd_history: Vec<String>,
    cmd_history_idx: Option<usize>,
    cmd_last_result: Option<(bool, String)>,

    // Insert Object
    insert_query: String,
    insert_category: String,

    // Activity History mirror (synced with Document undo_len / can_undo / can_redo)
    hist_doc_id: u64,
    hist_last_rev: u64,
    hist_last_undo_len: usize,
    hist_undo_stack: Vec<HistoryRecord>,
    hist_redo_stack: Vec<HistoryRecord>,

    // Find in Place
    find_query: String,
    find_case_sensitive: bool,
    find_class_filter: u8, // 0=All, 1=Script, 2=LocalScript, 3=ModuleScript
    find_hits: Vec<FindHit>,
    find_last_query: String,
    find_last_case: bool,
    find_last_filter: u8,
    find_last_rev: u64,

    // Object Browser
    browser_query: String,
    browser_selected: String,
    browser_member_filter: String,
    browser_show_inherited: bool,
    browser_show_hidden: bool,
    browser_tab: u8, // 0=Classes, 1=Enums
    browser_selected_enum: String,

    // Script Analysis
    analysis_diags: Vec<AnalysisDiag>,
    analysis_checked_scripts: usize,
    analysis_has_run: bool,
    analysis_include_warnings: bool,

    // Script Activity
    activity_filter: String,

    // Run Script
    run_script_code: String,
    run_script_path: String,
    run_script_target: EvalTarget,
    run_script_status: Option<(bool, String)>,
    run_script_picker: Option<Receiver<Option<PathBuf>>>,

    // Tag Editor
    tag_new_name: String,
    tag_filter: String,
    tag_custom: BTreeSet<String>,
}

impl Default for StudioToolsState {
    fn default() -> Self {
        Self {
            command_bar_open: false,
            insert_object_open: false,
            activity_history_open: false,
            drafts_open: false,
            find_in_place_open: false,
            object_browser_open: false,
            script_analysis_open: false,
            script_activity_open: false,
            run_script_open: false,
            tag_editor_open: false,
            task_scheduler_open: false,
            focus_next: None,

            next_eval_id: 900_000,
            pending_evals: Vec::new(),

            cmd_input: String::new(),
            cmd_multiline: false,
            cmd_target: EvalTarget::Edit,
            cmd_history: Vec::new(),
            cmd_history_idx: None,
            cmd_last_result: None,

            insert_query: String::new(),
            insert_category: "Common".to_string(),

            hist_doc_id: 0,
            hist_last_rev: u64::MAX,
            hist_last_undo_len: 0,
            hist_undo_stack: Vec::new(),
            hist_redo_stack: Vec::new(),

            find_query: String::new(),
            find_case_sensitive: false,
            find_class_filter: 0,
            find_hits: Vec::new(),
            find_last_query: String::new(),
            find_last_case: false,
            find_last_filter: 0,
            find_last_rev: u64::MAX,

            browser_query: String::new(),
            browser_selected: "Part".to_string(),
            browser_member_filter: String::new(),
            browser_show_inherited: true,
            browser_show_hidden: false,
            browser_tab: 0,
            browser_selected_enum: "Material".to_string(),

            analysis_diags: Vec::new(),
            analysis_checked_scripts: 0,
            analysis_has_run: false,
            analysis_include_warnings: true,

            activity_filter: String::new(),

            run_script_code: "-- Enter Luau code or pick a .lua / .luau file to run\nprint(\"Hello from Run Script!\")\n".to_string(),
            run_script_path: String::new(),
            run_script_target: EvalTarget::Edit,
            run_script_status: None,
            run_script_picker: None,

            tag_new_name: String::new(),
            tag_filter: String::new(),
            tag_custom: BTreeSet::new(),
        }
    }
}

impl StudioToolsState {
    pub fn is_open(&self, p: ToolPanel) -> bool {
        match p {
            ToolPanel::CommandBar => self.command_bar_open,
            ToolPanel::InsertObject => self.insert_object_open,
            ToolPanel::ActivityHistory => self.activity_history_open,
            ToolPanel::Drafts => self.drafts_open,
            ToolPanel::FindInPlace => self.find_in_place_open,
            ToolPanel::ObjectBrowser => self.object_browser_open,
            ToolPanel::ScriptAnalysis => self.script_analysis_open,
            ToolPanel::ScriptActivity => self.script_activity_open,
            ToolPanel::RunScript => self.run_script_open,
            ToolPanel::TagEditor => self.tag_editor_open,
            ToolPanel::TaskScheduler => self.task_scheduler_open,
        }
    }

    pub fn open_mut(&mut self, p: ToolPanel) -> &mut bool {
        match p {
            ToolPanel::CommandBar => &mut self.command_bar_open,
            ToolPanel::InsertObject => &mut self.insert_object_open,
            ToolPanel::ActivityHistory => &mut self.activity_history_open,
            ToolPanel::Drafts => &mut self.drafts_open,
            ToolPanel::FindInPlace => &mut self.find_in_place_open,
            ToolPanel::ObjectBrowser => &mut self.object_browser_open,
            ToolPanel::ScriptAnalysis => &mut self.script_analysis_open,
            ToolPanel::ScriptActivity => &mut self.script_activity_open,
            ToolPanel::RunScript => &mut self.run_script_open,
            ToolPanel::TagEditor => &mut self.tag_editor_open,
            ToolPanel::TaskScheduler => &mut self.task_scheduler_open,
        }
    }

    pub fn toggle(&mut self, p: ToolPanel) {
        let open = self.open_mut(p);
        *open = !*open;
        if *open {
            self.focus_next = Some(p);
        }
    }
}

/// Curated Insert Object categories matching Roblox Studio's most used classes.
const INSERT_CATEGORIES: &[(&str, &[&str])] = &[
    (
        "Parts & 3D",
        &[
            "Part", "WedgePart", "CornerWedgePart", "TrussPart", "Seat", "VehicleSeat",
            "SpawnLocation", "Model", "Folder", "MeshPart", "SpecialMesh", "Decal",
            "Texture", "Attachment", "Highlight", "ClickDetector", "ProximityPrompt",
        ],
    ),
    (
        "Scripts",
        &["Script", "LocalScript", "ModuleScript"],
    ),
    (
        "Events & Values",
        &[
            "RemoteEvent", "RemoteFunction", "UnreliableRemoteEvent", "BindableEvent",
            "BindableFunction", "StringValue", "IntValue", "NumberValue", "BoolValue",
            "ObjectValue", "Color3Value", "Vector3Value", "CFrameValue", "BrickColorValue",
            "Configuration",
        ],
    ),
    (
        "Lighting & Effects",
        &[
            "PointLight", "SpotLight", "SurfaceLight", "ParticleEmitter", "Beam", "Trail",
            "Fire", "Smoke", "Sparkles", "Explosion", "Sky", "Atmosphere", "Clouds",
            "BloomEffect", "BlurEffect", "ColorCorrectionEffect", "DepthOfFieldEffect",
            "SunRaysEffect", "Sound", "SoundGroup",
        ],
    ),
    (
        "UI / Gui",
        &[
            "ScreenGui", "BillboardGui", "SurfaceGui", "Frame", "ScrollingFrame",
            "CanvasGroup", "TextLabel", "TextButton", "TextBox", "ImageLabel",
            "ImageButton", "ViewportFrame", "UIListLayout", "UIGridLayout",
            "UIPageLayout", "UIPadding", "UICorner", "UIStroke", "UIGradient",
            "UIAspectRatioConstraint", "UISizeConstraint", "UIScale",
        ],
    ),
    (
        "Constraints & Physics",
        &[
            "WeldConstraint", "Motor6D", "RigidConstraint", "HingeConstraint",
            "BallSocketConstraint", "PrismaticConstraint", "CylindricalConstraint",
            "RopeConstraint", "RodConstraint", "SpringConstraint", "AlignPosition",
            "AlignOrientation", "LinearVelocity", "AngularVelocity", "VectorForce",
            "Torque", "NoCollisionConstraint", "PlaneConstraint",
        ],
    ),
    (
        "Character & Tools",
        &[
            "Tool", "Humanoid", "HumanoidDescription", "Animation", "Animator",
            "Accessory", "Shirt", "Pants", "BodyColors", "ForceField", "Team",
            "Dialog", "DialogChoice",
        ],
    ),
];

impl Editor {
    /// Toggles one of the 11 Studio tool panels open or closed.
    pub fn toggle_tool_panel(&mut self, panel: ToolPanel) {
        self.ui.tools.toggle(panel);
    }

    /// Whether a Studio tool panel is currently open.
    pub fn is_tool_panel_open(&self, panel: ToolPanel) -> bool {
        self.ui.tools.is_open(panel)
    }

    /// Polls pending Luau evals and draws all open Studio tool windows.
    /// Called every frame from `studio_panels::Editor::insert_popup_ui`.
    pub fn studio_tools_windows_ui(&mut self, ctx: &egui::Context) {
        self.sync_activity_history_mirror();
        self.poll_tool_evals();
        self.poll_run_script_picker();

        const ALL: [ToolPanel; 11] = [
            ToolPanel::CommandBar,
            ToolPanel::InsertObject,
            ToolPanel::ActivityHistory,
            ToolPanel::Drafts,
            ToolPanel::FindInPlace,
            ToolPanel::ObjectBrowser,
            ToolPanel::ScriptAnalysis,
            ToolPanel::ScriptActivity,
            ToolPanel::RunScript,
            ToolPanel::TagEditor,
            ToolPanel::TaskScheduler,
        ];

        for panel in ALL {
            if !self.ui.tools.is_open(panel) {
                continue;
            }
            let mut open = true;
            let focus = self.ui.tools.focus_next == Some(panel);
            if focus {
                self.ui.tools.focus_next = None;
            }
            let mut win = egui::Window::new(panel.title())
                .id(egui::Id::new(("studio_tool_window", panel.key())))
                .open(&mut open)
                .default_size(panel.default_size())
                .resizable(true)
                .collapsible(true);
            if panel == ToolPanel::CommandBar {
                let rect = ctx.content_rect();
                win = win.default_pos(egui::pos2(
                    (rect.center().x - 310.0).max(rect.min.x + 12.0),
                    (rect.max.y - 160.0).max(rect.min.y + 40.0),
                ));
            }
            let resp = win.show(ctx, |ui| match panel {
                ToolPanel::CommandBar => self.command_bar_panel_ui(ui),
                ToolPanel::InsertObject => self.insert_object_panel_ui(ui),
                ToolPanel::ActivityHistory => self.activity_history_panel_ui(ui),
                ToolPanel::Drafts => self.drafts_panel_ui(ui),
                ToolPanel::FindInPlace => self.find_in_place_panel_ui(ui),
                ToolPanel::ObjectBrowser => self.object_browser_panel_ui(ui),
                ToolPanel::ScriptAnalysis => self.script_analysis_panel_ui(ui),
                ToolPanel::ScriptActivity => self.script_activity_panel_ui(ui),
                ToolPanel::RunScript => self.run_script_panel_ui(ui),
                ToolPanel::TagEditor => self.tag_editor_panel_ui(ui),
                ToolPanel::TaskScheduler => self.task_scheduler_panel_ui(ui),
            });
            if focus {
                if let Some(r) = resp {
                    ctx.move_to_top(r.response.layer_id);
                }
            }
            if !open {
                *self.ui.tools.open_mut(panel) = false;
            }
        }
    }

    // =========================================================================
    // Shared Luau execution helper (reuses MCP's exact DataModel eval path)
    // =========================================================================

    fn execute_luau_snippet(
        &mut self,
        code: &str,
        target: EvalTarget,
        source_label: &'static str,
    ) -> Result<Option<String>, String> {
        let trimmed = code.trim();
        if trimmed.is_empty() {
            return Err("No Luau code to run.".into());
        }
        let playing = self.game.is_playing();
        let effective_side = match target {
            EvalTarget::Edit => {
                if playing {
                    return Err("Play is running: switch target to Server or Client, or Stop Play first.".into());
                }
                None
            }
            EvalTarget::Server => {
                if !playing {
                    return Err("Play isn't running: switch target to Edit, or start Play/Run first.".into());
                }
                Some(Side::Server)
            }
            EvalTarget::Client => {
                if !playing {
                    return Err("Play isn't running: switch target to Edit, or start Play first.".into());
                }
                if !self.game.play_sides().contains(&Side::Client) {
                    return Err("Current Run session has no Client: switch target to Server.".into());
                }
                Some(Side::Client)
            }
        };

        self.ui.tools.next_eval_id += 1;
        let id = self.ui.tools.next_eval_id;

        match effective_side {
            None => {
                let identity = rustudio_engine::runtime::security::COMMAND;
                let ea_len = self.game.edit_audio.lengths.clone();
                let sync_done = self.plugins.with_edit_vm(&mut self.game.doc, |rt| -> Result<Option<serde_json::Value>, String> {
                    if let Ok(mut w) = rustudio_engine::runtime::world(&rt.lua) {
                        for (&r, &len) in &ea_len {
                            w.input.sound_len.insert(r, len);
                        }
                    }
                    rustudio_player::luau_eval::start(&rt.lua, code, id, identity)?;
                    rt.dispatch();
                    Ok(rustudio_player::luau_eval::poll(&rt.lua, id))
                })??;
                if let Some(val) = sync_done {
                    self.game.doc.commit_pending_script_changes(source_label);
                    Ok(Some(self.handle_eval_result(source_label, val)))
                } else {
                    self.ui.tools.pending_evals.push(PendingEval { id, side: None, source_label });
                    Ok(None)
                }
            }
            Some(side) => {
                self.game.mcp_eval_start(side, code, id)?;
                self.ui.tools.pending_evals.push(PendingEval { id, side: Some(side), source_label });
                Ok(None)
            }
        }
    }

    fn poll_tool_evals(&mut self) {
        if self.ui.tools.pending_evals.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.ui.tools.pending_evals);
        let mut still_pending = Vec::new();
        for item in pending {
            let polled = match item.side {
                None => match self.plugins.with_edit_vm(&mut self.game.doc, |rt| {
                    rustudio_player::luau_eval::poll(&rt.lua, item.id)
                }) {
                    Ok(v) => v,
                    Err(e) => Some(serde_json::json!({ "ok": false, "error": e })),
                },
                Some(side) => {
                    if !self.game.is_playing() {
                        Some(serde_json::json!({ "ok": false, "error": "Play stopped before script finished." }))
                    } else {
                        self.game.mcp_eval_poll(side, item.id)
                    }
                }
            };
            if let Some(val) = polled {
                if item.side.is_none() {
                    self.game.doc.commit_pending_script_changes(item.source_label);
                }
                let summary = self.handle_eval_result(item.source_label, val);
                if item.source_label == "Command Bar" {
                    let ok = !summary.starts_with("Error:");
                    self.ui.tools.cmd_last_result = Some((ok, summary));
                } else {
                    let ok = !summary.starts_with("Error:");
                    self.ui.tools.run_script_status = Some((ok, summary));
                }
            } else {
                still_pending.push(item);
            }
        }
        self.ui.tools.pending_evals = still_pending;
    }

    fn handle_eval_result(&mut self, source_label: &str, val: serde_json::Value) -> String {
        let ok = val.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        if ok {
            let ret = val.get("values").or_else(|| val.get("result"));
            let text = match ret {
                Some(serde_json::Value::Array(arr)) if arr.is_empty() => "Ran successfully.".to_string(),
                Some(v) => format!("=> {}", v),
                None => "Ran successfully.".to_string(),
            };
            if text != "Ran successfully." {
                self.info(format!("[{source_label}] {text}"));
            }
            text
        } else {
            let err = val
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown Luau error")
                .to_string();
            self.error(format!("[{source_label}] {err}"));
            format!("Error: {err}")
        }
    }

    fn target_selector_ui(ui: &mut egui::Ui, playing: bool, has_client: bool, target: &mut EvalTarget) {
        if playing && *target == EvalTarget::Edit {
            *target = if has_client { EvalTarget::Client } else { EvalTarget::Server };
        } else if !playing && *target != EvalTarget::Edit {
            *target = EvalTarget::Edit;
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("Target:").color(theme::TEXT_DIM).small());
            if ui.add_enabled(!playing, egui::Button::selectable(*target == EvalTarget::Edit, "Edit")).clicked() {
                *target = EvalTarget::Edit;
            }
            if ui.add_enabled(playing, egui::Button::selectable(*target == EvalTarget::Server, "Server")).clicked() {
                *target = EvalTarget::Server;
            }
            if ui.add_enabled(playing && has_client, egui::Button::selectable(*target == EvalTarget::Client, "Client")).clicked() {
                *target = EvalTarget::Client;
            }
        });
    }

    // =========================================================================
    // 1. Command Bar
    // =========================================================================

    pub fn command_bar_panel_ui(&mut self, ui: &mut egui::Ui) {
        let playing = self.game.is_playing();
        let has_client = playing && self.game.play_sides().contains(&Side::Client);
        let mut target = self.ui.tools.cmd_target;
        let mut run_now = false;

        ui.horizontal(|ui| {
            Self::target_selector_ui(ui, playing, has_client, &mut target);
            self.ui.tools.cmd_target = target;
            ui.separator();
            ui.checkbox(&mut self.ui.tools.cmd_multiline, "Multi-line");
            if !self.ui.tools.cmd_history.is_empty() {
                ui.label(
                    RichText::new(format!("({} in history, ↑/↓)", self.ui.tools.cmd_history.len()))
                        .small()
                        .color(theme::TEXT_DIM),
                );
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let input_w = (ui.available_width() - 68.0).max(120.0);
            let resp = if self.ui.tools.cmd_multiline {
                ui.add(
                    egui::TextEdit::multiline(&mut self.ui.tools.cmd_input)
                        .font(FontId::monospace(13.0))
                        .desired_rows(3)
                        .desired_width(input_w)
                        .hint_text("Enter Luau command (Ctrl+Enter to run)…"),
                )
            } else {
                ui.add(
                    egui::TextEdit::singleline(&mut self.ui.tools.cmd_input)
                        .font(FontId::monospace(13.0))
                        .desired_width(input_w)
                        .hint_text("Run a Luau command… (Enter to run, ↑/↓ history)"),
                )
            };

            if resp.has_focus() {
                if !self.ui.tools.cmd_multiline && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    run_now = true;
                } else if self.ui.tools.cmd_multiline
                    && ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Enter))
                {
                    run_now = true;
                }
                if !self.ui.tools.cmd_multiline && ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                    if !self.ui.tools.cmd_history.is_empty() {
                        let next = match self.ui.tools.cmd_history_idx {
                            None => self.ui.tools.cmd_history.len() - 1,
                            Some(idx) => idx.saturating_sub(1),
                        };
                        self.ui.tools.cmd_history_idx = Some(next);
                        self.ui.tools.cmd_input = self.ui.tools.cmd_history[next].clone();
                    }
                }
                if !self.ui.tools.cmd_multiline && ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                    if let Some(idx) = self.ui.tools.cmd_history_idx {
                        if idx + 1 < self.ui.tools.cmd_history.len() {
                            self.ui.tools.cmd_history_idx = Some(idx + 1);
                            self.ui.tools.cmd_input = self.ui.tools.cmd_history[idx + 1].clone();
                        } else {
                            self.ui.tools.cmd_history_idx = None;
                            self.ui.tools.cmd_input.clear();
                        }
                    }
                }
            }

            if ui.button("Run").clicked() {
                run_now = true;
            }
        });

        if run_now {
            let code = self.ui.tools.cmd_input.clone();
            let trimmed = code.trim();
            if !trimmed.is_empty() {
                if self.ui.tools.cmd_history.last().map(String::as_str) != Some(trimmed) {
                    self.ui.tools.cmd_history.push(trimmed.to_string());
                    if self.ui.tools.cmd_history.len() > 100 {
                        self.ui.tools.cmd_history.remove(0);
                    }
                }
                self.ui.tools.cmd_history_idx = None;
                self.info(format!("> {trimmed}"));
                match self.execute_luau_snippet(trimmed, self.ui.tools.cmd_target, "Command Bar") {
                    Ok(Some(res)) => {
                        let ok = !res.starts_with("Error:");
                        self.ui.tools.cmd_last_result = Some((ok, res));
                    }
                    Ok(None) => {
                        self.ui.tools.cmd_last_result = Some((true, format!("Running on {}…", self.ui.tools.cmd_target.label())));
                    }
                    Err(e) => {
                        self.error(format!("[Command Bar] {e}"));
                        self.ui.tools.cmd_last_result = Some((false, format!("Error: {e}")));
                    }
                }
                if !self.ui.tools.cmd_multiline {
                    self.ui.tools.cmd_input.clear();
                }
            }
        }

        if let Some((ok, msg)) = &self.ui.tools.cmd_last_result {
            ui.add_space(4.0);
            let col = if *ok { Color32::from_rgb(130, 210, 130) } else { theme::ERROR };
            ui.label(RichText::new(msg).small().color(col));
        }
    }

    // =========================================================================
    // 2. Insert Object (dockable/floating panel)
    // =========================================================================

    pub fn insert_object_panel_ui(&mut self, ui: &mut egui::Ui) {
        let playing = self.game.is_playing();
        let parent = self.insert_parent();
        let parent_path = crate::mcp::tree::path(&self.game.doc, parent);

        ui.horizontal(|ui| {
            ui.label(RichText::new("Insert into:").small().color(theme::TEXT_DIM));
            ui.label(RichText::new(&parent_path).small().strong());
        });
        if playing {
            ui.label(
                RichText::new("Stop Play to insert instances into the Edit DataModel.")
                    .small()
                    .color(theme::WARN),
            );
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.ui.tools.insert_query)
                    .hint_text("Search class…")
                    .desired_width(ui.available_width() - 48.0),
            );
            if !self.ui.tools.insert_query.is_empty() && ui.small_button("Clear").clicked() {
                self.ui.tools.insert_query.clear();
            }
        });

        // Category pills when search is empty
        let q = self.ui.tools.insert_query.trim().to_lowercase();
        if q.is_empty() {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                for cat in std::iter::once("Common")
                    .chain(INSERT_CATEGORIES.iter().map(|(c, _)| *c))
                    .chain(std::iter::once("All"))
                {
                    if ui.selectable_label(self.ui.tools.insert_category == cat, cat).clicked() {
                        self.ui.tools.insert_category = cat.to_string();
                    }
                }
            });
        }
        ui.separator();

        let mut chosen: Option<String> = None;
        egui::ScrollArea::vertical()
            .id_salt("insert_object_panel_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if !q.is_empty() || self.ui.tools.insert_category == "All" {
                    let meta = class_meta();
                    let mut classes: Vec<&str> = rustudio_core::doc::db()
                        .classes
                        .keys()
                        .copied()
                        .filter(|c| rustudio_core::doc::is_creatable(c) && (q.is_empty() || c.to_lowercase().contains(&q)))
                        .collect();
                    if q.is_empty() {
                        classes.sort_by_key(|c| {
                            let cat = meta
                                .get(*c)
                                .map(|m| m.category.as_str())
                                .filter(|s| !s.is_empty())
                                .unwrap_or("Other");
                            (cat.to_string(), (*c).to_string())
                        });
                    } else {
                        classes.sort_by_key(|c| (!c.to_lowercase().starts_with(&q), c.len(), (*c).to_string()));
                    }
                    let mut last_cat = String::new();
                    for c in classes.iter().take(400) {
                        if q.is_empty() {
                            let cat = meta
                                .get(*c)
                                .map(|m| m.category.as_str())
                                .filter(|s| !s.is_empty())
                                .unwrap_or("Other");
                            if cat != last_cat {
                                ui.add_space(4.0);
                                ui.label(RichText::new(cat).small().strong().color(theme::TEXT_DIM));
                                last_cat = cat.to_string();
                            }
                        }
                        if Self::insert_class_row(ui, c, !playing) {
                            chosen = Some((*c).to_string());
                        }
                    }
                } else {
                    let selected_cat = self.ui.tools.insert_category.as_str();
                    for &(cat_name, items) in INSERT_CATEGORIES {
                        if selected_cat != "Common" && selected_cat != cat_name {
                            continue;
                        }
                        ui.add_space(4.0);
                        ui.label(RichText::new(cat_name).small().strong().color(theme::TEXT_DIM));
                        for &c in items {
                            if !rustudio_core::doc::is_creatable(c) {
                                continue;
                            }
                            if Self::insert_class_row(ui, c, !playing) {
                                chosen = Some(c.to_string());
                            }
                        }
                    }
                }
            });

        if let Some(c) = chosen {
            if !playing {
                self.insert_class(&c, parent);
            }
        }
    }

    fn insert_class_row(ui: &mut egui::Ui, class: &str, enabled: bool) -> bool {
        let w = ui.available_width().max(120.0);
        let sense = if enabled { Sense::click() } else { Sense::hover() };
        let (r, resp) = ui.allocate_exact_size(Vec2::new(w, 22.0), sense);
        if enabled && resp.hovered() {
            ui.painter().rect_filled(r, 2.0, theme::HOVER_BG);
        }
        let icon = egui::Rect::from_min_size(r.min + Vec2::new(4.0, 3.0), Vec2::splat(16.0));
        crate::icons::paint_class_icon(ui.painter(), icon, class);
        let col = if !enabled {
            theme::TEXT_DIM
        } else if resp.hovered() {
            Color32::WHITE
        } else {
            theme::TEXT
        };
        ui.painter().text(
            egui::pos2(r.min.x + 26.0, r.center().y),
            egui::Align2::LEFT_CENTER,
            class,
            FontId::proportional(13.0),
            col,
        );
        enabled && resp.clicked()
    }

    // =========================================================================
    // 3. Activity History
    // =========================================================================

    fn sync_activity_history_mirror(&mut self) {
        let doc = &self.game.doc;
        let t = &mut self.ui.tools;
        if t.hist_doc_id != doc.id {
            t.hist_doc_id = doc.id;
            t.hist_last_rev = doc.revision;
            t.hist_last_undo_len = doc.undo_len();
            t.hist_undo_stack.clear();
            t.hist_redo_stack.clear();
            if let Some(lbl) = doc.undo_display().or_else(|| doc.can_undo()) {
                for i in 0..doc.undo_len() {
                    let name = if i + 1 == doc.undo_len() {
                        lbl.to_string()
                    } else {
                        format!("Action #{}", i + 1)
                    };
                    t.hist_undo_stack.push(HistoryRecord { label: name });
                }
            }
            return;
        }
        let cur_len = doc.undo_len();
        if doc.revision == t.hist_last_rev && cur_len == t.hist_last_undo_len {
            return;
        }
        if cur_len == 0 && doc.can_redo().is_none() {
            t.hist_undo_stack.clear();
            t.hist_redo_stack.clear();
        } else if cur_len > t.hist_last_undo_len {
            let diff = cur_len - t.hist_last_undo_len;
            // Distinguish redo vs new edit: if can_redo is None or shrank, pop from redo or push new.
            for _ in 0..diff {
                if let Some(lbl) = doc.undo_display().or_else(|| doc.can_undo()) {
                    if t.hist_redo_stack.last().is_some_and(|r| r.label == lbl) {
                        if let Some(rec) = t.hist_redo_stack.pop() {
                            t.hist_undo_stack.push(rec);
                            continue;
                        }
                    }
                    t.hist_redo_stack.clear();
                    t.hist_undo_stack.push(HistoryRecord { label: lbl.to_string() });
                }
            }
        } else if cur_len < t.hist_last_undo_len {
            let diff = t.hist_last_undo_len - cur_len;
            for _ in 0..diff {
                if let Some(rec) = t.hist_undo_stack.pop() {
                    t.hist_redo_stack.push(rec);
                } else if let Some(lbl) = doc.can_redo() {
                    t.hist_redo_stack.push(HistoryRecord { label: lbl.to_string() });
                }
            }
        } else if let Some(lbl) = doc.undo_display().or_else(|| doc.can_undo()) {
            // Merged edit on top of stack
            if let Some(top) = t.hist_undo_stack.last_mut() {
                top.label = lbl.to_string();
            }
        }
        // Keep lengths exact
        while t.hist_undo_stack.len() > cur_len {
            t.hist_undo_stack.pop();
        }
        if doc.can_redo().is_none() {
            t.hist_redo_stack.clear();
        }
        t.hist_last_rev = doc.revision;
        t.hist_last_undo_len = cur_len;
    }

    pub fn activity_history_panel_ui(&mut self, ui: &mut egui::Ui) {
        let playing = self.game.is_playing();
        let can_undo = self.game.doc.can_undo().map(str::to_string);
        let can_redo = self.game.doc.can_redo().map(str::to_string);
        let undo_count = self.game.doc.undo_len();
        let chs_enabled = self.game.doc.chs_enabled();
        let recording = self.game.doc.chs_recording_in_progress(None);

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!playing && can_undo.is_some(), egui::Button::new("Undo"))
                .on_hover_text(can_undo.as_deref().unwrap_or("Nothing to undo"))
                .clicked()
            {
                self.run_action("undo");
                self.sync_activity_history_mirror();
            }
            if ui
                .add_enabled(!playing && can_redo.is_some(), egui::Button::new("Redo"))
                .on_hover_text(can_redo.as_deref().unwrap_or("Nothing to redo"))
                .clicked()
            {
                self.run_action("redo");
                self.sync_activity_history_mirror();
            }
            ui.separator();
            ui.label(
                RichText::new(format!("{} undo · {} redo", undo_count, self.ui.tools.hist_redo_stack.len()))
                    .small()
                    .color(theme::TEXT_DIM),
            );
            if recording {
                ui.label(RichText::new("● Recording").small().color(theme::WARN));
            }
            if !chs_enabled {
                ui.label(RichText::new("(History disabled)").small().color(theme::WARN));
            }
        });
        if playing {
            ui.label(
                RichText::new("Undo/Redo is paused while Play is active.")
                    .small()
                    .color(theme::TEXT_DIM),
            );
        }
        ui.separator();

        let mut undo_steps = 0usize;
        let mut redo_steps = 0usize;

        egui::ScrollArea::vertical()
            .id_salt("activity_history_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if self.ui.tools.hist_undo_stack.is_empty() && self.ui.tools.hist_redo_stack.is_empty() {
                    ui.add_space(12.0);
                    ui.label(RichText::new("No recorded actions yet.").color(theme::TEXT_DIM));
                    return;
                }

                // Redo stack (future actions, dimmed, top of redo = next to redo)
                for (i, rec) in self.ui.tools.hist_redo_stack.iter().enumerate() {
                    let steps = self.ui.tools.hist_redo_stack.len() - i;
                    let resp = ui.add_enabled(
                        !playing,
                        egui::Button::selectable(
                            false,
                            RichText::new(format!("↷ {}", rec.label)).color(theme::TEXT_DIM),
                        ),
                    );
                    if resp.clicked() {
                        redo_steps = steps;
                    }
                }

                // Current state marker
                ui.horizontal(|ui| {
                    ui.label(RichText::new("▸ Current State").strong().color(Color32::from_rgb(120, 195, 255)));
                });

                // Undo stack (most recent first)
                let total = self.ui.tools.hist_undo_stack.len();
                for (rev_idx, rec) in self.ui.tools.hist_undo_stack.iter().rev().enumerate() {
                    let is_latest = rev_idx == 0;
                    let label = format!("#{}  {}", total - rev_idx, rec.label);
                    let text = if is_latest {
                        RichText::new(label).strong()
                    } else {
                        RichText::new(label)
                    };
                    let resp = ui.add_enabled(!playing, egui::Button::selectable(is_latest, text));
                    if resp.clicked() {
                        undo_steps = rev_idx + 1;
                    }
                }
            });

        if !playing {
            for _ in 0..undo_steps {
                self.run_action("undo");
            }
            for _ in 0..redo_steps {
                self.run_action("redo");
            }
            if undo_steps > 0 || redo_steps > 0 {
                self.sync_activity_history_mirror();
            }
        }
    }

    // =========================================================================
    // 4. Drafts
    // =========================================================================

    pub fn drafts_panel_ui(&mut self, ui: &mut egui::Ui) {
        let live_refs = self.mcp_live_drafts();
        let is_live = self.live_team_create_place().is_some();

        // Also gather open script buffers that have local modifications or are open in the editor
        let mut local_entries: Vec<(Ref, String, String, bool, usize)> = Vec::new();
        for (&r, buf) in &self.scripts.bufs {
            let Some(inst) = self.game.doc.dom.get_by_ref(r) else { continue };
            let doc_src = match self.game.doc.prop(r, "Source") {
                Some(Variant::String(s)) => s,
                _ => String::new(),
            };
            let modified = buf.text != doc_src;
            let path = crate::mcp::tree::path(&self.game.doc, r);
            local_entries.push((r, inst.class.to_string(), path, modified, buf.text.lines().count().max(1)));
        }
        local_entries.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.2.cmp(&b.2)));

        ui.horizontal(|ui| {
            let mode_text = if is_live {
                format!("Team Create live drafts: {}", live_refs.len())
            } else {
                format!("Open script buffers: {} ({} modified)", local_entries.len(), local_entries.iter().filter(|e| e.3).count())
            };
            ui.label(RichText::new(mode_text).small().color(theme::TEXT_DIM));
        });
        ui.separator();

        let mut open_ref: Option<Ref> = None;
        let mut commit_live: Option<Ref> = None;
        let mut discard_live: Option<Ref> = None;
        let mut save_local: Option<(Ref, String)> = None;
        let mut revert_local: Option<Ref> = None;

        egui::ScrollArea::vertical()
            .id_salt("drafts_panel_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if is_live {
                    ui.label(RichText::new("Team Create Drafts").strong());
                    if live_refs.is_empty() {
                        ui.label(RichText::new("No uncommitted Team Create script drafts.").small().color(theme::TEXT_DIM));
                    }
                    for r in &live_refs {
                        let state = self.live_draft_state(*r);
                        let path = crate::mcp::tree::path(&self.game.doc, *r);
                        let class = self.game.doc.class_of(*r).map(|c| c.to_string()).unwrap_or_else(|| "Script".into());
                        let (badge, col) = match state {
                            DraftState::Conflict => ("Conflict", theme::ERROR),
                            DraftState::Rejected => ("Rejected", theme::ERROR),
                            DraftState::Committing => ("Committing…", theme::WARN),
                            DraftState::Submitted => ("Submitted", Color32::from_rgb(130, 200, 120)),
                            _ => ("Draft", Color32::from_rgb(120, 185, 255)),
                        };
                        ui.horizontal(|ui| {
                            crate::icons::class_icon(ui, &class, 16.0);
                            ui.label(&path);
                            ui.label(RichText::new(badge).small().color(col));
                            if ui.small_button("Open").clicked() {
                                open_ref = Some(*r);
                            }
                            if !matches!(state, DraftState::Committing | DraftState::Submitted)
                                && ui.small_button("Commit").clicked()
                            {
                                commit_live = Some(*r);
                            }
                            if state != DraftState::Committing && ui.small_button("Revert").clicked() {
                                discard_live = Some(*r);
                            }
                        });
                    }
                    ui.add_space(8.0);
                    ui.separator();
                }

                ui.label(RichText::new("Open Script Buffers").strong());
                if local_entries.is_empty() {
                    ui.label(
                        RichText::new("No scripts are open in the Script Editor yet.")
                            .small()
                            .color(theme::TEXT_DIM),
                    );
                }
                for (r, class, path, modified, lines) in &local_entries {
                    ui.horizontal(|ui| {
                        crate::icons::class_icon(ui, class, 16.0);
                        ui.label(path);
                        ui.label(RichText::new(format!("{lines}L")).small().color(theme::TEXT_DIM));
                        if *modified {
                            ui.label(RichText::new("Modified").small().color(theme::WARN));
                        } else {
                            ui.label(RichText::new("Synced").small().color(Color32::from_rgb(130, 200, 120)));
                        }
                        if ui.small_button("Open").clicked() {
                            open_ref = Some(*r);
                        }
                        if *modified {
                            if ui.small_button("Save").clicked() {
                                if let Some(b) = self.scripts.bufs.get(r) {
                                    save_local = Some((*r, b.text.clone()));
                                }
                            }
                            if ui.small_button("Revert").clicked() {
                                revert_local = Some(*r);
                            }
                        }
                    });
                }
            });

        if let Some(r) = open_ref {
            self.open_script.push(r);
        }
        if let Some(r) = commit_live {
            match self.mcp_commit_draft(r) {
                Ok(msg) => self.info(format!("[Drafts] {msg}")),
                Err(e) => self.error(format!("[Drafts] {e}")),
            }
        }
        if let Some(r) = discard_live {
            self.discard_live_draft(r);
            self.info("[Drafts] Reverted script draft to server version.");
        }
        if let Some((r, text)) = save_local {
            self.game.doc.set_props("Save Script Draft", vec![(r, ustr("Source"), Variant::String(text))], None);
            self.info("[Drafts] Saved script buffer to DataModel.");
        }
        if let Some(r) = revert_local {
            if let Some(Variant::String(src)) = self.game.doc.prop(r, "Source") {
                if let Some(buf) = self.scripts.bufs.get_mut(&r) {
                    buf.text = src;
                }
                self.info("[Drafts] Reverted script buffer to DataModel source.");
            }
        }
    }

    // =========================================================================
    // 5. Find in Place
    // =========================================================================

    pub fn find_in_place_panel_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.ui.tools.find_query)
                    .hint_text("Find text across all scripts…")
                    .desired_width((ui.available_width() - 120.0).max(120.0)),
            );
            ui.checkbox(&mut self.ui.tools.find_case_sensitive, "Match case");
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Scope:").small().color(theme::TEXT_DIM));
            for (idx, label) in [(0u8, "All Scripts"), (1, "Script"), (2, "LocalScript"), (3, "ModuleScript")] {
                if ui.selectable_label(self.ui.tools.find_class_filter == idx, label).clicked() {
                    self.ui.tools.find_class_filter = idx;
                }
            }
        });

        let needs_rescan = self.ui.tools.find_query != self.ui.tools.find_last_query
            || self.ui.tools.find_case_sensitive != self.ui.tools.find_last_case
            || self.ui.tools.find_class_filter != self.ui.tools.find_last_filter
            || self.game.doc.revision != self.ui.tools.find_last_rev;

        if needs_rescan {
            self.rescan_find_in_place();
        }

        let count = self.ui.tools.find_hits.len();
        ui.label(
            RichText::new(if self.ui.tools.find_query.is_empty() {
                "Type a search term to scan all Script, LocalScript, and ModuleScript sources.".to_string()
            } else {
                format!("{count} match{}", if count == 1 { "" } else { "es" })
            })
            .small()
            .color(theme::TEXT_DIM),
        );
        ui.separator();

        let mut jump: Option<(Ref, usize)> = None;
        egui::ScrollArea::vertical()
            .id_salt("find_in_place_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for hit in &self.ui.tools.find_hits {
                    let resp = ui
                        .horizontal(|ui| {
                            crate::icons::class_icon(ui, &hit.class, 14.0);
                            ui.label(RichText::new(format!("{}:{}", hit.path, hit.line)).strong().small());
                            ui.label(RichText::new(&hit.preview).font(FontId::monospace(12.0)).color(theme::TEXT_DIM));
                        })
                        .response
                        .interact(Sense::click());
                    if resp.hovered() {
                        ui.painter().rect_filled(resp.rect, 2.0, theme::HOVER_BG);
                    }
                    if resp.clicked() {
                        jump = Some((hit.script, hit.line));
                    }
                }
            });

        if let Some((r, line)) = jump {
            self.scripts.goto.insert(r, line);
            self.open_script.push(r);
        }
    }

    fn rescan_find_in_place(&mut self) {
        let q = self.ui.tools.find_query.clone();
        let case_sensitive = self.ui.tools.find_case_sensitive;
        let filter = self.ui.tools.find_class_filter;
        let q_lower = q.to_lowercase();

        let mut hits = Vec::new();
        if !q.is_empty() {
            let doc = &self.game.doc;
            for inst in doc.dom.descendants() {
                let cls = inst.class.as_str();
                if !matches!(cls, "Script" | "LocalScript" | "ModuleScript") {
                    continue;
                }
                match filter {
                    1 if cls != "Script" => continue,
                    2 if cls != "LocalScript" => continue,
                    3 if cls != "ModuleScript" => continue,
                    _ => {}
                }
                let Some(Variant::String(src)) = inst.properties.get(&ustr("Source")) else {
                    continue;
                };
                let path = crate::mcp::tree::path(doc, inst.referent());
                for (idx, line) in src.lines().enumerate() {
                    let matched = if case_sensitive {
                        line.contains(&q)
                    } else {
                        line.to_lowercase().contains(&q_lower)
                    };
                    if matched {
                        let mut preview = line.trim().to_string();
                        if preview.len() > 140 {
                            preview.truncate(140);
                            preview.push('…');
                        }
                        hits.push(FindHit {
                            script: inst.referent(),
                            class: cls.to_string(),
                            path: path.clone(),
                            line: idx + 1,
                            preview,
                        });
                        if hits.len() >= 1000 {
                            break;
                        }
                    }
                }
                if hits.len() >= 1000 {
                    break;
                }
            }
        }

        self.ui.tools.find_hits = hits;
        self.ui.tools.find_last_query = q;
        self.ui.tools.find_last_case = case_sensitive;
        self.ui.tools.find_last_filter = filter;
        self.ui.tools.find_last_rev = self.game.doc.revision;
    }

    // =========================================================================
    // 6. Object Browser
    // =========================================================================

    pub fn object_browser_panel_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.selectable_label(self.ui.tools.browser_tab == 0, "Classes").clicked() {
                self.ui.tools.browser_tab = 0;
            }
            if ui.selectable_label(self.ui.tools.browser_tab == 1, "Enums").clicked() {
                self.ui.tools.browser_tab = 1;
            }
            ui.separator();
            ui.add(
                egui::TextEdit::singleline(&mut self.ui.tools.browser_query)
                    .hint_text(if self.ui.tools.browser_tab == 0 { "Filter classes…" } else { "Filter enums…" })
                    .desired_width(180.0),
            );
            if self.ui.tools.browser_tab == 0 {
                ui.checkbox(&mut self.ui.tools.browser_show_inherited, "Inherited");
                ui.checkbox(&mut self.ui.tools.browser_show_hidden, "Hidden");
            }
        });
        ui.separator();

        if self.ui.tools.browser_tab == 1 {
            self.object_browser_enums_ui(ui);
            return;
        }

        let api = rustudio_core::api::get();
        let db = rustudio_core::doc::db();
        let q = self.ui.tools.browser_query.trim().to_lowercase();

        let mut class_names: Vec<String> = if let Some(a) = api {
            a.class_order
                .iter()
                .filter(|c| q.is_empty() || c.to_lowercase().contains(&q))
                .cloned()
                .collect()
        } else {
            let mut v: Vec<String> = db
                .classes
                .keys()
                .filter(|c| q.is_empty() || c.to_lowercase().contains(&q))
                .map(|c| (*c).to_string())
                .collect();
            v.sort();
            v
        };
        if !q.is_empty() {
            class_names.sort_by_key(|c| (!c.to_lowercase().starts_with(&q), c.len(), c.clone()));
        }

        let avail = ui.available_size();
        ui.horizontal(|ui| {
            // Left pane: Class list
            ui.vertical(|ui| {
                ui.set_width((avail.x * 0.34).clamp(150.0, 240.0));
                ui.label(RichText::new(format!("{} classes", class_names.len())).small().color(theme::TEXT_DIM));
                egui::ScrollArea::vertical()
                    .id_salt("obj_browser_classes")
                    .max_height((avail.y - 24.0).max(160.0))
                    .show(ui, |ui| {
                        for name in &class_names {
                            let selected = self.ui.tools.browser_selected == *name;
                            ui.horizontal(|ui| {
                                crate::icons::class_icon(ui, name, 14.0);
                                if ui.selectable_label(selected, name).clicked() {
                                    self.ui.tools.browser_selected = name.clone();
                                }
                            });
                        }
                    });
            });

            ui.separator();

            // Right pane: Selected class details
            ui.vertical(|ui| {
                let sel = self.ui.tools.browser_selected.clone();
                ui.horizontal(|ui| {
                    crate::icons::class_icon(ui, &sel, 18.0);
                    ui.label(RichText::new(&sel).strong().size(15.0));
                    if rustudio_core::doc::is_creatable(&sel) {
                        ui.label(RichText::new("Creatable").small().color(Color32::from_rgb(130, 200, 120)));
                    }
                    if api.and_then(|a| a.classes.get(&sel)).is_some_and(|c| c.service) {
                        ui.label(RichText::new("Service").small().color(Color32::from_rgb(120, 185, 255)));
                    }
                });

                // Inheritance chain
                let mut chain = Vec::new();
                let mut cur = Some(sel.clone());
                while let Some(c) = cur {
                    chain.push(c.clone());
                    cur = if let Some(a) = api {
                        a.classes.get(&c).and_then(|cl| cl.superclass.clone())
                    } else {
                        db.classes.get(c.as_str()).and_then(|cl| cl.superclass.as_ref().map(|s| s.to_string()))
                    };
                }
                if chain.len() > 1 {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Hierarchy:").small().color(theme::TEXT_DIM));
                        for (i, ancestor) in chain.iter().enumerate() {
                            if i > 0 {
                                ui.label(RichText::new("→").small().color(theme::TEXT_DIM));
                            }
                            if ui.small_button(ancestor).clicked() {
                                self.ui.tools.browser_selected = ancestor.clone();
                            }
                        }
                    });
                }

                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.ui.tools.browser_member_filter)
                            .hint_text("Filter members…")
                            .desired_width(200.0),
                    );
                });
                ui.separator();

                let mf = self.ui.tools.browser_member_filter.trim().to_lowercase();
                let show_inherited = self.ui.tools.browser_show_inherited;
                let show_hidden = self.ui.tools.browser_show_hidden;

                egui::ScrollArea::vertical()
                    .id_salt("obj_browser_members")
                    .max_height((avail.y - 96.0).max(120.0))
                    .show(ui, |ui| {
                        if let Some(a) = api {
                            let walk_classes: Vec<&str> = if show_inherited {
                                chain.iter().map(String::as_str).collect()
                            } else {
                                vec![sel.as_str()]
                            };
                            let mut seen = BTreeSet::new();
                            let mut props = Vec::new();
                            let mut funcs = Vec::new();
                            let mut events = Vec::new();
                            let mut callbacks = Vec::new();
                            for &cls in &walk_classes {
                                let Some(cdef) = a.classes.get(cls) else { continue };
                                for mname in &cdef.order {
                                    if !seen.insert(mname.clone()) {
                                        continue;
                                    }
                                    let Some(m) = cdef.members.get(mname) else { continue };
                                    if m.hidden && !show_hidden {
                                        continue;
                                    }
                                    if !mf.is_empty() && !mname.to_lowercase().contains(&mf) {
                                        continue;
                                    }
                                    let entry = (mname.clone(), cls.to_string(), m.read_only, m.deprecated, m.security.clone());
                                    match m.kind {
                                        rustudio_core::api::Kind::Property => props.push(entry),
                                        rustudio_core::api::Kind::Function => funcs.push(entry),
                                        rustudio_core::api::Kind::Event => events.push(entry),
                                        rustudio_core::api::Kind::Callback => callbacks.push(entry),
                                    }
                                }
                            }
                            for (section, list, badge_col) in [
                                ("Properties", props, Color32::from_rgb(130, 195, 255)),
                                ("Methods / Functions", funcs, Color32::from_rgb(200, 160, 255)),
                                ("Events", events, Color32::from_rgb(255, 200, 110)),
                                ("Callbacks", callbacks, Color32::from_rgb(150, 220, 160)),
                            ] {
                                if list.is_empty() {
                                    continue;
                                }
                                egui::CollapsingHeader::new(format!("{section} ({})", list.len()))
                                    .default_open(section == "Properties" || section == "Methods / Functions")
                                    .show(ui, |ui| {
                                        for (name, owner, ro, dep, sec) in list {
                                            ui.horizontal(|ui| {
                                                ui.label(RichText::new(&name).color(badge_col));
                                                if owner != sel {
                                                    ui.label(RichText::new(format!("({owner})")).small().color(theme::TEXT_DIM));
                                                }
                                                if ro {
                                                    ui.label(RichText::new("RO").small().color(theme::TEXT_DIM));
                                                }
                                                if dep {
                                                    ui.label(RichText::new("Deprecated").small().color(theme::WARN));
                                                }
                                                if sec != "None" {
                                                    ui.label(RichText::new(sec).small().color(theme::TEXT_DIM));
                                                }
                                            });
                                        }
                                    });
                            }
                        } else {
                            // Fallback to compiled-in rbx_reflection database (always available on Android)
                            let walk_classes: Vec<&str> = if show_inherited {
                                chain.iter().map(String::as_str).collect()
                            } else {
                                vec![sel.as_str()]
                            };
                            let mut seen = BTreeSet::new();
                            let mut props: Vec<(String, String)> = Vec::new();
                            for &cls in &walk_classes {
                                let Some(cdef) = db.classes.get(cls) else { continue };
                                for pname in cdef.properties.keys() {
                                    let p = pname.to_string();
                                    if !seen.insert(p.clone()) {
                                        continue;
                                    }
                                    if !mf.is_empty() && !p.to_lowercase().contains(&mf) {
                                        continue;
                                    }
                                    props.push((p, cls.to_string()));
                                }
                            }
                            props.sort_by(|a, b| a.0.cmp(&b.0));
                            ui.label(RichText::new(format!("Properties ({})", props.len())).strong());
                            for (p, owner) in props {
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(&p).color(Color32::from_rgb(130, 195, 255)));
                                    if owner != sel {
                                        ui.label(RichText::new(format!("({owner})")).small().color(theme::TEXT_DIM));
                                    }
                                });
                            }
                        }
                    });
            });
        });
    }

    fn object_browser_enums_ui(&mut self, ui: &mut egui::Ui) {
        let q = self.ui.tools.browser_query.trim().to_lowercase();
        let api = rustudio_core::api::get();
        let db = rustudio_core::doc::db();

        let mut enum_names: Vec<String> = if let Some(a) = api {
            a.enums.keys().filter(|e| q.is_empty() || e.to_lowercase().contains(&q)).cloned().collect()
        } else {
            db.enums.keys().filter(|e| q.is_empty() || e.to_lowercase().contains(&q)).map(|e| (*e).to_string()).collect()
        };
        enum_names.sort();

        let avail = ui.available_size();
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width((avail.x * 0.34).clamp(150.0, 240.0));
                ui.label(RichText::new(format!("{} enums", enum_names.len())).small().color(theme::TEXT_DIM));
                egui::ScrollArea::vertical()
                    .id_salt("obj_browser_enums")
                    .max_height((avail.y - 24.0).max(160.0))
                    .show(ui, |ui| {
                        for name in &enum_names {
                            let selected = self.ui.tools.browser_selected_enum == *name;
                            if ui.selectable_label(selected, name).clicked() {
                                self.ui.tools.browser_selected_enum = name.clone();
                            }
                        }
                    });
            });
            ui.separator();
            ui.vertical(|ui| {
                let sel = self.ui.tools.browser_selected_enum.clone();
                ui.label(RichText::new(format!("Enum.{sel}")).strong().size(15.0));
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("obj_browser_enum_items")
                    .max_height((avail.y - 48.0).max(140.0))
                    .show(ui, |ui| {
                        if let Some(items) = api.and_then(|a| a.enums.get(&sel)) {
                            for (item, val) in items {
                                ui.horizontal(|ui| {
                                    ui.label(item);
                                    ui.label(RichText::new(format!("= {val}")).small().color(theme::TEXT_DIM));
                                });
                            }
                        } else if let Some(edef) = db.enums.get(sel.as_str()) {
                            let mut items: Vec<(&str, u32)> = edef.items.iter().map(|(k, v)| (k.as_ref(), *v)).collect();
                            items.sort_by_key(|(_, v)| *v);
                            for (item, val) in items {
                                ui.horizontal(|ui| {
                                    ui.label(item);
                                    ui.label(RichText::new(format!("= {val}")).small().color(theme::TEXT_DIM));
                                });
                            }
                        }
                    });
            });
        });
    }

    // =========================================================================
    // 7. Script Analysis
    // =========================================================================

    pub fn script_analysis_panel_ui(&mut self, ui: &mut egui::Ui) {
        if !self.ui.tools.analysis_has_run {
            self.run_script_analysis();
        }
        let mut refresh = false;
        ui.horizontal(|ui| {
            if ui.button("Refresh").clicked() {
                refresh = true;
            }
            if ui.checkbox(&mut self.ui.tools.analysis_include_warnings, "Include warnings").changed() {
                refresh = true;
            }
            ui.separator();
            let errs = self.ui.tools.analysis_diags.iter().filter(|d| d.severity == DiagSeverity::Error).count();
            let warns = self.ui.tools.analysis_diags.iter().filter(|d| d.severity == DiagSeverity::Warning).count();
            ui.label(
                RichText::new(format!(
                    "{} scripts checked · {} error{} · {} warning{}",
                    self.ui.tools.analysis_checked_scripts,
                    errs,
                    if errs == 1 { "" } else { "s" },
                    warns,
                    if warns == 1 { "" } else { "s" },
                ))
                .small()
                .color(theme::TEXT_DIM),
            );
        });
        if refresh {
            self.run_script_analysis();
        }
        ui.separator();

        let mut jump: Option<(Ref, usize)> = None;
        egui::ScrollArea::vertical()
            .id_salt("script_analysis_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if self.ui.tools.analysis_diags.is_empty() {
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new("No syntax errors or warnings found.")
                            .color(Color32::from_rgb(130, 200, 120)),
                    );
                    return;
                }
                for diag in &self.ui.tools.analysis_diags {
                    let (badge, col) = match diag.severity {
                        DiagSeverity::Error => ("ERR", theme::ERROR),
                        DiagSeverity::Warning => ("WARN", theme::WARN),
                    };
                    let resp = ui
                        .horizontal(|ui| {
                            ui.label(RichText::new(badge).small().strong().color(col));
                            ui.label(RichText::new(format!("{}:{}", diag.path, diag.line)).strong().small());
                            ui.label(RichText::new(&diag.message).small());
                        })
                        .response
                        .interact(Sense::click());
                    if resp.hovered() {
                        ui.painter().rect_filled(resp.rect, 2.0, theme::HOVER_BG);
                    }
                    if resp.clicked() {
                        jump = Some((diag.script, diag.line));
                    }
                }
            });

        if let Some((r, line)) = jump {
            self.scripts.goto.insert(r, line);
            self.open_script.push(r);
        }
    }

    fn run_script_analysis(&mut self) {
        let include_warnings = self.ui.tools.analysis_include_warnings;
        let doc = &self.game.doc;
        let mut diags = Vec::new();
        let mut checked = 0usize;

        for inst in doc.dom.descendants() {
            if !matches!(inst.class.as_str(), "Script" | "LocalScript" | "ModuleScript") {
                continue;
            }
            checked += 1;
            let r = inst.referent();
            let path = crate::mcp::tree::path(doc, r);
            // Prefer open editor buffer if present so live edits are analyzed immediately
            let src = self
                .scripts
                .bufs
                .get(&r)
                .map(|b| b.text.clone())
                .unwrap_or_else(|| match inst.properties.get(&ustr("Source")) {
                    Some(Variant::String(s)) => s.clone(),
                    _ => String::new(),
                });

            if let Some((line, msg)) = rustudio_engine::runtime::check_syntax(&src, &path) {
                diags.push(AnalysisDiag {
                    script: r,
                    path: path.clone(),
                    line: line.max(1),
                    severity: DiagSeverity::Error,
                    message: msg,
                });
            } else if include_warnings {
                for (idx, line) in src.lines().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("--") {
                        continue;
                    }
                    for (legacy, modern) in [("wait(", "task.wait("), ("spawn(", "task.spawn("), ("delay(", "task.delay(")] {
                        if let Some(pos) = trimmed.find(legacy) {
                            let prev = pos.checked_sub(1).and_then(|p| trimmed.as_bytes().get(p)).copied();
                            let boundary = prev.is_none_or(|b| !b.is_ascii_alphanumeric() && b != b'_' && b != b'.' && b != b':');
                            if boundary {
                                diags.push(AnalysisDiag {
                                    script: r,
                                    path: path.clone(),
                                    line: idx + 1,
                                    severity: DiagSeverity::Warning,
                                    message: format!("Deprecated global `{legacy})` — consider `{modern})`"),
                                });
                                break;
                            }
                        }
                    }
                }
            }
        }

        diags.sort_by_key(|d| (d.severity != DiagSeverity::Error, d.path.clone(), d.line));
        self.ui.tools.analysis_diags = diags;
        self.ui.tools.analysis_checked_scripts = checked;
        self.ui.tools.analysis_has_run = true;
    }

    // =========================================================================
    // 8. Script Activity
    // =========================================================================

    pub fn script_activity_panel_ui(&mut self, ui: &mut egui::Ui) {
        let playing = self.game.is_playing();
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.ui.tools.activity_filter)
                    .hint_text("Filter scripts…")
                    .desired_width(180.0),
            );
            let status = if let Some(s) = self.game.play_status() {
                format!("Live: {s}")
            } else {
                "Edit Mode".to_string()
            };
            ui.label(RichText::new(status).small().color(theme::TEXT_DIM));
        });
        ui.separator();

        let q = self.ui.tools.activity_filter.trim().to_lowercase();
        let doc = &self.game.doc;
        let mut rows: Vec<(Ref, String, String, bool, usize)> = Vec::new();
        for inst in doc.dom.descendants() {
            let cls = inst.class.as_str();
            if !matches!(cls, "Script" | "LocalScript" | "ModuleScript") {
                continue;
            }
            let r = inst.referent();
            let path = crate::mcp::tree::path(doc, r);
            if !q.is_empty() && !path.to_lowercase().contains(&q) && !cls.to_lowercase().contains(&q) {
                continue;
            }
            let disabled = matches!(inst.properties.get(&ustr("Disabled")), Some(Variant::Bool(true)));
            let lines = match inst.properties.get(&ustr("Source")) {
                Some(Variant::String(s)) => s.lines().count().max(1),
                _ => 0,
            };
            rows.push((r, cls.to_string(), path, disabled, lines));
        }
        rows.sort_by(|a, b| a.2.cmp(&b.2));

        let mut open_ref: Option<Ref> = None;
        let mut toggle_disabled: Option<(Ref, bool)> = None;

        egui::ScrollArea::vertical()
            .id_salt("script_activity_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if rows.is_empty() {
                    ui.label(RichText::new("No scripts found.").color(theme::TEXT_DIM));
                    return;
                }
                for (r, cls, path, disabled, lines) in rows {
                    ui.horizontal(|ui| {
                        crate::icons::class_icon(ui, &cls, 16.0);
                        let active_badge = if cls == "ModuleScript" {
                            ("Module", theme::TEXT_DIM)
                        } else if disabled {
                            ("Disabled", theme::TEXT_DIM)
                        } else if playing {
                            ("Running", Color32::from_rgb(120, 215, 130))
                        } else {
                            ("Ready", Color32::from_rgb(130, 185, 240))
                        };
                        ui.label(RichText::new(active_badge.0).small().color(active_badge.1));
                        if ui.link(&path).clicked() {
                            open_ref = Some(r);
                        }
                        ui.label(RichText::new(format!("{lines}L")).small().color(theme::TEXT_DIM));
                        if cls != "ModuleScript" {
                            let mut dis = disabled;
                            if ui.add_enabled(!playing, egui::Checkbox::new(&mut dis, "Disabled")).changed() {
                                toggle_disabled = Some((r, dis));
                            }
                        }
                    });
                }
            });

        if let Some(r) = open_ref {
            self.open_script.push(r);
        }
        if let Some((r, dis)) = toggle_disabled {
            self.game.doc.set_props(
                "Toggle Script Disabled",
                vec![(r, ustr("Disabled"), Variant::Bool(dis))],
                None,
            );
        }
    }

    // =========================================================================
    // 9. Run Script
    // =========================================================================

    fn start_run_script_picker(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_os = "android"))]
        {
            let (tx, rx) = mpsc::channel();
            let ctx = ctx.clone();
            let dlg = rfd::AsyncFileDialog::new()
                .set_title("Run Luau Script")
                .add_filter("Luau Scripts", &["luau", "lua", "txt"]);
            std::thread::spawn(move || {
                let picked = pollster::block_on(dlg.pick_file()).map(|h| crate::file_dialog::handle_path(&h));
                let _ = tx.send(picked);
                ctx.request_repaint();
            });
            self.ui.tools.run_script_picker = Some(rx);
        }
        #[cfg(target_os = "android")]
        {
            let _ = ctx;
            self.ui.tools.run_script_status = Some((
                true,
                "Enter a .lua/.luau path below or paste code directly to run.".to_string(),
            ));
        }
    }

    fn poll_run_script_picker(&mut self) {
        let Some(rx) = self.ui.tools.run_script_picker.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Some(path)) => {
                self.ui.tools.run_script_picker = None;
                self.ui.tools.run_script_path = path.display().to_string();
                match std::fs::read_to_string(&path) {
                    Ok(code) => {
                        self.ui.tools.run_script_code = code.clone();
                        self.info(format!("[Run Script] Loaded {}", path.display()));
                        match self.execute_luau_snippet(&code, self.ui.tools.run_script_target, "Run Script") {
                            Ok(Some(res)) => {
                                let ok = !res.starts_with("Error:");
                                self.ui.tools.run_script_status = Some((ok, res));
                            }
                            Ok(None) => {
                                self.ui.tools.run_script_status = Some((true, "Running script…".into()));
                            }
                            Err(e) => {
                                self.error(format!("[Run Script] {e}"));
                                self.ui.tools.run_script_status = Some((false, format!("Error: {e}")));
                            }
                        }
                    }
                    Err(e) => {
                        let msg = format!("Could not read {}: {e}", path.display());
                        self.error(format!("[Run Script] {msg}"));
                        self.ui.tools.run_script_status = Some((false, format!("Error: {msg}")));
                    }
                }
            }
            Ok(None) | Err(mpsc::TryRecvError::Disconnected) => {
                self.ui.tools.run_script_picker = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    pub fn run_script_panel_ui(&mut self, ui: &mut egui::Ui) {
        let playing = self.game.is_playing();
        let has_client = playing && self.game.play_sides().contains(&Side::Client);
        let mut target = self.ui.tools.run_script_target;
        let ctx = ui.ctx().clone();

        ui.horizontal(|ui| {
            Self::target_selector_ui(ui, playing, has_client, &mut target);
            self.ui.tools.run_script_target = target;
            ui.separator();
            let picking = self.ui.tools.run_script_picker.is_some();
            if ui.add_enabled(!picking, egui::Button::new("Browse .lua / .luau…")).clicked() {
                self.start_run_script_picker(&ctx);
            }
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.ui.tools.run_script_path)
                    .hint_text("Path to .lua / .luau file…")
                    .desired_width((ui.available_width() - 80.0).max(120.0)),
            );
            if ui.button("Load File").clicked() {
                let p = self.ui.tools.run_script_path.trim().to_string();
                if !p.is_empty() {
                    match std::fs::read_to_string(&p) {
                        Ok(text) => {
                            self.ui.tools.run_script_code = text;
                            self.ui.tools.run_script_status = Some((true, format!("Loaded {p}")));
                        }
                        Err(e) => {
                            self.ui.tools.run_script_status = Some((false, format!("Error: {e}")));
                        }
                    }
                }
            }
        });

        ui.add_space(4.0);
        egui::ScrollArea::vertical()
            .id_salt("run_script_editor_scroll")
            .max_height(200.0)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.ui.tools.run_script_code)
                        .font(FontId::monospace(12.5))
                        .desired_rows(10)
                        .desired_width(ui.available_width()),
                );
            });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Run Now").clicked() {
                let code = self.ui.tools.run_script_code.clone();
                match self.execute_luau_snippet(&code, self.ui.tools.run_script_target, "Run Script") {
                    Ok(Some(res)) => {
                        let ok = !res.starts_with("Error:");
                        self.ui.tools.run_script_status = Some((ok, res));
                    }
                    Ok(None) => {
                        self.ui.tools.run_script_status = Some((true, "Running…".into()));
                    }
                    Err(e) => {
                        self.error(format!("[Run Script] {e}"));
                        self.ui.tools.run_script_status = Some((false, format!("Error: {e}")));
                    }
                }
            }
            if let Some((ok, msg)) = &self.ui.tools.run_script_status {
                let col = if *ok { Color32::from_rgb(130, 210, 130) } else { theme::ERROR };
                ui.label(RichText::new(msg).small().color(col));
            }
        });
    }

    // =========================================================================
    // 10. Tag Editor (CollectionService)
    // =========================================================================

    pub fn tag_editor_panel_ui(&mut self, ui: &mut egui::Ui) {
        let playing = self.game.is_playing();
        let sel = self.game.doc.selection.clone();

        // Collect all tags in the place -> list of tagged Refs
        let mut tag_map: BTreeMap<String, Vec<Ref>> = BTreeMap::new();
        for custom in &self.ui.tools.tag_custom {
            tag_map.entry(custom.clone()).or_default();
        }
        for inst in self.game.doc.dom.descendants() {
            if let Some(Variant::Tags(tags)) = inst.properties.get(&ustr("Tags")) {
                for t in tags.iter() {
                    tag_map.entry(t.to_string()).or_default().push(inst.referent());
                }
            }
        }

        // Add new tag row
        ui.horizontal(|ui| {
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.ui.tools.tag_new_name)
                    .hint_text("New CollectionService tag…")
                    .desired_width((ui.available_width() - 80.0).max(120.0)),
            );
            let add_clicked = ui.button("Add Tag").clicked()
                || (resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
            if add_clicked {
                let name = self.ui.tools.tag_new_name.trim().to_string();
                if !name.is_empty() {
                    self.ui.tools.tag_custom.insert(name.clone());
                    self.ui.tools.tag_new_name.clear();
                    if !playing && !sel.is_empty() {
                        self.set_tag_on_selection(&name, true);
                    }
                }
            }
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.ui.tools.tag_filter)
                    .hint_text("Filter tags…")
                    .desired_width(150.0),
            );
            ui.label(
                RichText::new(format!("{} selected", sel.len()))
                    .small()
                    .color(theme::TEXT_DIM),
            );
        });
        ui.separator();

        let q = self.ui.tools.tag_filter.trim().to_lowercase();
        let mut toggle_action: Option<(String, bool)> = None;
        let mut select_refs: Option<Vec<Ref>> = None;

        egui::ScrollArea::vertical()
            .id_salt("tag_editor_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if tag_map.is_empty() {
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("No CollectionService tags in this place yet. Add one above.")
                            .small()
                            .color(theme::TEXT_DIM),
                    );
                    return;
                }
                for (tag, refs) in &tag_map {
                    if !q.is_empty() && !tag.to_lowercase().contains(&q) {
                        continue;
                    }
                    let sel_has = !sel.is_empty() && sel.iter().all(|r| refs.contains(r));
                    let sel_some = !sel.is_empty() && sel.iter().any(|r| refs.contains(r));
                    let mut checked = sel_has;

                    ui.horizontal(|ui| {
                        let cb_label = if sel_some && !sel_has {
                            format!("{tag} (partial)")
                        } else {
                            tag.clone()
                        };
                        if ui
                            .add_enabled(!playing && !sel.is_empty(), egui::Checkbox::new(&mut checked, cb_label))
                            .changed()
                        {
                            toggle_action = Some((tag.clone(), checked));
                        }
                        ui.label(RichText::new(format!("({})", refs.len())).small().color(theme::TEXT_DIM));
                        if !refs.is_empty() && ui.small_button("Select").clicked() {
                            select_refs = Some(refs.clone());
                        }
                    });
                }
            });

        if let Some((tag, add)) = toggle_action {
            self.set_tag_on_selection(&tag, add);
        }
        if let Some(refs) = select_refs {
            self.select(refs);
        }
    }

    fn set_tag_on_selection(&mut self, tag: &str, add: bool) {
        let sel = self.game.doc.selection.clone();
        if sel.is_empty() {
            return;
        }
        let mut edits = Vec::new();
        for r in sel {
            let mut list: Vec<String> = match self.game.doc.prop(r, "Tags") {
                Some(Variant::Tags(t)) => t.iter().map(|s| s.to_string()).collect(),
                _ => Vec::new(),
            };
            if add {
                if !list.iter().any(|x| x == tag) {
                    list.push(tag.to_string());
                }
            } else {
                list.retain(|x| x != tag);
            }
            let mut next = Tags::default();
            for item in &list {
                next.push(item.as_str());
            }
            edits.push((r, ustr("Tags"), Variant::Tags(next)));
        }
        let label = if add { format!("Add Tag {tag}") } else { format!("Remove Tag {tag}") };
        self.game.doc.set_props(&label, edits, None);
    }

    // =========================================================================
    // 11. Task Scheduler
    // =========================================================================

    pub fn task_scheduler_panel_ui(&mut self, ui: &mut egui::Ui) {
        let metrics = &self.perf_overlay.metrics;
        let fps = if metrics.sample_count() > 0 { metrics.smoothed_fps() } else { self.fps };
        let frame_ms = if metrics.sample_count() > 0 {
            metrics.avg_frame_ms()
        } else if self.fps > 0.0 {
            1000.0 / self.fps
        } else {
            0.0
        };
        let low1 = metrics.one_percent_low_fps();
        let cpu_ms = if metrics.sample_count() > 0 { metrics.avg_cpu_ms() } else { self.last_work_ms };
        let gpu_ms = metrics.avg_gpu_ms();
        let rss_mb = metrics.rss_mb();

        // Count DataModel stats
        let mut total_instances = 0usize;
        let mut part_count = 0usize;
        let mut script_count = 0usize;
        let mut active_scripts = 0usize;
        for inst in self.game.doc.dom.descendants() {
            total_instances += 1;
            let cls = inst.class.as_str();
            if rustudio_core::doc::is_a(cls, "BasePart") {
                part_count += 1;
            }
            if matches!(cls, "Script" | "LocalScript" | "ModuleScript") {
                script_count += 1;
                let disabled = matches!(inst.properties.get(&ustr("Disabled")), Some(Variant::Bool(true)));
                if !disabled && cls != "ModuleScript" {
                    active_scripts += 1;
                }
            }
        }

        let play_state = match &self.game.play {
            None => "Edit (Simulation Idle)".to_string(),
            Some(p) => format!("{:?} — Viewing {:?}", p.mode, p.showing),
        };

        egui::Grid::new("task_scheduler_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label(RichText::new("Simulation State").strong());
                ui.label(&play_state);
                ui.end_row();

                ui.label(RichText::new("Frame Rate (FPS)").strong());
                let low_str = match low1 {
                    Some(v) => format!("{fps:.1} fps  (1% low: {v:.1} fps)"),
                    None => format!("{fps:.1} fps"),
                };
                ui.label(low_str);
                ui.end_row();

                ui.label(RichText::new("Frame Time").strong());
                ui.label(format!("{frame_ms:.2} ms"));
                ui.end_row();

                ui.label(RichText::new("CPU Work Time").strong());
                ui.label(format!("{cpu_ms:.2} ms"));
                ui.end_row();

                ui.label(RichText::new("GPU Scene Time").strong());
                ui.label(crate::perf_overlay::format_gpu_ms(gpu_ms));
                ui.end_row();

                ui.label(RichText::new("Render Target").strong());
                ui.label(crate::perf_overlay::format_render_scale(
                    metrics.render_scale(),
                    metrics.render_target_px(),
                ));
                ui.end_row();

                if !metrics.backend().is_empty() {
                    ui.label(RichText::new("Graphics Adapter").strong());
                    ui.label(format!("{} — {}", metrics.backend(), metrics.adapter_name()));
                    ui.end_row();
                }

                ui.label(RichText::new("Process Memory (RSS)").strong());
                ui.label(match rss_mb {
                    Some(mb) => format!("{mb:.1} MB"),
                    None => "N/A".to_string(),
                });
                ui.end_row();

                ui.label(RichText::new("DataModel Instances").strong());
                ui.label(format!("{total_instances} total · {part_count} BaseParts"));
                ui.end_row();

                ui.label(RichText::new("Scripts").strong());
                ui.label(format!(
                    "{script_count} total · {active_scripts} enabled ({} open tabs)",
                    self.scripts.bufs.len()
                ));
                ui.end_row();

                ui.label(RichText::new("Undo Stack").strong());
                ui.label(format!("{} steps · rev {}", self.game.doc.undo_len(), self.game.doc.revision));
                ui.end_row();

                ui.label(RichText::new("Pending Tool Evals").strong());
                ui.label(format!("{}", self.ui.tools.pending_evals.len()));
                ui.end_row();
            });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("Open Script Activity").clicked() {
                *self.ui.tools.open_mut(ToolPanel::ScriptActivity) = true;
                self.ui.tools.focus_next = Some(ToolPanel::ScriptActivity);
            }
            if ui.button("Open MicroProfiler").clicked() {
                self.ui.panel_requests.push((crate::app::Tab::Profiler, Some(true)));
            }
        });
    }
}
