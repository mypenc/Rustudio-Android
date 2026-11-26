//! Native Animation Editor (Clip Editor & Graph Editor) and Avatar Tools:
//! - Clip Editor & Graph Editor (live Motor6D posing, timeline scrubbing,
//!   keyframe editing, Euler curve graph, and Publish Animation to Roblox)
//! - Avatar Setup & Rig Inspector (R6/R15 rig joints, Humanoid stats, Publish Model)
//! - Accessory Fitter (attach Accessory/Handle and adjust AttachmentPos/AttachmentForward)
//! - Layered Clothing / Wrap Inspector (WrapLayer / WrapTarget inspector & editor)
//! - UGC Validation (real checks on selected Model/Accessory)

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver};
use web_time::Instant;

use egui::{Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use glam::{Mat3, Quat, Vec3};
use rbx_dom_weak::types::{CFrame, Ref, Variant};
use rbx_dom_weak::{ustr, InstanceBuilder};
use rustudio_core::doc::Document;
use rustudio_core::math::{v3, Xform};
use rustudio_engine::rig::{write_pose, Rig};
use rustudio_engine::theme;

use crate::app::Editor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimViewMode {
    Clip,
    Graph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvatarToolPanel {
    AnimEditor,
    AvatarSetup,
    AccessoryFitter,
    AdaptiveWrap,
    UgcValidation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PublishStatus {
    Idle,
    Uploading,
    Success(u64),
    Error(String),
    NotSignedIn,
}

struct PublishDialog {
    asset_type: &'static str,
    target_refs: Vec<Ref>,
    name: String,
    description: String,
    status: PublishStatus,
    rx: Option<Receiver<Result<u64, String>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckLevel {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone)]
struct UgcCheckResult {
    label: String,
    detail: String,
    level: CheckLevel,
}

/// Persistent UI and playback state for Animation Editor and Avatar Tools.
pub struct AnimEditorState {
    pub anim_open: bool,
    pub plugin_popup_dismissed: bool,
    pub mode: AnimViewMode,
    pub setup_open: bool,
    pub accessory_open: bool,
    pub adaptive_open: bool,
    pub ugc_open: bool,
    focus_next: Option<AvatarToolPanel>,

    // Active rig & clip
    rig_ref: Option<Ref>,
    clip_ref: Option<Ref>,
    selected_joint: Option<String>,
    selected_kf_time: Option<f32>,

    // Playback state
    pub playing: bool,
    looping: bool,
    speed: f32,
    time: f32,
    last_tick: Option<Instant>,

    // Pose snapshot for restoring rest pose on Stop / Close
    active_pose_rig: Option<Ref>,
    rest_root: Option<Xform>,
    saved_part_cframes: Vec<(Ref, CFrame)>,

    // Publish modal
    publish: Option<PublishDialog>,

    // Accessory Fitter
    accessory_target_part: usize,

    // UGC Validation cache
    ugc_results: Vec<UgcCheckResult>,
    ugc_last_target: Option<Ref>,
}

impl Default for AnimEditorState {
    fn default() -> Self {
        Self {
            anim_open: false,
            plugin_popup_dismissed: false,
            mode: AnimViewMode::Clip,
            setup_open: false,
            accessory_open: false,
            adaptive_open: false,
            ugc_open: false,
            focus_next: None,

            rig_ref: None,
            clip_ref: None,
            selected_joint: None,
            selected_kf_time: None,

            playing: false,
            looping: true,
            speed: 1.0,
            time: 0.0,
            last_tick: None,

            active_pose_rig: None,
            rest_root: None,
            saved_part_cframes: Vec::new(),

            publish: None,

            accessory_target_part: 0,

            ugc_results: Vec::new(),
            ugc_last_target: None,
        }
    }
}

const SPEEDS: [f32; 5] = [0.25, 0.5, 1.0, 1.5, 2.0];

const ACCESSORY_ATTACH_TARGETS: [(&str, &str); 6] = [
    ("Head (HatAttachment)", "HatAttachment"),
    ("Head (HairAttachment)", "HairAttachment"),
    ("Head (FaceFrontAttachment)", "FaceFrontAttachment"),
    ("Torso / UpperTorso (BodyBackAttachment)", "BodyBackAttachment"),
    ("Torso / UpperTorso (BodyFrontAttachment)", "BodyFrontAttachment"),
    ("Torso / UpperTorso (NeckAttachment)", "NeckAttachment"),
];

// ---------------------------------------------------------------------------
// DOM Inspection Helpers
// ---------------------------------------------------------------------------

fn is_rig_model(doc: &Document, r: Ref) -> bool {
    let Some(inst) = doc.dom.get_by_ref(r) else {
        return false;
    };
    if inst.class != "Model" {
        return false;
    }
    doc.dom.descendants_of(r).any(|d| {
        d.referent() != r && matches!(d.class.as_str(), "Humanoid" | "Motor6D" | "AnimationConstraint")
    })
}

fn find_all_rigs(doc: &Document) -> Vec<(Ref, String)> {
    let Some(ws) = doc.workspace() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for inst in doc.dom.descendants_of(ws) {
        let r = inst.referent();
        if r != ws && is_rig_model(doc, r) {
            out.push((r, inst.name.clone()));
        }
    }
    out
}

fn detect_selected_rig(doc: &Document) -> Option<Ref> {
    for &sel in &doc.selection {
        let mut cur = sel;
        while cur.is_some() {
            if is_rig_model(doc, cur) {
                return Some(cur);
            }
            cur = doc.dom.get_by_ref(cur).map(|i| i.parent()).unwrap_or(Ref::none());
        }
    }
    None
}

fn find_clips(doc: &Document, rig: Option<Ref>) -> Vec<(Ref, String)> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    if let Some(rig_r) = rig {
        for inst in doc.dom.descendants_of(rig_r) {
            if inst.class == "KeyframeSequence" && seen.insert(inst.referent()) {
                out.push((inst.referent(), inst.name.clone()));
            }
        }
        // Also check siblings of rig (e.g. parent model containing rig + AnimSaves)
        if let Some(parent) = doc.dom.get_by_ref(rig_r).map(|i| i.parent()) {
            if parent.is_some() && doc.workspace() != Some(parent) {
                for inst in doc.dom.descendants_of(parent) {
                    if inst.class == "KeyframeSequence" && seen.insert(inst.referent()) {
                        out.push((inst.referent(), inst.name.clone()));
                    }
                }
            }
        }
    }
    if let Some(ws) = doc.workspace() {
        for inst in doc.dom.descendants_of(ws) {
            if inst.class == "KeyframeSequence" && seen.insert(inst.referent()) {
                out.push((inst.referent(), format!("{} (Workspace)", inst.name)));
            }
        }
    }
    out
}

#[derive(Clone)]
struct RigJointInfo {
    motor: Ref,
    motor_name: String,
    part0_name: String,
    part1_name: String,
}

fn rig_joints(doc: &Document, rig_r: Ref) -> Vec<RigJointInfo> {
    let rig = Rig::read(doc, rig_r);
    let mut out = Vec::new();
    for j in &rig.joints {
        let p0 = doc
            .dom
            .get_by_ref(j.part0)
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "Part0".into());
        let p1 = doc
            .dom
            .get_by_ref(j.part1)
            .map(|i| i.name.clone())
            .unwrap_or_else(|| j.name.clone());
        out.push(RigJointInfo {
            motor: j.motor,
            motor_name: j.name.clone(),
            part0_name: p0,
            part1_name: p1,
        });
    }
    if out.is_empty() {
        // Fallback: list all Motor6D descendants even if HumanoidRootPart is missing
        for inst in doc.dom.descendants_of(rig_r) {
            if inst.class == "Motor6D" {
                let p0 = match doc.prop(inst.referent(), "Part0") {
                    Some(Variant::Ref(r)) => doc.dom.get_by_ref(r).map(|i| i.name.clone()).unwrap_or_default(),
                    _ => String::new(),
                };
                let p1 = match doc.prop(inst.referent(), "Part1") {
                    Some(Variant::Ref(r)) => doc.dom.get_by_ref(r).map(|i| i.name.clone()).unwrap_or_else(|| inst.name.clone()),
                    _ => inst.name.clone(),
                };
                out.push(RigJointInfo {
                    motor: inst.referent(),
                    motor_name: inst.name.clone(),
                    part0_name: p0,
                    part1_name: p1,
                });
            }
        }
    }
    out
}

#[derive(Clone)]
struct KeyframeData {
    kf_ref: Ref,
    time: f32,
    poses: HashMap<String, (Ref, Xform, f32)>,
}

fn read_keyframes(doc: &Document, clip_r: Ref) -> Vec<KeyframeData> {
    let Some(clip) = doc.dom.get_by_ref(clip_r) else {
        return Vec::new();
    };
    let mut kfs = Vec::new();
    for &child in clip.children() {
        let Some(kf_inst) = doc.dom.get_by_ref(child) else {
            continue;
        };
        if kf_inst.class != "Keyframe" {
            continue;
        }
        let time = match doc.prop(child, "Time") {
            Some(Variant::Float32(t)) => t,
            Some(Variant::Float64(t)) => t as f32,
            _ => 0.0,
        };
        let mut poses = HashMap::new();
        for desc in doc.dom.descendants_of(child) {
            if desc.class == "Pose" {
                let pr = desc.referent();
                let xf = match doc.prop(pr, "CFrame") {
                    Some(Variant::CFrame(cf)) => Xform::from_cframe(&cf),
                    _ => Xform {
                        pos: Vec3::ZERO,
                        rot: Mat3::IDENTITY,
                    },
                };
                let weight = match doc.prop(pr, "Weight") {
                    Some(Variant::Float32(w)) => w,
                    Some(Variant::Float64(w)) => w as f32,
                    _ => 1.0,
                };
                poses.insert(desc.name.clone(), (pr, xf, weight));
            }
        }
        kfs.push(KeyframeData {
            kf_ref: child,
            time,
            poses,
        });
    }
    kfs.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    kfs
}

fn clip_duration(kfs: &[KeyframeData]) -> f32 {
    kfs.iter().map(|k| k.time).fold(0.0f32, f32::max).max(1.0)
}

fn slerp_xform(a: Xform, b: Xform, alpha: f32) -> Xform {
    let t = alpha.clamp(0.0, 1.0);
    let pos = a.pos.lerp(b.pos, t);
    let qa = Quat::from_mat3(&a.rot).normalize();
    let qb = Quat::from_mat3(&b.rot).normalize();
    let q = qa.slerp(qb, t);
    Xform {
        pos,
        rot: Mat3::from_quat(q),
    }
}

/// Evaluates the KeyframeSequence at time `t`, returning per-joint-name Xform.
fn evaluate_clip(kfs: &[KeyframeData], t: f32) -> HashMap<String, Xform> {
    let mut tracks: HashMap<String, Vec<(f32, Xform)>> = HashMap::new();
    for kf in kfs {
        for (name, &(_, xf, weight)) in &kf.poses {
            if weight > 0.0 {
                tracks.entry(name.clone()).or_default().push((kf.time, xf));
            }
        }
    }
    let mut out = HashMap::new();
    for (name, samples) in tracks {
        if samples.is_empty() {
            continue;
        }
        let xf = if samples.len() == 1 || t <= samples[0].0 {
            samples[0].1
        } else if t >= samples[samples.len() - 1].0 {
            samples[samples.len() - 1].1
        } else {
            let mut res = samples[0].1;
            for w in samples.windows(2) {
                if t >= w[0].0 && t <= w[1].0 {
                    let span = (w[1].0 - w[0].0).max(1e-5);
                    res = slerp_xform(w[0].1, w[1].1, (t - w[0].0) / span);
                    break;
                }
            }
            res
        };
        out.insert(name, xf);
    }
    out
}

fn part_cframe(doc: &Document, r: Ref) -> Xform {
    match doc.prop(r, "CFrame") {
        Some(Variant::CFrame(cf)) => Xform::from_cframe(&cf),
        _ => Xform {
            pos: Vec3::ZERO,
            rot: Mat3::IDENTITY,
        },
    }
}

fn cframe_identity() -> CFrame {
    Xform {
        pos: Vec3::ZERO,
        rot: Mat3::IDENTITY,
    }
    .to_cframe()
}

// ---------------------------------------------------------------------------
// Editor Integration Methods
// ---------------------------------------------------------------------------

impl Editor {
    /// Toggle or switch the native Animation Editor (Clip or Graph mode).
    pub fn toggle_native_anim_editor(&mut self, mode: AnimViewMode) {
        if self.ui.anim.anim_open && self.ui.anim.mode == mode {
            self.stop_anim_preview();
            self.ui.anim.anim_open = false;
        } else {
            self.ui.anim.anim_open = true;
            self.ui.anim.mode = mode;
            self.ui.anim.focus_next = Some(AvatarToolPanel::AnimEditor);
            self.sync_anim_selection();
        }
    }

    pub fn toggle_avatar_tool_panel(&mut self, panel: AvatarToolPanel) {
        let flag = match panel {
            AvatarToolPanel::AnimEditor => &mut self.ui.anim.anim_open,
            AvatarToolPanel::AvatarSetup => &mut self.ui.anim.setup_open,
            AvatarToolPanel::AccessoryFitter => &mut self.ui.anim.accessory_open,
            AvatarToolPanel::AdaptiveWrap => &mut self.ui.anim.adaptive_open,
            AvatarToolPanel::UgcValidation => &mut self.ui.anim.ugc_open,
        };
        *flag = !*flag;
        if *flag {
            self.ui.anim.focus_next = Some(panel);
            if panel == AvatarToolPanel::UgcValidation {
                self.run_ugc_validation();
            }
        } else if panel == AvatarToolPanel::AnimEditor {
            self.stop_anim_preview();
        }
    }

    pub fn is_avatar_tool_open(&self, panel: AvatarToolPanel) -> bool {
        match panel {
            AvatarToolPanel::AnimEditor => self.ui.anim.anim_open,
            AvatarToolPanel::AvatarSetup => self.ui.anim.setup_open,
            AvatarToolPanel::AccessoryFitter => self.ui.anim.accessory_open,
            AvatarToolPanel::AdaptiveWrap => self.ui.anim.adaptive_open,
            AvatarToolPanel::UgcValidation => self.ui.anim.ugc_open,
        }
    }

    fn sync_anim_selection(&mut self) {
        if let Some(sel_rig) = detect_selected_rig(&self.game.doc) {
            if self.ui.anim.rig_ref != Some(sel_rig) {
                self.stop_anim_preview();
                self.ui.anim.rig_ref = Some(sel_rig);
                self.ui.anim.clip_ref = None;
                self.ui.anim.selected_joint = None;
            }
        } else if self
            .ui
            .anim
            .rig_ref
            .is_none_or(|r| !self.game.doc.exists(r) || !is_rig_model(&self.game.doc, r))
        {
            self.ui.anim.rig_ref = find_all_rigs(&self.game.doc).first().map(|(r, _)| *r);
        }

        if let Some(rig_r) = self.ui.anim.rig_ref {
            if self.ui.anim.clip_ref.is_none_or(|c| !self.game.doc.exists(c)) {
                let clips = find_clips(&self.game.doc, Some(rig_r));
                if let Some((c, _)) = clips.first() {
                    self.ui.anim.clip_ref = Some(*c);
                    if let Some(Variant::Bool(l)) = self.game.doc.prop(*c, "Loop") {
                        self.ui.anim.looping = l;
                    }
                }
            }
            if self.ui.anim.selected_joint.is_none() {
                let joints = rig_joints(&self.game.doc, rig_r);
                self.ui.anim.selected_joint = joints.first().map(|j| j.part1_name.clone());
            }
        }
    }

    /// Snapshot rest pose of the rig before applying live transforms.
    fn ensure_pose_snapshot(&mut self, rig_r: Ref) {
        if self.ui.anim.active_pose_rig == Some(rig_r) {
            return;
        }
        let was_playing = self.ui.anim.playing;
        let was_tick = self.ui.anim.last_tick;
        self.stop_anim_preview();
        self.ui.anim.playing = was_playing;
        self.ui.anim.last_tick = was_tick;
        let rig = Rig::read(&self.game.doc, rig_r);
        let root_xf = if rig.root.is_some() {
            part_cframe(&self.game.doc, rig.root)
        } else {
            Xform {
                pos: Vec3::ZERO,
                rot: Mat3::IDENTITY,
            }
        };
        let mut saved = Vec::new();
        for desc in self.game.doc.dom.descendants_of(rig_r) {
            if let Some(Variant::CFrame(cf)) = self.game.doc.prop(desc.referent(), "CFrame") {
                saved.push((desc.referent(), cf));
            }
        }
        self.ui.anim.active_pose_rig = Some(rig_r);
        self.ui.anim.rest_root = Some(root_xf);
        self.ui.anim.saved_part_cframes = saved;
    }

    /// Evaluates the active clip at `self.ui.anim.time` and poses the rig live in the 3D viewport.
    fn apply_live_pose(&mut self, rig_r: Ref, clip_r: Ref) {
        self.ensure_pose_snapshot(rig_r);
        let kfs = read_keyframes(&self.game.doc, clip_r);
        let evaluated = evaluate_clip(&kfs, self.ui.anim.time);
        let rig = Rig::read(&self.game.doc, rig_r);

        let mut transforms: HashMap<Ref, Xform> = HashMap::new();
        for j in &rig.joints {
            let p1_name = self
                .game
                .doc
                .dom
                .get_by_ref(j.part1)
                .map(|i| i.name.as_str())
                .unwrap_or("");
            let xf = evaluated
                .get(p1_name)
                .or_else(|| evaluated.get(&j.name))
                .copied()
                .unwrap_or(Xform {
                    pos: Vec3::ZERO,
                    rot: Mat3::IDENTITY,
                });
            transforms.insert(j.motor, xf);
            self.game
                .doc
                .set_prop_live(j.motor, ustr("Transform"), Variant::CFrame(xf.to_cframe()));
        }

        if rig.root.is_some() {
            let root_xf = self
                .ui
                .anim
                .rest_root
                .unwrap_or_else(|| part_cframe(&self.game.doc, rig.root));
            let solved = rig.solve(root_xf, &transforms);
            write_pose(&mut self.game.doc, &solved, None);
        }
    }

    /// Stops playback and restores identity Motor6D.Transform + rest CFrames on the rig.
    pub fn stop_anim_preview(&mut self) {
        self.ui.anim.playing = false;
        self.ui.anim.last_tick = None;
        if let Some(rig_r) = self.ui.anim.active_pose_rig.take() {
            if self.game.doc.exists(rig_r) {
                let id_cf = cframe_identity();
                let motors: Vec<Ref> = self
                    .game
                    .doc
                    .dom
                    .descendants_of(rig_r)
                    .filter(|i| matches!(i.class.as_str(), "Motor6D" | "AnimationConstraint"))
                    .map(|i| i.referent())
                    .collect();
                for m in motors {
                    self.game
                        .doc
                        .set_prop_live(m, ustr("Transform"), Variant::CFrame(id_cf));
                }
                for (part_r, cf) in self.ui.anim.saved_part_cframes.drain(..) {
                    if self.game.doc.exists(part_r) {
                        self.game
                            .doc
                            .set_prop_live(part_r, ustr("CFrame"), Variant::CFrame(cf));
                    }
                }
            }
        }
        self.ui.anim.rest_root = None;
        self.ui.anim.saved_part_cframes.clear();
    }

    /// Creates a new KeyframeSequence inside `AnimSaves` (creating the Folder if needed) with t = 0.0.
    fn create_new_clip(&mut self, rig_r: Ref) {
        let anim_saves = self
            .game
            .doc
            .dom
            .get_by_ref(rig_r)
            .and_then(|i| {
                i.children().iter().copied().find(|&c| {
                    self.game
                        .doc
                        .dom
                        .get_by_ref(c)
                        .is_some_and(|ci| ci.name == "AnimSaves")
                })
            })
            .unwrap_or_else(|| {
                self.game
                    .doc
                    .insert(rig_r, InstanceBuilder::new("Folder").with_name("AnimSaves"))
            });

        let existing = find_clips(&self.game.doc, Some(rig_r)).len();
        let clip_name = format!("Animation{}", existing + 1);
        let clip_r = self.game.doc.insert(
            anim_saves,
            InstanceBuilder::new("KeyframeSequence")
                .with_name(clip_name)
                .with_property("Loop", Variant::Bool(true)),
        );
        let kf_r = self.game.doc.insert(
            clip_r,
            InstanceBuilder::new("Keyframe")
                .with_name("Keyframe")
                .with_property("Time", Variant::Float32(0.0)),
        );
        // Add initial Pose for HumanoidRootPart or first joint
        let root_pose = self.game.doc.insert(
            kf_r,
            InstanceBuilder::new("Pose")
                .with_name("HumanoidRootPart")
                .with_property("CFrame", Variant::CFrame(cframe_identity()))
                .with_property("Weight", Variant::Float32(1.0)),
        );
        for j in rig_joints(&self.game.doc, rig_r) {
            self.game.doc.insert(
                root_pose,
                InstanceBuilder::new("Pose")
                    .with_name(j.part1_name)
                    .with_property("CFrame", Variant::CFrame(cframe_identity()))
                    .with_property("Weight", Variant::Float32(1.0)),
            );
        }
        self.ui.anim.clip_ref = Some(clip_r);
        self.ui.anim.time = 0.0;
        self.ui.anim.selected_kf_time = Some(0.0);
        self.ui.anim.looping = true;
    }

    /// Finds an existing Keyframe near `time` or creates a new one (copying current evaluated poses).
    fn ensure_keyframe_at(&mut self, clip_r: Ref, time: f32) -> Ref {
        let kfs = read_keyframes(&self.game.doc, clip_r);
        if let Some(existing) = kfs.iter().find(|k| (k.time - time).abs() < 0.01) {
            return existing.kf_ref;
        }
        let evaluated = evaluate_clip(&kfs, time);
        let kf_r = self.game.doc.insert(
            clip_r,
            InstanceBuilder::new("Keyframe")
                .with_name("Keyframe")
                .with_property("Time", Variant::Float32(time)),
        );
        let root_pose = self.game.doc.insert(
            kf_r,
            InstanceBuilder::new("Pose")
                .with_name("HumanoidRootPart")
                .with_property("CFrame", Variant::CFrame(cframe_identity()))
                .with_property("Weight", Variant::Float32(1.0)),
        );
        if let Some(rig_r) = self.ui.anim.rig_ref {
            for j in rig_joints(&self.game.doc, rig_r) {
                let xf = evaluated.get(&j.part1_name).copied().unwrap_or(Xform {
                    pos: Vec3::ZERO,
                    rot: Mat3::IDENTITY,
                });
                self.game.doc.insert(
                    root_pose,
                    InstanceBuilder::new("Pose")
                        .with_name(j.part1_name)
                        .with_property("CFrame", Variant::CFrame(xf.to_cframe()))
                        .with_property("Weight", Variant::Float32(1.0)),
                );
            }
        }
        kf_r
    }

    /// Writes `xf` into the `Pose` named `joint_name` at `time`, creating Keyframe/Pose if needed.
    fn write_joint_pose(&mut self, clip_r: Ref, time: f32, joint_name: &str, xf: Xform) {
        let kf_r = self.ensure_keyframe_at(clip_r, time);
        let existing_pose = self
            .game
            .doc
            .dom
            .descendants_of(kf_r)
            .find(|i| i.class == "Pose" && i.name == joint_name)
            .map(|i| i.referent());
        let cf = Variant::CFrame(xf.to_cframe());
        if let Some(pr) = existing_pose {
            self.game.doc.set_prop_live(pr, ustr("CFrame"), cf);
            self.game.doc.set_prop_live(pr, ustr("Weight"), Variant::Float32(1.0));
        } else {
            let parent = self
                .game
                .doc
                .dom
                .get_by_ref(kf_r)
                .and_then(|i| {
                    i.children()
                        .iter()
                        .copied()
                        .find(|&c| self.game.doc.class_of(c).as_deref() == Some("Pose"))
                })
                .unwrap_or(kf_r);
            self.game.doc.insert(
                parent,
                InstanceBuilder::new("Pose")
                    .with_name(joint_name)
                    .with_property("CFrame", cf)
                    .with_property("Weight", Variant::Float32(1.0)),
            );
        }
    }

    /// Starts the Roblox Publish dialog for an Animation or Model.
    fn open_publish_dialog(&mut self, asset_type: &'static str, refs: Vec<Ref>, default_name: String) {
        let signed_in = self.current_account_session().is_some();
        self.ui.anim.publish = Some(PublishDialog {
            asset_type,
            target_refs: refs,
            name: default_name,
            description: format!("Published from Rustudio ({asset_type})"),
            status: if signed_in {
                PublishStatus::Idle
            } else {
                PublishStatus::NotSignedIn
            },
            rx: None,
        });
    }

    fn current_account_session(&self) -> Option<rustudio_account::session::Session> {
        if std::env::var_os("RUSTUDIO_NO_SESSION").is_some() {
            return None;
        }
        let shared = self.game.account.shared_client()?;
        let guard = shared.try_lock().ok()?;
        guard.as_ref().map(|c| c.session.downloader())
    }

    fn start_publish_upload(&mut self) {
        let Some(session) = self.current_account_session() else {
            if let Some(pub_dlg) = self.ui.anim.publish.as_mut() {
                pub_dlg.status = PublishStatus::NotSignedIn;
            }
            return;
        };
        let (asset_type, refs, name, desc) = {
            let Some(pub_dlg) = self.ui.anim.publish.as_ref() else {
                return;
            };
            (
                pub_dlg.asset_type,
                pub_dlg.target_refs.clone(),
                pub_dlg.name.trim().to_string(),
                pub_dlg.description.trim().to_string(),
            )
        };
        if name.is_empty() {
            if let Some(pub_dlg) = self.ui.anim.publish.as_mut() {
                pub_dlg.status = PublishStatus::Error("Please enter a name.".into());
            }
            return;
        }
        let top = self.game.doc.top_level(&refs);
        if top.is_empty() {
            if let Some(pub_dlg) = self.ui.anim.publish.as_mut() {
                pub_dlg.status = PublishStatus::Error("No valid instances to serialize.".into());
            }
            return;
        }
        let mut bytes = Vec::new();
        if let Err(e) = rbx_binary::to_writer(&mut bytes, &self.game.doc.dom, &top) {
            if let Some(pub_dlg) = self.ui.anim.publish.as_mut() {
                pub_dlg.status = PublishStatus::Error(format!("Serialization failed: {e}"));
            }
            return;
        }

        let (tx, rx) = mpsc::channel();
        let safe_name: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();
        let filename = format!("{safe_name}.rbxm");

        rt_thread::spawn(move || {
            let api = rustudio_account::cloud::CloudApi::new(
                session,
                rustudio_account::cloud::Endpoints::default(),
                None,
            );
            let res = api
                .upload_asset(
                    asset_type,
                    &name,
                    &desc,
                    None,
                    &filename,
                    "model/x-rbxm",
                    &bytes,
                )
                .map_err(|e| e.0);
            let _ = tx.send(res);
        });

        if let Some(pub_dlg) = self.ui.anim.publish.as_mut() {
            pub_dlg.status = PublishStatus::Uploading;
            pub_dlg.rx = Some(rx);
        }
    }

    // -----------------------------------------------------------------------
    // Main Per-Frame Window Entry Point (called from insert_popup_ui)
    // -----------------------------------------------------------------------

    pub fn anim_and_avatar_windows_ui(&mut self, ctx: &egui::Context) {
        self.builtin_plugins_download_popup_ui(ctx);
        // 1. Poll publish upload background thread
        if let Some(pub_dlg) = self.ui.anim.publish.as_mut() {
            if let Some(rx) = pub_dlg.rx.as_ref() {
                if let Ok(res) = rx.try_recv() {
                    pub_dlg.rx = None;
                    match res {
                        Ok(id) => pub_dlg.status = PublishStatus::Success(id),
                        Err(e) => pub_dlg.status = PublishStatus::Error(e),
                    }
                } else {
                    ctx.request_repaint_after(std::time::Duration::from_millis(80));
                }
            }
        }

        // 2. Advance live animation playback
        if self.ui.anim.playing {
            let now = Instant::now();
            let dt = self
                .ui
                .anim
                .last_tick
                .map(|t| now.duration_since(t).as_secs_f32().min(0.1))
                .unwrap_or(0.0);
            self.ui.anim.last_tick = Some(now);

            if let (Some(rig_r), Some(clip_r)) = (self.ui.anim.rig_ref, self.ui.anim.clip_ref) {
                if self.game.doc.exists(rig_r) && self.game.doc.exists(clip_r) {
                    let kfs = read_keyframes(&self.game.doc, clip_r);
                    let dur = clip_duration(&kfs);
                    self.ui.anim.time += dt * self.ui.anim.speed;
                    if self.ui.anim.time > dur {
                        if self.ui.anim.looping {
                            self.ui.anim.time = if dur > 0.0 { self.ui.anim.time % dur } else { 0.0 };
                        } else {
                            self.ui.anim.time = dur;
                            self.ui.anim.playing = false;
                            self.ui.anim.last_tick = None;
                        }
                    }
                    self.apply_live_pose(rig_r, clip_r);
                    ctx.request_repaint();
                } else {
                    self.stop_anim_preview();
                }
            } else {
                self.ui.anim.playing = false;
            }
        }

        // 3. Animation Editor is now hosted as a dockable Tab::AnimEditor (Bottom by default,
        // movable to Left/Right/Bottom or popped out as a floating window).
        if false && self.ui.anim.anim_open {
            let mut open = true;
            let focus = self.ui.anim.focus_next == Some(AvatarToolPanel::AnimEditor);
            if focus {
                self.ui.anim.focus_next = None;
            }
            let screen = ctx.content_rect();
            let w = (screen.width() * 0.86).clamp(320.0, 760.0);
            let h = (screen.height() * 0.44).clamp(240.0, 380.0);
            let resp = egui::Window::new("Animation Editor")
                .id(egui::Id::new("rustudio_native_anim_editor"))
                .open(&mut open)
                .default_size(Vec2::new(w, h))
                .default_pos(Pos2::new(
                    (screen.center().x - w * 0.5).max(screen.min.x + 8.0),
                    (screen.max.y - h - 28.0).max(screen.min.y + 40.0),
                ))
                .resizable(true)
                .collapsible(true)
                .show(ctx, |ui| self.anim_editor_body_ui(ui));
            if focus {
                if let Some(r) = resp {
                    ctx.move_to_top(r.response.layer_id);
                }
            }
            if !open {
                self.stop_anim_preview();
                self.ui.anim.anim_open = false;
            }
        }

        // 4. Render Avatar Setup & Rig Inspector Window
        if self.ui.anim.setup_open {
            let mut open = true;
            let focus = self.ui.anim.focus_next == Some(AvatarToolPanel::AvatarSetup);
            if focus {
                self.ui.anim.focus_next = None;
            }
            let resp = egui::Window::new("Avatar Setup & Rig Inspector")
                .id(egui::Id::new("rustudio_avatar_setup"))
                .open(&mut open)
                .default_size(Vec2::new(360.0, 420.0))
                .resizable(true)
                .collapsible(true)
                .show(ctx, |ui| self.avatar_setup_panel_ui(ui));
            if focus {
                if let Some(r) = resp {
                    ctx.move_to_top(r.response.layer_id);
                }
            }
            if !open {
                self.ui.anim.setup_open = false;
            }
        }

        // 5. Render Accessory Fitter Window
        if self.ui.anim.accessory_open {
            let mut open = true;
            let focus = self.ui.anim.focus_next == Some(AvatarToolPanel::AccessoryFitter);
            if focus {
                self.ui.anim.focus_next = None;
            }
            let resp = egui::Window::new("Accessory Fitter")
                .id(egui::Id::new("rustudio_accessory_fitter"))
                .open(&mut open)
                .default_size(Vec2::new(360.0, 380.0))
                .resizable(true)
                .collapsible(true)
                .show(ctx, |ui| self.accessory_fitter_panel_ui(ui));
            if focus {
                if let Some(r) = resp {
                    ctx.move_to_top(r.response.layer_id);
                }
            }
            if !open {
                self.ui.anim.accessory_open = false;
            }
        }

        // 6. Render Layered Clothing / Wrap Inspector Window
        if self.ui.anim.adaptive_open {
            let mut open = true;
            let focus = self.ui.anim.focus_next == Some(AvatarToolPanel::AdaptiveWrap);
            if focus {
                self.ui.anim.focus_next = None;
            }
            let resp = egui::Window::new("Layered Clothing / Wrap Inspector")
                .id(egui::Id::new("rustudio_adaptive_wrap"))
                .open(&mut open)
                .default_size(Vec2::new(380.0, 360.0))
                .resizable(true)
                .collapsible(true)
                .show(ctx, |ui| self.adaptive_wrap_panel_ui(ui));
            if focus {
                if let Some(r) = resp {
                    ctx.move_to_top(r.response.layer_id);
                }
            }
            if !open {
                self.ui.anim.adaptive_open = false;
            }
        }

        // 7. Render UGC Validation Window
        if self.ui.anim.ugc_open {
            let mut open = true;
            let focus = self.ui.anim.focus_next == Some(AvatarToolPanel::UgcValidation);
            if focus {
                self.ui.anim.focus_next = None;
            }
            let resp = egui::Window::new("UGC Validation")
                .id(egui::Id::new("rustudio_ugc_validation"))
                .open(&mut open)
                .default_size(Vec2::new(380.0, 380.0))
                .resizable(true)
                .collapsible(true)
                .show(ctx, |ui| self.ugc_validation_panel_ui(ui));
            if focus {
                if let Some(r) = resp {
                    ctx.move_to_top(r.response.layer_id);
                }
            }
            if !open {
                self.ui.anim.ugc_open = false;
            }
        }

        // 8. Render Publish to Roblox Modal
        self.publish_modal_ui(ctx);
    }

    // -----------------------------------------------------------------------
    // 1. Native Animation Editor UI (Clip Editor & Graph Editor)
    // -----------------------------------------------------------------------

    fn anim_editor_body_ui(&mut self, ui: &mut egui::Ui) {
        self.sync_anim_selection();

        let rigs = find_all_rigs(&self.game.doc);
        let rig_r = self.ui.anim.rig_ref;
        let clips = find_clips(&self.game.doc, rig_r);
        let clip_r = self.ui.anim.clip_ref;

        // Top toolbar: Rig selector, Clip selector, + New Clip, Mode tabs, Publish
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Rig:").strong());
            let rig_label = rig_r
                .and_then(|r| self.game.doc.dom.get_by_ref(r))
                .map(|i| i.name.clone())
                .unwrap_or_else(|| "(No Rig)".into());
            egui::ComboBox::from_id_salt("anim_rig_combo")
                .selected_text(rig_label)
                .width(120.0)
                .show_ui(ui, |ui| {
                    for (r, name) in &rigs {
                        if ui.selectable_label(self.ui.anim.rig_ref == Some(*r), name).clicked() {
                            self.stop_anim_preview();
                            self.ui.anim.rig_ref = Some(*r);
                            self.ui.anim.clip_ref = None;
                            self.ui.anim.selected_joint = None;
                        }
                    }
                });

            ui.separator();
            ui.label(RichText::new("Clip:").strong());
            let clip_label = clip_r
                .and_then(|c| self.game.doc.dom.get_by_ref(c))
                .map(|i| i.name.clone())
                .unwrap_or_else(|| "(None)".into());
            egui::ComboBox::from_id_salt("anim_clip_combo")
                .selected_text(clip_label)
                .width(140.0)
                .show_ui(ui, |ui| {
                    for (c, name) in &clips {
                        if ui.selectable_label(self.ui.anim.clip_ref == Some(*c), name).clicked() {
                            self.ui.anim.clip_ref = Some(*c);
                            self.ui.anim.time = 0.0;
                            if let Some(Variant::Bool(l)) = self.game.doc.prop(*c, "Loop") {
                                self.ui.anim.looping = l;
                            }
                            if let Some(rr) = rig_r {
                                self.apply_live_pose(rr, *c);
                            }
                        }
                    }
                });

            if let Some(rr) = rig_r {
                if ui.button("+ New Clip").clicked() {
                    self.create_new_clip(rr);
                }
            }

            ui.separator();
            if ui
                .selectable_label(self.ui.anim.mode == AnimViewMode::Clip, "Clip Editor")
                .clicked()
            {
                self.ui.anim.mode = AnimViewMode::Clip;
            }
            if ui
                .selectable_label(self.ui.anim.mode == AnimViewMode::Graph, "Graph Editor")
                .clicked()
            {
                self.ui.anim.mode = AnimViewMode::Graph;
            }

            ui.separator();
            let can_publish = clip_r.is_some();
            if ui
                .add_enabled(can_publish, egui::Button::new("Publish Animation to Roblox"))
                .clicked()
            {
                if let Some(cr) = clip_r {
                    let cname = self
                        .game
                        .doc
                        .dom
                        .get_by_ref(cr)
                        .map(|i| i.name.clone())
                        .unwrap_or_else(|| "Animation".into());
                    self.open_publish_dialog("Animation", vec![cr], cname);
                }
            }
        });

        ui.separator();

        let Some(rig_r) = self.ui.anim.rig_ref else {
            ui.label(
                RichText::new("No Rig found in Workspace. Select or insert an R6 / R15 character Model.")
                    .color(theme::TEXT_DIM),
            );
            return;
        };

        let Some(clip_r) = self.ui.anim.clip_ref else {
            ui.horizontal(|ui| {
                ui.label(RichText::new("No KeyframeSequence selected.").color(theme::TEXT_DIM));
                if ui.button("+ New Clip").clicked() {
                    self.create_new_clip(rig_r);
                }
            });
            return;
        };

        let kfs = read_keyframes(&self.game.doc, clip_r);
        let duration = clip_duration(&kfs);

        // Transport bar: separate Play & Pause (prevents stray/double tap from immediately
        // toggling Play back off after 1 frame), Stop, Loop, Speed, Scrub slider, + Add / Delete Keyframe.
        ui.horizontal_wrapped(|ui| {
            let btn_h = if cfg!(target_os = "android") { 28.0 } else { 22.0 };
            if ui
                .add_enabled(!self.ui.anim.playing, egui::Button::new("Play").min_size(Vec2::new(52.0, btn_h)))
                .clicked()
            {
                self.ui.anim.playing = true;
                self.ui.anim.last_tick = Some(Instant::now());
                if self.ui.anim.time >= duration {
                    self.ui.anim.time = 0.0;
                }
                self.apply_live_pose(rig_r, clip_r);
                ui.ctx().request_repaint();
            }
            if ui
                .add_enabled(self.ui.anim.playing, egui::Button::new("Pause").min_size(Vec2::new(54.0, btn_h)))
                .clicked()
            {
                self.ui.anim.playing = false;
                self.ui.anim.last_tick = None;
            }
            if ui
                .add(egui::Button::new("Stop").min_size(Vec2::new(48.0, btn_h)))
                .clicked()
            {
                self.ui.anim.time = 0.0;
                self.stop_anim_preview();
            }
            if ui.checkbox(&mut self.ui.anim.looping, "Loop").changed() {
                self.game
                    .doc
                    .set_prop_live(clip_r, ustr("Loop"), Variant::Bool(self.ui.anim.looping));
            }
            egui::ComboBox::from_id_salt("anim_speed_combo")
                .selected_text(format!("{}x", self.ui.anim.speed))
                .width(58.0)
                .show_ui(ui, |ui| {
                    for s in SPEEDS {
                        if ui
                            .selectable_label((self.ui.anim.speed - s).abs() < 0.01, format!("{s}x"))
                            .clicked()
                        {
                            self.ui.anim.speed = s;
                        }
                    }
                });

            let slider = ui.add(
                egui::Slider::new(&mut self.ui.anim.time, 0.0..=duration)
                    .suffix("s")
                    .fixed_decimals(2),
            );
            if slider.drag_started() || slider.dragged() || (slider.changed() && slider.has_focus()) {
                self.ui.anim.playing = false;
                self.ui.anim.last_tick = None;
                self.apply_live_pose(rig_r, clip_r);
            }

            if ui.button("+ Add Keyframe").clicked() {
                let t = (self.ui.anim.time * 100.0).round() / 100.0;
                self.ensure_keyframe_at(clip_r, t);
                self.ui.anim.selected_kf_time = Some(t);
            }

            let near_kf = kfs
                .iter()
                .find(|k| (k.time - self.ui.anim.time).abs() < 0.025)
                .map(|k| k.kf_ref);
            if ui
                .add_enabled(near_kf.is_some() && kfs.len() > 1, egui::Button::new("Delete Keyframe"))
                .clicked()
            {
                if let Some(kr) = near_kf {
                    self.game.doc.remove(&[kr]);
                    self.apply_live_pose(rig_r, clip_r);
                }
            }
        });

        ui.separator();

        // Main body: left joint list, right Clip timeline + pose sliders OR Graph Editor
        let joints = rig_joints(&self.game.doc, rig_r);
        let avail_h = ui.available_height().max(140.0);

        ui.columns(2, |cols| {
            // Left column (narrower feel via scroll area): Joint list
            cols[0].set_max_width(210.0);
            cols[0].label(RichText::new("Rig Joints").strong());
            egui::ScrollArea::vertical()
                .id_salt("anim_joints_scroll")
                .max_height(avail_h - 22.0)
                .show(&mut cols[0], |ui| {
                    if joints.is_empty() {
                        ui.label(RichText::new("No Motor6D joints found.").color(theme::TEXT_DIM));
                    }
                    for j in &joints {
                        let sel = self.ui.anim.selected_joint.as_deref() == Some(j.part1_name.as_str());
                        let label = if j.motor_name != j.part1_name {
                            format!("{} ({})", j.part1_name, j.motor_name)
                        } else {
                            j.part1_name.clone()
                        };
                        if ui.selectable_label(sel, label).clicked() {
                            self.ui.anim.selected_joint = Some(j.part1_name.clone());
                        }
                    }
                });

            // Right column: Clip Editor or Graph Editor
            match self.ui.anim.mode {
                AnimViewMode::Clip => {
                    self.clip_editor_right_ui(&mut cols[1], rig_r, clip_r, &kfs, duration);
                }
                AnimViewMode::Graph => {
                    self.graph_editor_right_ui(&mut cols[1], rig_r, clip_r, &kfs, duration);
                }
            }
        });
    }

    fn clip_editor_right_ui(
        &mut self,
        ui: &mut egui::Ui,
        rig_r: Ref,
        clip_r: Ref,
        kfs: &[KeyframeData],
        duration: f32,
    ) {
        // 1. Visual Timeline with diamond keyframe markers
        let timeline_w = ui.available_width().max(200.0);
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(timeline_w, 44.0), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, theme::PANEL_DARK);
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, theme::DIVIDER), egui::StrokeKind::Inside);

        let pad_x = 12.0;
        let track_w = (rect.width() - pad_x * 2.0).max(1.0);
        let time_to_x = |t: f32| rect.min.x + pad_x + (t / duration).clamp(0.0, 1.0) * track_w;

        // Tick marks
        let steps = 5;
        for i in 0..=steps {
            let frac = i as f32 / steps as f32;
            let x = rect.min.x + pad_x + frac * track_w;
            painter.line_segment(
                [Pos2::new(x, rect.min.y + 2.0), Pos2::new(x, rect.min.y + 12.0)],
                Stroke::new(1.0, theme::DIVIDER),
            );
            painter.text(
                Pos2::new(x, rect.min.y + 14.0),
                egui::Align2::CENTER_TOP,
                format!("{:.2}s", frac * duration),
                FontId::proportional(9.5),
                theme::TEXT_DIM,
            );
        }

        // Scrub by clicking/dragging timeline
        if resp.clicked() || resp.dragged() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let frac = ((pos.x - (rect.min.x + pad_x)) / track_w).clamp(0.0, 1.0);
                self.ui.anim.time = frac * duration;
                self.ui.anim.playing = false;
                self.ui.anim.last_tick = None;
                self.apply_live_pose(rig_r, clip_r);
            }
        }

        // Diamond markers for each keyframe
        let cy = rect.max.y - 12.0;
        for kf in kfs {
            let x = time_to_x(kf.time);
            let is_near = (kf.time - self.ui.anim.time).abs() < 0.025;
            let fill = if is_near { theme::SELECT } else { theme::WARN };
            let r = 5.5;
            let pts = vec![
                Pos2::new(x, cy - r),
                Pos2::new(x + r, cy),
                Pos2::new(x, cy + r),
                Pos2::new(x - r, cy),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::new(1.0, Color32::BLACK)));
        }

        // Playhead line
        let px = time_to_x(self.ui.anim.time);
        painter.line_segment(
            [Pos2::new(px, rect.min.y), Pos2::new(px, rect.max.y)],
            Stroke::new(2.0, theme::ACCENT),
        );

        ui.add_space(6.0);

        // 2. Touch-friendly Rotation (deg) & Position sliders for selected joint
        let Some(joint_name) = self.ui.anim.selected_joint.clone() else {
            ui.label(RichText::new("Select a joint on the left to edit its pose.").color(theme::TEXT_DIM));
            return;
        };

        let evaluated = evaluate_clip(kfs, self.ui.anim.time);
        let cur_xf = evaluated.get(&joint_name).copied().unwrap_or(Xform {
            pos: Vec3::ZERO,
            rot: Mat3::IDENTITY,
        });
        let mut rot = cur_xf.orientation_deg();
        let mut pos = cur_xf.pos;
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Pose: {joint_name}")).strong());
            if ui.small_button("Reset Pose").clicked() {
                rot = Vec3::ZERO;
                pos = Vec3::ZERO;
                changed = true;
            }
        });

        egui::Grid::new("anim_pose_sliders")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label(RichText::new("Rot X").color(Color32::from_rgb(240, 100, 100)));
                changed |= ui
                    .add(egui::Slider::new(&mut rot.x, -180.0..=180.0).suffix("°").fixed_decimals(1))
                    .changed();
                ui.end_row();

                ui.label(RichText::new("Rot Y").color(Color32::from_rgb(100, 220, 120)));
                changed |= ui
                    .add(egui::Slider::new(&mut rot.y, -180.0..=180.0).suffix("°").fixed_decimals(1))
                    .changed();
                ui.end_row();

                ui.label(RichText::new("Rot Z").color(Color32::from_rgb(100, 160, 250)));
                changed |= ui
                    .add(egui::Slider::new(&mut rot.z, -180.0..=180.0).suffix("°").fixed_decimals(1))
                    .changed();
                ui.end_row();

                ui.label("Pos X");
                changed |= ui
                    .add(egui::Slider::new(&mut pos.x, -5.0..=5.0).fixed_decimals(2))
                    .changed();
                ui.end_row();

                ui.label("Pos Y");
                changed |= ui
                    .add(egui::Slider::new(&mut pos.y, -5.0..=5.0).fixed_decimals(2))
                    .changed();
                ui.end_row();

                ui.label("Pos Z");
                changed |= ui
                    .add(egui::Slider::new(&mut pos.z, -5.0..=5.0).fixed_decimals(2))
                    .changed();
                ui.end_row();
            });

        if changed {
            let new_xf = Xform::from_orientation_deg(pos, rot);
            let t = (self.ui.anim.time * 100.0).round() / 100.0;
            self.write_joint_pose(clip_r, t, &joint_name, new_xf);
            self.apply_live_pose(rig_r, clip_r);
        }
    }

    fn graph_editor_right_ui(
        &mut self,
        ui: &mut egui::Ui,
        rig_r: Ref,
        clip_r: Ref,
        kfs: &[KeyframeData],
        duration: f32,
    ) {
        let Some(joint_name) = self.ui.anim.selected_joint.clone() else {
            ui.label(RichText::new("Select a joint on the left to view its rotation curves.").color(theme::TEXT_DIM));
            return;
        };

        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Euler Rotation Curves: {joint_name}")).strong());
            ui.label(RichText::new("X").color(Color32::from_rgb(240, 90, 90)));
            ui.label(RichText::new("Y").color(Color32::from_rgb(90, 220, 110)));
            ui.label(RichText::new("Z").color(Color32::from_rgb(90, 155, 250)));
        });

        let avail = Vec2::new(ui.available_width().max(220.0), ui.available_height().max(160.0));
        let (rect, resp) = ui.allocate_exact_size(avail, Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 4.0, theme::PANEL_DARK);
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, theme::DIVIDER), egui::StrokeKind::Inside);

        let pad_l = 36.0;
        let pad_r = 12.0;
        let pad_t = 12.0;
        let pad_b = 20.0;
        let plot = Rect::from_min_max(
            Pos2::new(rect.min.x + pad_l, rect.min.y + pad_t),
            Pos2::new(rect.max.x - pad_r, rect.max.y - pad_b),
        );

        let t_to_x = |t: f32| plot.min.x + (t / duration).clamp(0.0, 1.0) * plot.width().max(1.0);
        let deg_to_y = |deg: f32| {
            let norm = (deg.clamp(-180.0, 180.0) + 180.0) / 360.0;
            plot.max.y - norm * plot.height().max(1.0)
        };

        // Horizontal degree grid lines (-180, -90, 0, +90, +180)
        for deg in [-180.0, -90.0, 0.0, 90.0, 180.0] {
            let y = deg_to_y(deg);
            let col = if deg == 0.0 {
                Color32::from_gray(90)
            } else {
                theme::DIVIDER
            };
            painter.line_segment([Pos2::new(plot.min.x, y), Pos2::new(plot.max.x, y)], Stroke::new(1.0, col));
            painter.text(
                Pos2::new(plot.min.x - 4.0, y),
                egui::Align2::RIGHT_CENTER,
                format!("{deg:.0}°"),
                FontId::proportional(9.5),
                theme::TEXT_DIM,
            );
        }

        // Vertical time grid lines
        for i in 0..=4 {
            let frac = i as f32 / 4.0;
            let x = plot.min.x + frac * plot.width();
            painter.line_segment(
                [Pos2::new(x, plot.min.y), Pos2::new(x, plot.max.y)],
                Stroke::new(1.0, theme::DIVIDER),
            );
            painter.text(
                Pos2::new(x, plot.max.y + 3.0),
                egui::Align2::CENTER_TOP,
                format!("{:.2}s", frac * duration),
                FontId::proportional(9.5),
                theme::TEXT_DIM,
            );
        }

        // Collect keyframe samples for this joint
        let mut pts_x = Vec::new();
        let mut pts_y = Vec::new();
        let mut pts_z = Vec::new();
        let mut kf_times = Vec::new();
        for kf in kfs {
            if let Some(&(_, xf, w)) = kf.poses.get(&joint_name) {
                if w > 0.0 {
                    let deg = xf.orientation_deg();
                    let x = t_to_x(kf.time);
                    pts_x.push(Pos2::new(x, deg_to_y(deg.x)));
                    pts_y.push(Pos2::new(x, deg_to_y(deg.y)));
                    pts_z.push(Pos2::new(x, deg_to_y(deg.z)));
                    kf_times.push(kf.time);
                }
            }
        }

        let col_x = Color32::from_rgb(240, 90, 90);
        let col_y = Color32::from_rgb(90, 220, 110);
        let col_z = Color32::from_rgb(90, 155, 250);

        for (pts, col) in [(&pts_x, col_x), (&pts_y, col_y), (&pts_z, col_z)] {
            if pts.len() >= 2 {
                painter.add(egui::Shape::line(pts.clone(), Stroke::new(2.0, col)));
            }
            for &p in pts {
                painter.circle_filled(p, 4.0, col);
                painter.circle_stroke(p, 4.0, Stroke::new(1.0, Color32::BLACK));
            }
        }

        // Playhead
        let px = t_to_x(self.ui.anim.time);
        painter.line_segment(
            [Pos2::new(px, plot.min.y), Pos2::new(px, plot.max.y)],
            Stroke::new(1.5, theme::ACCENT),
        );

        // Click/drag on graph: jump to clicked keyframe dot or scrub time
        if resp.clicked() || resp.dragged() {
            if let Some(pointer) = resp.interact_pointer_pos() {
                let mut snapped = None;
                if resp.clicked() {
                    for (idx, &kt) in kf_times.iter().enumerate() {
                        if pts_x[idx].distance(pointer) < 10.0
                            || pts_y[idx].distance(pointer) < 10.0
                            || pts_z[idx].distance(pointer) < 10.0
                        {
                            snapped = Some(kt);
                            break;
                        }
                    }
                }
                let new_t = snapped.unwrap_or_else(|| {
                    let frac = ((pointer.x - plot.min.x) / plot.width().max(1.0)).clamp(0.0, 1.0);
                    frac * duration
                });
                self.ui.anim.time = new_t;
                self.ui.anim.playing = false;
                self.ui.anim.last_tick = None;
                self.apply_live_pose(rig_r, clip_r);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 2. Avatar Setup & Rig Inspector
    // -----------------------------------------------------------------------

    fn avatar_setup_panel_ui(&mut self, ui: &mut egui::Ui) {
        self.sync_anim_selection();
        let rigs = find_all_rigs(&self.game.doc);

        ui.horizontal(|ui| {
            ui.label(RichText::new("Active Rig:").strong());
            let label = self
                .ui
                .anim
                .rig_ref
                .and_then(|r| self.game.doc.dom.get_by_ref(r))
                .map(|i| i.name.clone())
                .unwrap_or_else(|| "(None)".into());
            egui::ComboBox::from_id_salt("setup_rig_combo")
                .selected_text(label)
                .show_ui(ui, |ui| {
                    for (r, name) in &rigs {
                        if ui.selectable_label(self.ui.anim.rig_ref == Some(*r), name).clicked() {
                            self.ui.anim.rig_ref = Some(*r);
                        }
                    }
                });
        });

        ui.separator();

        // Publish Selected Model button (works for any selected Model or the active rig)
        let publish_target = self
            .game
            .doc
            .selection
            .first()
            .copied()
            .filter(|r| self.game.doc.exists(*r))
            .or(self.ui.anim.rig_ref);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    publish_target.is_some(),
                    egui::Button::new("Publish Selected Model to Roblox"),
                )
                .clicked()
            {
                if let Some(tr) = publish_target {
                    let name = self
                        .game
                        .doc
                        .dom
                        .get_by_ref(tr)
                        .map(|i| i.name.clone())
                        .unwrap_or_else(|| "Model".into());
                    self.open_publish_dialog("Model", vec![tr], name);
                }
            }
        });

        ui.separator();

        let Some(rig_r) = self.ui.anim.rig_ref else {
            ui.label(RichText::new("No Rig selected or found in Workspace.").color(theme::TEXT_DIM));
            return;
        };

        let joints = rig_joints(&self.game.doc, rig_r);
        let has_upper_torso = self.game.doc.find_named(rig_r, "UpperTorso").is_some();
        let has_torso = self.game.doc.find_named(rig_r, "Torso").is_some();
        let detected_type = if has_upper_torso {
            "R15"
        } else if has_torso {
            "R6"
        } else {
            "Custom"
        };

        ui.label(RichText::new(format!("Detected Rig Type: {detected_type} ({} joints)", joints.len())).strong());

        // Humanoid stats
        if let Some(hum_r) = self.game.doc.find_child(rig_r, "Humanoid") {
            ui.add_space(4.0);
            ui.label(RichText::new("Humanoid Stats").strong());
            let read_f32 = |prop: &str, def: f32| match self.game.doc.prop(hum_r, prop) {
                Some(Variant::Float32(v)) => v,
                Some(Variant::Float64(v)) => v as f32,
                _ => def,
            };
            let mut walk_speed = read_f32("WalkSpeed", 16.0);
            let mut jump_power = read_f32("JumpPower", 50.0);
            let mut hip_height = read_f32("HipHeight", 0.0);
            let mut max_health = read_f32("MaxHealth", 100.0);

            egui::Grid::new("avatar_setup_hum_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("WalkSpeed");
                    if ui.add(egui::DragValue::new(&mut walk_speed).speed(0.5).range(0.0..=200.0)).changed() {
                        self.game.doc.set_prop_live(hum_r, ustr("WalkSpeed"), Variant::Float32(walk_speed));
                    }
                    ui.end_row();

                    ui.label("JumpPower");
                    if ui.add(egui::DragValue::new(&mut jump_power).speed(0.5).range(0.0..=300.0)).changed() {
                        self.game.doc.set_prop_live(hum_r, ustr("JumpPower"), Variant::Float32(jump_power));
                    }
                    ui.end_row();

                    ui.label("HipHeight");
                    if ui.add(egui::DragValue::new(&mut hip_height).speed(0.05).range(-5.0..=20.0)).changed() {
                        self.game.doc.set_prop_live(hum_r, ustr("HipHeight"), Variant::Float32(hip_height));
                    }
                    ui.end_row();

                    ui.label("MaxHealth");
                    if ui.add(egui::DragValue::new(&mut max_health).speed(1.0).range(1.0..=10000.0)).changed() {
                        self.game.doc.set_prop_live(hum_r, ustr("MaxHealth"), Variant::Float32(max_health));
                    }
                    ui.end_row();
                });
        } else {
            ui.label(RichText::new("No Humanoid child found on this rig.").color(theme::WARN));
        }

        ui.add_space(6.0);
        ui.label(RichText::new("Motor6D Joints").strong());
        egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
            for j in &joints {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&j.motor_name).strong());
                    ui.label(
                        RichText::new(format!("{} -> {}", j.part0_name, j.part1_name))
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                });
            }
        });
    }

    // -----------------------------------------------------------------------
    // 3. Accessory Fitter
    // -----------------------------------------------------------------------

    fn accessory_fitter_panel_ui(&mut self, ui: &mut egui::Ui) {
        self.sync_anim_selection();

        // Find selected Accessory or first Accessory in Workspace
        let selected_acc = self
            .game
            .doc
            .selection
            .iter()
            .copied()
            .find_map(|sel| {
                let mut cur = sel;
                while cur.is_some() {
                    if self.game.doc.class_of(cur).as_deref() == Some("Accessory") {
                        return Some(cur);
                    }
                    cur = self.game.doc.dom.get_by_ref(cur).map(|i| i.parent()).unwrap_or(Ref::none());
                }
                None
            })
            .or_else(|| {
                let ws = self.game.doc.workspace()?;
                self.game
                    .doc
                    .dom
                    .descendants_of(ws)
                    .find(|i| i.class == "Accessory")
                    .map(|i| i.referent())
            });

        let Some(acc_r) = selected_acc else {
            ui.label(
                RichText::new("No Accessory selected or found in Workspace. Select an Accessory to inspect or fit it.")
                    .color(theme::TEXT_DIM),
            );
            return;
        };

        let acc_name = self
            .game
            .doc
            .dom
            .get_by_ref(acc_r)
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "Accessory".into());
        let handle_r = self.game.doc.find_named(acc_r, "Handle");

        ui.label(RichText::new(format!("Accessory: {acc_name}")).strong());
        ui.label(
            RichText::new(if handle_r.is_some() {
                "Handle: Found"
            } else {
                "Handle: Missing (required for fitting)"
            })
            .color(if handle_r.is_some() { theme::TEXT_DIM } else { theme::WARN }),
        );

        ui.separator();

        // Attach to character
        ui.horizontal(|ui| {
            ui.label("Attach Point:");
            egui::ComboBox::from_id_salt("acc_attach_combo")
                .selected_text(ACCESSORY_ATTACH_TARGETS[self.ui.anim.accessory_target_part].0)
                .show_ui(ui, |ui| {
                    for (idx, (label, _)) in ACCESSORY_ATTACH_TARGETS.iter().enumerate() {
                        ui.selectable_value(&mut self.ui.anim.accessory_target_part, idx, *label);
                    }
                });
        });

        let can_attach = self.ui.anim.rig_ref.is_some() && handle_r.is_some();
        if ui
            .add_enabled(can_attach, egui::Button::new("Attach to Active Character"))
            .clicked()
        {
            if let (Some(rig_r), Some(hr)) = (self.ui.anim.rig_ref, handle_r) {
                let attach_name = ACCESSORY_ATTACH_TARGETS[self.ui.anim.accessory_target_part].1;
                // Ensure Attachment exists inside Handle
                let existing = self.game.doc.find_named(hr, attach_name);
                if existing.is_none() {
                    self.game.doc.insert(
                        hr,
                        InstanceBuilder::new("Attachment").with_name(attach_name),
                    );
                }
                // Reparent Accessory into the character rig if not already a child
                let cur_parent = self.game.doc.dom.get_by_ref(acc_r).map(|i| i.parent());
                if cur_parent != Some(rig_r) {
                    self.game.doc.reparent(&[acc_r], rig_r);
                }
                // Position Handle on the matching character part
                let target_part = if attach_name.contains("Hat")
                    || attach_name.contains("Hair")
                    || attach_name.contains("Face")
                {
                    self.game.doc.find_named(rig_r, "Head")
                } else {
                    self.game
                        .doc
                        .find_named(rig_r, "UpperTorso")
                        .or_else(|| self.game.doc.find_named(rig_r, "Torso"))
                };
                if let Some(tp) = target_part {
                    let tp_xf = part_cframe(&self.game.doc, tp);
                    let ap_xf = match self.game.doc.prop(acc_r, "AttachmentPoint") {
                        Some(Variant::CFrame(cf)) => Xform::from_cframe(&cf),
                        _ => Xform {
                            pos: Vec3::ZERO,
                            rot: Mat3::IDENTITY,
                        },
                    };
                    let placed = Xform {
                        pos: tp_xf.apply(-ap_xf.pos),
                        rot: tp_xf.rot * ap_xf.rot.transpose(),
                    };
                    self.game
                        .doc
                        .set_prop_live(hr, ustr("CFrame"), Variant::CFrame(placed.to_cframe()));
                }
                self.info(format!("Attached '{acc_name}' to rig ({attach_name})."));
            }
        }

        ui.separator();
        ui.label(RichText::new("AttachmentPoint (AttachmentPos / AttachmentForward)").strong());

        let mut ap = match self.game.doc.prop(acc_r, "AttachmentPoint") {
            Some(Variant::CFrame(cf)) => Xform::from_cframe(&cf),
            _ => Xform {
                pos: Vec3::ZERO,
                rot: Mat3::IDENTITY,
            },
        };
        let mut pos = ap.pos;
        let mut fwd = -ap.rot.z_axis;
        let mut changed_pos = false;
        let mut changed_fwd = false;

        egui::Grid::new("acc_fit_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("AttachmentPos X");
                changed_pos |= ui.add(egui::Slider::new(&mut pos.x, -4.0..=4.0).fixed_decimals(2)).changed();
                ui.end_row();
                ui.label("AttachmentPos Y");
                changed_pos |= ui.add(egui::Slider::new(&mut pos.y, -4.0..=4.0).fixed_decimals(2)).changed();
                ui.end_row();
                ui.label("AttachmentPos Z");
                changed_pos |= ui.add(egui::Slider::new(&mut pos.z, -4.0..=4.0).fixed_decimals(2)).changed();
                ui.end_row();

                ui.label("Forward X");
                changed_fwd |= ui.add(egui::Slider::new(&mut fwd.x, -1.0..=1.0).fixed_decimals(2)).changed();
                ui.end_row();
                ui.label("Forward Y");
                changed_fwd |= ui.add(egui::Slider::new(&mut fwd.y, -1.0..=1.0).fixed_decimals(2)).changed();
                ui.end_row();
                ui.label("Forward Z");
                changed_fwd |= ui.add(egui::Slider::new(&mut fwd.z, -1.0..=1.0).fixed_decimals(2)).changed();
                ui.end_row();
            });

        if changed_pos || changed_fwd {
            ap.pos = pos;
            if changed_fwd && fwd.length_squared() > 1e-4 {
                let back = -fwd.normalize();
                let up_ref = if back.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
                let right = up_ref.cross(back).normalize_or(Vec3::X);
                let up = back.cross(right).normalize_or(Vec3::Y);
                ap.rot = Mat3::from_cols(right, up, back);
            }
            self.game
                .doc
                .set_prop_live(acc_r, ustr("AttachmentPoint"), Variant::CFrame(ap.to_cframe()));
        }
    }

    // -----------------------------------------------------------------------
    // 4. Layered Clothing / Wrap Inspector
    // -----------------------------------------------------------------------

    fn adaptive_wrap_panel_ui(&mut self, ui: &mut egui::Ui) {
        self.sync_anim_selection();
        let root_r = self.ui.anim.rig_ref.or_else(|| self.game.doc.workspace());

        let Some(root_r) = root_r else {
            ui.label(RichText::new("No Rig or Workspace available.").color(theme::TEXT_DIM));
            return;
        };

        let root_name = self
            .game
            .doc
            .dom
            .get_by_ref(root_r)
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "Workspace".into());
        ui.label(RichText::new(format!("Inspecting WrapLayer & WrapTarget in: {root_name}")).strong());
        ui.separator();

        let wraps: Vec<(Ref, String, String, String)> = self
            .game
            .doc
            .dom
            .descendants_of(root_r)
            .filter(|i| matches!(i.class.as_str(), "WrapLayer" | "WrapTarget"))
            .map(|i| {
                let parent_name = self
                    .game
                    .doc
                    .dom
                    .get_by_ref(i.parent())
                    .map(|p| p.name.clone())
                    .unwrap_or_default();
                (i.referent(), i.name.clone(), i.class.to_string(), parent_name)
            })
            .collect();

        if wraps.is_empty() {
            ui.label(
                RichText::new("No WrapLayer or WrapTarget instances found on the selected rig.")
                    .color(theme::TEXT_DIM),
            );
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            for (wr, name, class, parent_name) in wraps {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(&name).strong());
                        ui.label(RichText::new(format!("[{class}]")).color(theme::SELECT).small());
                        ui.label(RichText::new(format!("on {parent_name}")).color(theme::TEXT_DIM).small());
                    });
                    if class == "WrapLayer" {
                        let mut order = match self.game.doc.prop(wr, "Order") {
                            Some(Variant::Int32(v)) => v,
                            _ => 1,
                        };
                        let mut puff = match self.game.doc.prop(wr, "Puffiness") {
                            Some(Variant::Float32(v)) => v,
                            _ => 1.0,
                        };
                        ui.horizontal(|ui| {
                            ui.label("Order:");
                            if ui.add(egui::DragValue::new(&mut order).range(0..=100)).changed() {
                                self.game.doc.set_prop_live(wr, ustr("Order"), Variant::Int32(order));
                            }
                            ui.label("Puffiness:");
                            if ui.add(egui::Slider::new(&mut puff, 0.0..=2.0).fixed_decimals(2)).changed() {
                                self.game.doc.set_prop_live(wr, ustr("Puffiness"), Variant::Float32(puff));
                            }
                        });
                    } else if class == "WrapTarget" {
                        let mut stiffness = match self.game.doc.prop(wr, "Stiffness") {
                            Some(Variant::Float32(v)) => v,
                            _ => 0.0,
                        };
                        ui.horizontal(|ui| {
                            ui.label("Stiffness:");
                            if ui.add(egui::Slider::new(&mut stiffness, 0.0..=1.0).fixed_decimals(2)).changed() {
                                self.game
                                    .doc
                                    .set_prop_live(wr, ustr("Stiffness"), Variant::Float32(stiffness));
                            }
                        });
                    }
                });
            }
        });
    }

    // -----------------------------------------------------------------------
    // 5. UGC Validation Panel
    // -----------------------------------------------------------------------

    fn run_ugc_validation(&mut self) {
        let target = self
            .game
            .doc
            .selection
            .first()
            .copied()
            .filter(|r| self.game.doc.exists(*r))
            .or(self.ui.anim.rig_ref);

        self.ui.anim.ugc_last_target = target;
        self.ui.anim.ugc_results.clear();

        let Some(tr) = target else {
            self.ui.anim.ugc_results.push(UgcCheckResult {
                label: "Selection".into(),
                detail: "No Model or Accessory selected.".into(),
                level: CheckLevel::Fail,
            });
            return;
        };

        let Some(inst) = self.game.doc.dom.get_by_ref(tr) else {
            return;
        };
        let class = inst.class.to_string();
        let is_acc = class == "Accessory";
        let is_model = class == "Model";

        // 1. Class check
        self.ui.anim.ugc_results.push(UgcCheckResult {
            label: "Root Class".into(),
            detail: format!("Selected '{}' ({class})", inst.name),
            level: if is_acc || is_model {
                CheckLevel::Pass
            } else {
                CheckLevel::Warn
            },
        });

        // Gather descendants
        let mut part_count = 0usize;
        let mut mesh_count = 0usize;
        let mut attachment_count = 0usize;
        let mut script_count = 0usize;
        let mut min_pt = Vec3::splat(f32::INFINITY);
        let mut max_pt = Vec3::splat(f32::NEG_INFINITY);

        for d in self.game.doc.dom.descendants_of(tr) {
            let c = d.class.as_str();
            if rustudio_core::doc::is_a(c, "BasePart") {
                part_count += 1;
                let xf = part_cframe(&self.game.doc, d.referent());
                let sz = match self.game.doc.prop(d.referent(), "Size") {
                    Some(Variant::Vector3(v)) => v3(v),
                    _ => Vec3::ONE,
                };
                let half = sz * 0.5;
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        for sz_s in [-1.0, 1.0] {
                            let corner = xf.apply(Vec3::new(sx * half.x, sy * half.y, sz_s * half.z));
                            min_pt = min_pt.min(corner);
                            max_pt = max_pt.max(corner);
                        }
                    }
                }
            }
            if matches!(c, "MeshPart" | "SpecialMesh" | "FileMesh") {
                mesh_count += 1;
            }
            if c == "Attachment" {
                attachment_count += 1;
            }
            if matches!(c, "Script" | "LocalScript" | "ModuleScript") {
                script_count += 1;
            }
        }

        // 2. Handle & Attachment check
        if is_acc {
            let has_handle = self.game.doc.find_named(tr, "Handle").is_some();
            self.ui.anim.ugc_results.push(UgcCheckResult {
                label: "Handle Part".into(),
                detail: if has_handle {
                    "Handle found inside Accessory.".into()
                } else {
                    "Accessory must contain a 'Handle' BasePart.".into()
                },
                level: if has_handle { CheckLevel::Pass } else { CheckLevel::Fail },
            });
        }
        self.ui.anim.ugc_results.push(UgcCheckResult {
            label: "Attachments".into(),
            detail: format!("{attachment_count} Attachment(s) found."),
            level: if attachment_count > 0 {
                CheckLevel::Pass
            } else if is_acc {
                CheckLevel::Fail
            } else {
                CheckLevel::Warn
            },
        });

        // 3. Part count check
        let part_level = if part_count == 0 {
            CheckLevel::Fail
        } else if is_acc && part_count > 1 {
            CheckLevel::Warn
        } else if part_count > 50 {
            CheckLevel::Warn
        } else {
            CheckLevel::Pass
        };
        self.ui.anim.ugc_results.push(UgcCheckResult {
            label: "Part Count".into(),
            detail: format!("{part_count} BasePart(s)"),
            level: part_level,
        });

        // 4. Bounding box size check
        if part_count > 0 && min_pt.is_finite() && max_pt.is_finite() {
            let ext = max_pt - min_pt;
            let limit = if is_acc { 4.0 } else { 12.0 };
            let ok = ext.x <= limit && ext.y <= limit && ext.z <= limit;
            self.ui.anim.ugc_results.push(UgcCheckResult {
                label: "Bounding Box".into(),
                detail: format!("{:.2} x {:.2} x {:.2} studs (limit {limit:.0})", ext.x, ext.y, ext.z),
                level: if ok { CheckLevel::Pass } else { CheckLevel::Warn },
            });
        }

        // 5. Mesh presence check
        self.ui.anim.ugc_results.push(UgcCheckResult {
            label: "Mesh Geometry".into(),
            detail: format!("{mesh_count} MeshPart / SpecialMesh instance(s)"),
            level: if mesh_count > 0 {
                CheckLevel::Pass
            } else {
                CheckLevel::Warn
            },
        });

        // 6. Disallowed scripts check
        self.ui.anim.ugc_results.push(UgcCheckResult {
            label: "No Scripts".into(),
            detail: if script_count == 0 {
                "No scripts found (UGC compliant).".into()
            } else {
                format!("{script_count} script(s) found — remove before UGC upload.")
            },
            level: if script_count == 0 {
                CheckLevel::Pass
            } else {
                CheckLevel::Fail
            },
        });
    }

    fn ugc_validation_panel_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Run UGC Validation").clicked() {
                self.run_ugc_validation();
            }
            if let Some(tr) = self.ui.anim.ugc_last_target {
                if let Some(inst) = self.game.doc.dom.get_by_ref(tr) {
                    ui.label(RichText::new(format!("Target: {}", inst.name)).color(theme::TEXT_DIM));
                }
            }
        });
        ui.separator();

        if self.ui.anim.ugc_results.is_empty() {
            self.run_ugc_validation();
        }

        let passes = self.ui.anim.ugc_results.iter().filter(|r| r.level == CheckLevel::Pass).count();
        let warns = self.ui.anim.ugc_results.iter().filter(|r| r.level == CheckLevel::Warn).count();
        let fails = self.ui.anim.ugc_results.iter().filter(|r| r.level == CheckLevel::Fail).count();
        ui.label(
            RichText::new(format!("Summary: {passes} Pass | {warns} Warn | {fails} Fail")).strong(),
        );
        ui.add_space(4.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            for res in &self.ui.anim.ugc_results {
                let (badge, col) = match res.level {
                    CheckLevel::Pass => ("PASS", Color32::from_rgb(90, 210, 120)),
                    CheckLevel::Warn => ("WARN", theme::WARN),
                    CheckLevel::Fail => ("FAIL", theme::ERROR),
                };
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(format!("[{badge}]")).color(col).strong());
                    ui.label(RichText::new(&res.label).strong());
                    ui.label(RichText::new(&res.detail).color(theme::TEXT_DIM));
                });
            }
        });
    }

    // -----------------------------------------------------------------------
    // 6. Publish Animation / Model to Roblox Modal
    // -----------------------------------------------------------------------

    fn publish_modal_ui(&mut self, ctx: &egui::Context) {
        if self.ui.anim.publish.is_none() {
            return;
        }
        let mut open = true;
        let mut do_upload = false;
        let mut do_sign_in = false;
        let mut copy_id: Option<u64> = None;

        let title = {
            let d = self.ui.anim.publish.as_ref().unwrap();
            format!("Publish {} to Roblox", d.asset_type)
        };

        egui::Window::new(title)
            .id(egui::Id::new("rustudio_publish_asset_modal"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(360.0)
            .show(ctx, |ui| {
            if !rustudio_core::studio_content::builtin_plugins_installed() {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(44, 48, 62))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(8, 4))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new("Built-in Animation Editor (official Roblox plugin not downloaded — get online to download).")
                                    .small()
                                    .color(egui::Color32::from_rgb(245, 200, 95)),
                            );
                            if ui.small_button("Download Official").clicked() {
                                self.ui.anim.plugin_popup_dismissed = false;
                                rustudio_core::studio_content::retry_background_install();
                            }
                        });
                    });
                ui.add_space(4.0);
            }

                let d = self.ui.anim.publish.as_mut().unwrap();
                ui.label("Name:");
                ui.text_edit_singleline(&mut d.name);
                ui.add_space(4.0);
                ui.label("Description:");
                ui.text_edit_multiline(&mut d.description);
                ui.add_space(8.0);

                match &d.status {
                    PublishStatus::Idle => {
                        if ui.button(format!("Upload {} to Roblox", d.asset_type)).clicked() {
                            do_upload = true;
                        }
                    }
                    PublishStatus::Uploading => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Serializing .rbxm and uploading to Roblox Cloud...");
                        });
                    }
                    PublishStatus::Success(id) => {
                        let id_val = *id;
                        ui.label(
                            RichText::new(format!("Published! rbxassetid://{id_val}"))
                                .color(Color32::from_rgb(90, 220, 120))
                                .strong(),
                        );
                        ui.horizontal(|ui| {
                            if ui.button("Copy ID").clicked() {
                                copy_id = Some(id_val);
                            }
                            if ui.button("Upload Again").clicked() {
                                do_upload = true;
                            }
                        });
                    }
                    PublishStatus::Error(err) => {
                        ui.label(RichText::new(format!("Upload failed: {err}")).color(theme::ERROR));
                        if ui.button("Retry Upload").clicked() {
                            do_upload = true;
                        }
                    }
                    PublishStatus::NotSignedIn => {
                        ui.label(
                            RichText::new("You are not signed in to a Roblox account. Sign in first to publish.")
                                .color(theme::WARN),
                        );
                        if ui.button("Sign In with Browser").clicked() {
                            do_sign_in = true;
                        }
                    }
                }
            });

        if let Some(id) = copy_id {
            ctx.copy_text(format!("rbxassetid://{id}"));
            self.info(format!("Copied rbxassetid://{id} to clipboard."));
        }
        if do_sign_in {
            self.game.account.browser_sign_in();
        }
        if do_upload {
            self.start_publish_upload();
        }
        if !open {
            self.ui.anim.publish = None;
        }
    }
}


impl Editor {
    fn builtin_plugins_download_popup_ui(&mut self, ctx: &egui::Context) {
        use rustudio_core::studio_content::{self, DownloadUiStatus};
        if self.ui.anim.plugin_popup_dismissed {
            return;
        }
        let status = studio_content::download_ui_status();
        match status {
            DownloadUiStatus::Idle | DownloadUiStatus::Ready => return,
            DownloadUiStatus::Downloading(msg) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(120));
                egui::Window::new("Downloading Official Roblox Plugins")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                    .fixed_size(egui::vec2(380.0, 130.0))
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(egui::RichText::new("Fetching official Roblox Studio plugins…").strong());
                        });
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(msg).small().color(theme::TEXT_DIM));
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("Once complete, restart or reopen Clip Editor to use the official plugin. You can also continue offline with the built-in editor.")
                                .small()
                                .color(theme::TEXT_DIM),
                        );
                        ui.add_space(8.0);
                        if ui.button("Continue in Background").clicked() {
                            self.ui.anim.plugin_popup_dismissed = true;
                        }
                    });
            }
            DownloadUiStatus::OfflineError(err) => {
                egui::Window::new("Official Roblox Plugins Offline")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                    .fixed_size(egui::vec2(400.0, 155.0))
                    .show(ctx, |ui| {
                        ui.label(
                            egui::RichText::new("Could not download official Roblox Studio plugins (you appear to be offline).")
                                .strong(),
                        );
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new("Rustudio will use its built-in Animation Editor & Avatar tools for now. Connect to the internet and tap Retry to download the official plugins.")
                                .small()
                                .color(theme::TEXT_DIM),
                        );
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(format!("Details: {err}")).small().color(theme::ERROR));
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Retry Download").clicked() {
                                studio_content::retry_background_install();
                            }
                            if ui.button("Use Built-In (Offline)").clicked() {
                                self.ui.anim.plugin_popup_dismissed = true;
                            }
                        });
                    });
            }
        }
    }
}


impl Editor {
    /// Renders the Animation Editor inside a dockable/floating `Tab::AnimEditor`
    /// with vertical scrolling and the offline official-plugin notice banner.
    pub fn anim_editor_panel_ui(&mut self, ui: &mut egui::Ui) {
        self.ui.anim.anim_open = true;
        if !rustudio_core::studio_content::builtin_plugins_installed() {
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(44, 48, 62))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(8, 4))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new("Built-in Animation Editor (official Roblox plugin not downloaded — get online to download).")
                                .small()
                                .color(egui::Color32::from_rgb(245, 200, 95)),
                        );
                        if ui.small_button("Download Official").clicked() {
                            self.ui.anim.plugin_popup_dismissed = false;
                            rustudio_core::studio_content::retry_background_install();
                        }
                    });
                });
            ui.add_space(4.0);
        }
        egui::ScrollArea::vertical()
            .id_salt("anim_editor_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.anim_editor_body_ui(ui);
            });
    }
}
