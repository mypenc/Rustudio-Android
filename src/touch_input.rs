//! Fingers on the editor. egui's own touch support is thin: the first finger
//! is the mouse pointer (with the left button down), so a finger laid on an
//! Explorer row starts a drag of that row instead of scrolling the list,
//! and a finger on the 3D view box-selects instead of turning the camera.
//! This layer sits in `App::raw_input_hook`, before egui reads the frame's
//! events, and turns the raw `Event::Touch` stream into what a touch UI
//! expects, dropping the pointer events egui-winit made up for the finger:
//!
//! * A finger that moves at once scrolls the panel under it (wheel events,
//!   which every `ScrollArea` takes) and keeps going for a moment after the
//!   lift (a fling).
//! * A finger that stays put for [`HOLD`] and then moves is a normal
//!   press and drag (Explorer rows, text selection, sliders, panel
//!   splitters, windows), as press-and-hold does in every mobile UI.
//! * A finger lifted without moving is a tap (a click).
//! * A finger still for [`LONG_PRESS`] is a right click where it landed:
//!   the context menus (Explorer rows, dock tabs, the text fields' Cut /
//!   Copy / Paste menu, src/text_menu.rs). No left press goes out before
//!   it, so a selection under the finger is kept for the menu.
//! * On the 3D view a finger that moves at once is a right-button drag:
//!   the editor camera's look, the game's camera in Play. A finger on a
//!   transform handle drags it at once. A second finger makes a pinch
//!   (zoom) or a two-finger pan, read by the viewport from
//!   `InputState::multi_touch`.
//! * In Play, a finger landing bottom-left is the movement stick and
//!   bottom-right the jump button: they hold the W / A / S / D and Space
//!   keys down, which the character and UserInputService already follow.
//!   That is the fallback for a game run without touch identity
//!   (`RUSTUDIO_TOUCH=0`). With it ([`Zones::game`], the default on
//!   Android), the fingers that land on the game view are the game's:
//!   they go out as they came through [`Touch::take_game_events`] — one
//!   `UserInputType.Touch` InputObject each, for the game's GUIs, its
//!   scripts, and Roblox's own thumbstick and jump button (the
//!   PlayerModule draws those once `UserInputService.TouchEnabled` is
//!   true: src/play_game.rs, "Fingers in Play") — and egui never sees
//!   them: no pointer, no scroll, no pinch of the editor's.
//!
//! A finger's lift leaves the pointer where it was for the rest of that
//! frame: the `PointerGone` goes to egui at the start of the next one.
//! egui's `latest_pos` (what `Response::hover_pos` reads) is cleared by a
//! `PointerGone` in the frame it comes, and the editor's GUI input
//! (src/gui_input.rs, the game's and the plugin widgets' GuiButtons)
//! takes a press or release only where the pointer hovers — a tap whose
//! press, release and `PointerGone` came in one frame was no click there.
//!
//! The viewport tells this layer where things are each frame ([`Zones`]);
//! the rest is [`Touch::filter`].

use std::collections::{BTreeSet, VecDeque};
use std::time::Duration;
use web_time::Instant;

use crate::mech;
use egui::{Event, Key, Modifiers, MouseWheelUnit, PointerButton, Pos2, Rect, TouchId, TouchPhase, Vec2};

// Tuning values: src/mech.rs (touch screens), docs/MECHANICS.md.
/// Still for this long: the finger is a press (then a drag), not a scroll.
pub const HOLD: Duration = Duration::from_millis(mech::TOUCH_HOLD_MS);
/// Moved this far (points) before [`HOLD`]: a scroll (or a look).
pub const SLOP: f32 = mech::TOUCH_SLOP_POINTS;
/// Still for this long on the chrome: a right click (a context menu).
pub const LONG_PRESS: Duration = Duration::from_millis(mech::TOUCH_LONG_PRESS_MS);
/// A fling slower than this (points / s) stops.
const FLING_STOP: f32 = mech::TOUCH_FLING_STOP_POINTS_PER_S;
/// Fling speed decays by e^-FLING_DECAY per second.
const FLING_DECAY: f32 = mech::TOUCH_FLING_DECAY_PER_S;
/// The lift's speed is averaged over the finger's last samples this old.
const FLING_WINDOW: Duration = Duration::from_millis(mech::TOUCH_FLING_WINDOW_MS);
/// The stick's knob moves this far (points) from where the finger landed.
pub const STICK_RADIUS: f32 = mech::TOUCH_STICK_RADIUS_POINTS;
/// The stick does nothing inside this radius.
const STICK_DEAD: f32 = mech::TOUCH_STICK_DEAD_POINTS;
/// The jump button's side (points) and its distance from the view's corner.
const JUMP_SIZE: f32 = mech::TOUCH_JUMP_SIZE_POINTS;
const JUMP_MARGIN: f32 = mech::TOUCH_JUMP_MARGIN_POINTS;

/// Where the movement stick and the jump button are on a view of `rect`
/// while the game has the input: the stick anywhere in the bottom-left
/// part (Roblox's dynamic thumbstick appears under the finger there), the
/// jump button a square in the bottom-right corner.
pub fn play_zones(rect: Rect) -> (Option<Rect>, Option<Rect>) {
    if !rect.is_positive() || rect.width() < 200.0 || rect.height() < 120.0 {
        return (None, None);
    }
    let stick = Rect::from_min_max(
        Pos2::new(rect.min.x, rect.min.y + rect.height() * 0.5),
        Pos2::new(rect.min.x + rect.width() * 0.35, rect.max.y),
    );
    let jump = Rect::from_min_size(
        Pos2::new(rect.max.x - JUMP_MARGIN - JUMP_SIZE, rect.max.y - JUMP_MARGIN - JUMP_SIZE),
        Vec2::splat(JUMP_SIZE),
    );
    (Some(stick), Some(jump))
}

/// What the viewport laid out last frame.
#[derive(Clone, Debug)]
pub struct Zones {
    /// The 3D view's panel (`Rect::NOTHING` when it isn't on screen).
    pub viewport: Rect,
    /// Play, with the game taking the input (the stick and jump show).
    pub playing: bool,
    /// The game takes the fingers that land on the view (the module doc):
    /// reported through [`Touch::take_game_events`]; no stick, no jump
    /// button of this layer's.
    pub game: bool,
    /// The editor's own widgets over the game view (the hotbar's slots,
    /// the chat box): fingers landing there stay egui's with `game` on.
    pub ui: Vec<Rect>,
    /// Where a finger starts the movement stick / presses jump.
    pub stick: Option<Rect>,
    pub jump: Option<Rect>,
    /// A transform handle is under the pointer (a press goes to it at once).
    pub handle_hover: bool,
    /// `Editor::frame_no` when these were written; older zones are stale.
    pub frame: u64,
}

impl Default for Zones {
    fn default() -> Self {
        Self { viewport: Rect::NOTHING, playing: false, game: false, ui: Vec::new(), stick: None, jump: None, handle_hover: false, frame: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Landed; tap, hold or move decide what it is.
    Pending,
    /// Scrolling the chrome.
    Scroll,
    /// A right-button drag on the 3D view (camera).
    Look,
    /// A delivered press (drag until the lift).
    Press,
    /// Two or more fingers on the 3D view: pinch / pan, no pointer.
    Multi,
    /// Delivered a long press (its menu is open): nothing until the lift.
    Done,
}

struct Pointer {
    id: TouchId,
    start: Pos2,
    pos: Pos2,
    since: Instant,
    in_view: bool,
    mode: Mode,
    /// Recent positions for the fling speed.
    samples: VecDeque<(Instant, Pos2)>,
}

struct Fling {
    pos: Pos2,
    vel: Vec2,
    last: Instant,
}

/// A finger the game has ([`Zones::game`]), as it came from the screen
/// (window coordinates, points).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameTouch {
    pub id: TouchId,
    pub phase: TouchPhase,
    pub pos: Pos2,
}

/// Game touches kept between two reads at most (frames that drew no
/// viewport): more than this means nobody is reading them.
const GAME_EVENTS_MAX: usize = 512;

/// The movement stick while a finger holds it.
#[derive(Clone, Copy, Debug)]
pub struct Stick {
    pub id: TouchId,
    /// Where the finger landed (the base).
    pub center: Pos2,
    /// Where it is now.
    pub pos: Pos2,
    since: Instant,
    /// Left the dead zone at some point (a quick lift before that is a
    /// tap on whatever GUI is there, not a stick).
    moved: bool,
}

impl Stick {
    /// The knob's position, within [`STICK_RADIUS`] of the base.
    pub fn knob(&self) -> Pos2 {
        let d = self.pos - self.center;
        let len = d.length();
        if len <= STICK_RADIUS {
            self.pos
        } else {
            self.center + d * (STICK_RADIUS / len)
        }
    }

    /// The movement keys this deflection holds.
    fn keys(&self) -> BTreeSet<Key> {
        let d = self.pos - self.center;
        let len = d.length();
        let mut k = BTreeSet::new();
        if len < STICK_DEAD {
            return k;
        }
        // Eight directions: a component counts past ~22° off the other axis.
        let t = len * 0.38;
        if -d.y > t {
            k.insert(Key::W);
        }
        if d.y > t {
            k.insert(Key::S);
        }
        if d.x > t {
            k.insert(Key::D);
        }
        if -d.x > t {
            k.insert(Key::A);
        }
        k
    }
}

#[derive(Default)]
pub struct Touch {
    pub zones: Zones,
    /// Fingers on the screen now.
    active: BTreeSet<TouchId>,
    pointer: Option<Pointer>,
    fling: Option<Fling>,
    pub stick: Option<Stick>,
    jump: Option<TouchId>,
    /// Keys the stick / jump hold down.
    held: BTreeSet<Key>,
    /// Frames since the last touch: a mouse's own events pass when fingers
    /// have been away for a while.
    pub last_touch: Option<Instant>,
    /// Switched off ([`Touch::set_enabled`]): events pass as they come.
    off: bool,
    /// A `PointerGone` owed to egui at the next frame (the module comment).
    gone: bool,
    /// Fingers that landed while switched off (or were down when it was
    /// switched off): egui's, with the pointer events winit makes for
    /// them, until they lift.
    foreign: BTreeSet<TouchId>,
    /// Fingers the game has ([`Zones::game`]), and their events since
    /// the viewport last read them.
    game_fingers: BTreeSet<TouchId>,
    game_events: Vec<GameTouch>,
}

fn key_event(key: Key, pressed: bool) -> Event {
    Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers: Modifiers::NONE }
}

fn button(pos: Pos2, button: PointerButton, pressed: bool) -> Event {
    Event::PointerButton { pos, button, pressed, modifiers: Modifiers::NONE }
}

fn wheel(delta: Vec2, phase: TouchPhase) -> Event {
    Event::MouseWheel { unit: MouseWheelUnit::Point, delta, phase, modifiers: Modifiers::NONE }
}

impl Touch {
    /// Whether the zones describe the frame that just ended.
    fn zones_fresh(&self, frame_no: u64) -> bool {
        self.zones.frame == frame_no
    }

    /// Whether `pos` is on the 3D view itself (not a window or menu over it).
    fn on_view(&self, ctx: &egui::Context, pos: Pos2, frame_no: u64) -> bool {
        self.zones_fresh(frame_no)
            && self.zones.viewport.contains(pos)
            && ctx.layer_id_at(pos).is_none_or(|l| l.order == egui::Order::Background)
    }

    /// Rewrites the frame's raw input (see the module doc). `frame_no` is
    /// the editor's frame counter, to tell fresh zones from stale ones.
    pub fn filter(&mut self, ctx: &egui::Context, raw: &mut egui::RawInput, frame_no: u64) {
        let now = Instant::now();
        // The pointer of a finger lifted last frame leaves now, ahead of
        // this frame's events.
        if std::mem::take(&mut self.gone) && self.pointer.is_none() {
            raw.events.insert(0, Event::PointerGone);
        }
        // Switched off, or fingers from then still down: egui gets the
        // events as winit made them (its own touch rules: drag-to-scroll,
        // long press menus), until every such finger has lifted.
        if self.off || !self.foreign.is_empty() {
            for ev in &raw.events {
                match ev {
                    Event::Touch { id, phase: TouchPhase::Start, .. } => {
                        self.last_touch = Some(now);
                        self.foreign.insert(*id);
                    }
                    Event::Touch { id, phase: TouchPhase::End | TouchPhase::Cancel, .. } => {
                        self.last_touch = Some(now);
                        self.foreign.remove(id);
                    }
                    Event::Touch { .. } => self.last_touch = Some(now),
                    Event::WindowFocused(false) => self.foreign.clear(),
                    _ => {}
                }
            }
            return;
        }
        let mut out: Vec<Event> = Vec::with_capacity(raw.events.len() + 8);
        // Pointer events in a batch with touch events (or while a finger is
        // down) are egui-winit's stand-ins for the finger: ours replace them.
        let mut synthetic = !self.active.is_empty();
        for ev in raw.events.drain(..) {
            match ev {
                Event::WindowFocused(false) => {
                    // No lifts come for fingers down when the activity
                    // pauses: everything lets go.
                    self.let_go(&mut out);
                    self.active.clear();
                    out.push(ev);
                }
                Event::Touch { id, phase, pos, .. } => {
                    synthetic = true;
                    self.last_touch = Some(now);
                    let control = self.on_touch(ctx, id, phase, pos, now, frame_no, &mut out);
                    // egui keeps seeing the fingers (its multi-touch gestures,
                    // its long press), except the stick and jump ones, which
                    // would otherwise count as a pinch with a look finger.
                    if !control {
                        out.push(ev);
                    }
                }
                Event::PointerMoved(_) | Event::PointerButton { .. } | Event::PointerGone if synthetic => {}
                other => out.push(other),
            }
        }
        self.tick(ctx, now, frame_no, &mut out);
        if self.gone {
            ctx.request_repaint();
        }
        raw.events = out;
    }

    /// One finger event. True when the finger is (or was, for a lift) the
    /// stick or the jump button.
    fn on_touch(
        &mut self,
        ctx: &egui::Context,
        id: TouchId,
        phase: TouchPhase,
        pos: Pos2,
        now: Instant,
        frame_no: u64,
        out: &mut Vec<Event>,
    ) -> bool {
        match phase {
            TouchPhase::Start => {
                self.active.insert(id);
                if self.fling.take().is_some() {
                    out.push(wheel(Vec2::ZERO, TouchPhase::End));
                    out.push(Event::PointerGone);
                }
                let fresh = self.zones_fresh(frame_no);
                if fresh
                    && self.zones.playing
                    && self.zones.game
                    && self.on_view(ctx, pos, frame_no)
                    && !self.zones.ui.iter().any(|r| r.contains(pos))
                {
                    // The game's finger: out through take_game_events,
                    // never egui's.
                    self.game_fingers.insert(id);
                    self.push_game(GameTouch { id, phase, pos });
                    return true;
                }
                if fresh && self.zones.playing {
                    if self.stick.is_none() && self.zones.stick.is_some_and(|z| z.contains(pos)) {
                        self.stick = Some(Stick { id, center: pos, pos, since: now, moved: false });
                        return true;
                    }
                    if self.jump.is_none() && self.zones.jump.is_some_and(|z| z.contains(pos)) {
                        self.jump = Some(id);
                        self.set_held(Key::Space, true, out);
                        return true;
                    }
                }
                if self.pointer.is_none() {
                    // (a lift in this same frame owes no PointerGone: the
                    // pointer is this finger now)
                    self.gone = false;
                    let in_view = self.on_view(ctx, pos, frame_no);
                    self.pointer = Some(Pointer {
                        id,
                        start: pos,
                        pos,
                        since: now,
                        in_view,
                        mode: Mode::Pending,
                        samples: VecDeque::from([(now, pos)]),
                    });
                    // Hover follows at once (the viewport's handle hit test
                    // and the chrome's highlights use it).
                    out.push(Event::PointerMoved(pos));
                } else if let Some(p) = self.pointer.as_mut() {
                    // A second finger on the 3D view: a pinch or a pan from
                    // here on; whatever the first finger was doing ends.
                    if p.in_view && p.mode != Mode::Multi {
                        match p.mode {
                            Mode::Look => out.push(button(p.pos, PointerButton::Secondary, false)),
                            Mode::Press => out.push(button(p.pos, PointerButton::Primary, false)),
                            _ => {}
                        }
                        p.mode = Mode::Multi;
                    }
                }
            }
            TouchPhase::Move => {
                if self.game_fingers.contains(&id) {
                    self.push_game(GameTouch { id, phase, pos });
                    return true;
                }
                if self.stick.is_some_and(|s| s.id == id) {
                    if let Some(s) = self.stick.as_mut() {
                        s.pos = pos;
                        s.moved |= (pos - s.center).length() >= STICK_DEAD;
                    }
                    self.sync_stick_keys(out);
                    return true;
                }
                if self.jump == Some(id) {
                    return true;
                }
                let Some(p) = self.pointer.as_mut() else { return false };
                if p.id != id {
                    return false;
                }
                p.pos = pos;
                p.samples.push_back((now, pos));
                while p.samples.len() > 12 {
                    p.samples.pop_front();
                }
                match p.mode {
                    Mode::Pending => {
                        if (pos - p.start).length() > SLOP {
                            if p.in_view {
                                out.push(Event::PointerMoved(p.start));
                                out.push(button(p.start, PointerButton::Secondary, true));
                                out.push(Event::PointerMoved(pos));
                                p.mode = Mode::Look;
                            } else if now.duration_since(p.since) >= HOLD {
                                // Held, then moved: a drag from where it landed.
                                out.push(Event::PointerMoved(p.start));
                                out.push(button(p.start, PointerButton::Primary, true));
                                out.push(Event::PointerMoved(pos));
                                p.mode = Mode::Press;
                            } else {
                                out.push(wheel(Vec2::ZERO, TouchPhase::Start));
                                out.push(wheel(pos - p.start, TouchPhase::Move));
                                out.push(Event::PointerMoved(pos));
                                p.mode = Mode::Scroll;
                            }
                        }
                    }
                    Mode::Scroll => {
                        let last = p.samples.iter().rev().nth(1).map(|(_, q)| *q).unwrap_or(pos);
                        out.push(wheel(pos - last, TouchPhase::Move));
                        out.push(Event::PointerMoved(pos));
                    }
                    Mode::Look | Mode::Press => out.push(Event::PointerMoved(pos)),
                    Mode::Done => {
                        // Nothing came of the long press (no menu under the
                        // finger): a move from here is the hold-then-drag
                        // of a slow hand (a splitter, a window's title).
                        if !egui::Popup::is_any_open(ctx) && (pos - p.start).length() > SLOP {
                            out.push(Event::PointerMoved(p.start));
                            out.push(button(p.start, PointerButton::Primary, true));
                            out.push(Event::PointerMoved(pos));
                            p.mode = Mode::Press;
                        }
                    }
                    Mode::Multi => {}
                }
            }
            TouchPhase::End | TouchPhase::Cancel => {
                self.active.remove(&id);
                if self.game_fingers.remove(&id) {
                    self.push_game(GameTouch { id, phase, pos });
                    return true;
                }
                if let Some(s) = self.stick.filter(|s| s.id == id) {
                    self.stick = None;
                    self.sync_stick_keys(out);
                    // A quick touch that never deflected: a tap on the game's
                    // GUI there (a button in the corner), when no other
                    // finger is the pointer.
                    if !s.moved && phase == TouchPhase::End && now.duration_since(s.since) < HOLD && self.pointer.is_none() {
                        out.push(Event::PointerMoved(s.center));
                        out.push(button(s.center, PointerButton::Primary, true));
                        out.push(button(s.center, PointerButton::Primary, false));
                        self.gone = true;
                    }
                    return true;
                }
                if self.jump == Some(id) {
                    self.jump = None;
                    self.set_held(Key::Space, false, out);
                    return true;
                }
                let Some(p) = self.pointer.as_ref() else { return false };
                if p.id != id {
                    return false;
                }
                let p = self.pointer.take().expect("checked");
                // (the PointerGone follows at the next frame: `gone`)
                match p.mode {
                    Mode::Pending => {
                        if phase == TouchPhase::End {
                            out.push(Event::PointerMoved(p.start));
                            out.push(button(p.start, PointerButton::Primary, true));
                            out.push(button(p.start, PointerButton::Primary, false));
                        }
                        self.gone = true;
                    }
                    Mode::Scroll => {
                        let vel = fling_velocity(&p.samples, now);
                        if phase == TouchPhase::End && vel.length() > FLING_STOP * 2.0 {
                            self.fling = Some(Fling { pos: p.pos, vel, last: now });
                        } else {
                            out.push(wheel(Vec2::ZERO, TouchPhase::End));
                            self.gone = true;
                        }
                    }
                    Mode::Look => {
                        out.push(Event::PointerMoved(p.pos));
                        out.push(button(p.pos, PointerButton::Secondary, false));
                        self.gone = true;
                    }
                    Mode::Press => {
                        out.push(button(p.pos, PointerButton::Primary, false));
                        self.gone = true;
                    }
                    Mode::Multi | Mode::Done => self.gone = true,
                }
            }
        }
        false
    }

    /// Timers: the hold that turns a pending finger into a press, the
    /// handle check, the fling.
    fn tick(&mut self, ctx: &egui::Context, now: Instant, frame_no: u64, out: &mut Vec<Event>) {
        // Play ended under a held stick or jump: their keys come up.
        if (self.stick.is_some() || self.jump.is_some()) && self.zones_fresh(frame_no) && !self.zones.playing {
            self.release_all(out);
        }
        // Play ended (or the game let go of the fingers) under the game's
        // fingers: they end for the game, and egui doesn't inherit them.
        if !self.game_fingers.is_empty() && self.zones_fresh(frame_no) && !(self.zones.playing && self.zones.game) {
            self.cancel_game_fingers();
        }
        if let Some(p) = self.pointer.as_mut() {
            if p.mode == Mode::Pending {
                let held = now.duration_since(p.since);
                let on_handle = p.in_view && self.zones.frame == frame_no && self.zones.handle_hover;
                if p.in_view && (held >= HOLD || on_handle) {
                    // On the 3D view the press goes out at the hold: a part
                    // is grabbed where the finger rests, then dragged.
                    out.push(Event::PointerMoved(p.start));
                    out.push(button(p.start, PointerButton::Primary, true));
                    if p.pos != p.start {
                        out.push(Event::PointerMoved(p.pos));
                    }
                    p.mode = Mode::Press;
                } else if !p.in_view && held >= LONG_PRESS {
                    // On the chrome a still finger is a right click where
                    // it landed (context menus); the press for a drag waits
                    // for the first move (`on_touch`), so nothing has
                    // touched the widget under it before its menu opens.
                    out.push(Event::PointerMoved(p.start));
                    out.push(button(p.start, PointerButton::Secondary, true));
                    out.push(button(p.start, PointerButton::Secondary, false));
                    p.mode = Mode::Done;
                } else {
                    let next = if p.in_view { HOLD } else { LONG_PRESS };
                    ctx.request_repaint_after(next.saturating_sub(held).max(Duration::from_millis(16)));
                }
            }
        }
        if let Some(f) = self.fling.as_mut() {
            let dt = now.duration_since(f.last).as_secs_f32().min(0.1);
            f.last = now;
            let step = f.vel * dt;
            f.vel *= (-FLING_DECAY * dt).exp();
            out.push(Event::PointerMoved(f.pos));
            out.push(wheel(step, TouchPhase::Move));
            if f.vel.length() < FLING_STOP {
                self.fling = None;
                out.push(wheel(Vec2::ZERO, TouchPhase::End));
                out.push(Event::PointerGone);
            } else {
                ctx.request_repaint();
            }
        }
    }

    fn set_held(&mut self, key: Key, down: bool, out: &mut Vec<Event>) {
        let was = self.held.contains(&key);
        if down && !was {
            self.held.insert(key);
            out.push(key_event(key, true));
        } else if !down && was {
            self.held.remove(&key);
            out.push(key_event(key, false));
        }
    }

    fn sync_stick_keys(&mut self, out: &mut Vec<Event>) {
        let want = self.stick.map(|s| s.keys()).unwrap_or_default();
        for k in [Key::W, Key::A, Key::S, Key::D] {
            self.set_held(k, want.contains(&k), out);
        }
    }

    /// Switches the layer off or on. Off, the frame's events reach egui
    /// as winit made them: the Start Page is plain egui (scroll areas
    /// that scroll by drag, cards that are taps, long presses for menus)
    /// with nothing of the editor's to tell a scroll from a drag for.
    /// Switching off lets go of everything held (the events go in front
    /// of this frame's); fingers down at a switch stay egui's until they
    /// lift, so no press is left half delivered.
    pub fn set_enabled(&mut self, on: bool, raw: &mut egui::RawInput) {
        if on == !self.off {
            return;
        }
        self.off = !on;
        if on {
            return;
        }
        let mut out: Vec<Event> = Vec::new();
        self.let_go(&mut out);
        let down = std::mem::take(&mut self.active);
        self.foreign.extend(down);
        if !out.is_empty() {
            out.append(&mut raw.events);
            raw.events = out;
        }
    }

    /// Whether the layer is rewriting events (not switched off).
    pub fn enabled(&self) -> bool {
        !self.off
    }

    /// Ends whatever is in progress the way a lift would, without a tap:
    /// the held keys, a press or look's button, a scroll or fling's wheel.
    fn let_go(&mut self, out: &mut Vec<Event>) {
        self.release_all(out);
        self.cancel_game_fingers();
        if let Some(p) = self.pointer.take() {
            match p.mode {
                Mode::Pending | Mode::Multi | Mode::Done => {}
                Mode::Scroll => out.push(wheel(Vec2::ZERO, TouchPhase::End)),
                Mode::Look => {
                    out.push(Event::PointerMoved(p.pos));
                    out.push(button(p.pos, PointerButton::Secondary, false));
                }
                Mode::Press => out.push(button(p.pos, PointerButton::Primary, false)),
            }
            out.push(Event::PointerGone);
        }
        self.gone = false;
        if self.fling.take().is_some() {
            out.push(wheel(Vec2::ZERO, TouchPhase::End));
            out.push(Event::PointerGone);
        }
    }

    /// Lets go of everything (the window lost focus, Play stopped).
    pub fn release_all(&mut self, out: &mut Vec<Event>) {
        self.stick = None;
        self.jump = None;
        for k in [Key::W, Key::A, Key::S, Key::D, Key::Space] {
            self.set_held(k, false, out);
        }
    }

    fn push_game(&mut self, t: GameTouch) {
        if self.game_events.len() >= GAME_EVENTS_MAX {
            // Nobody is reading: keep the ends, drop the moves.
            self.game_events.retain(|e| matches!(e.phase, TouchPhase::End | TouchPhase::Cancel));
            if self.game_events.len() >= GAME_EVENTS_MAX {
                self.game_events.clear();
            }
        }
        self.game_events.push(t);
    }

    /// Ends every finger the game has, as a cancel.
    fn cancel_game_fingers(&mut self) {
        for id in std::mem::take(&mut self.game_fingers) {
            let pos = self.game_events.iter().rev().find(|e| e.id == id).map(|e| e.pos).unwrap_or(Pos2::ZERO);
            self.push_game(GameTouch { id, phase: TouchPhase::Cancel, pos });
        }
    }

    /// The game's fingers' events since the last call ([`Zones::game`]),
    /// in the order they came.
    pub fn take_game_events(&mut self) -> Vec<GameTouch> {
        std::mem::take(&mut self.game_events)
    }

    /// Whether `id` is a finger the game has now.
    pub fn is_game_finger(&self, id: TouchId) -> bool {
        self.game_fingers.contains(&id)
    }

    /// The pointer finger's mode, for the viewport (a look in progress
    /// hides the hover outline, for one).
    pub fn mode(&self) -> Option<Mode> {
        self.pointer.as_ref().map(|p| p.mode)
    }

    /// Whether a finger is on the screen.
    pub fn any(&self) -> bool {
        !self.active.is_empty()
    }

    /// Whether the jump button is held.
    pub fn jumping(&self) -> bool {
        self.jump.is_some()
    }

    /// Draws the Play controls over the game view: the jump button in its
    /// corner, the stick's base and knob while a finger holds it.
    pub fn paint(&self, painter: &egui::Painter) {
        use egui::{Color32, Stroke, StrokeKind};
        if !self.zones.playing || self.zones.game {
            return;
        }
        let dim = Color32::from_rgba_unmultiplied(255, 255, 255, 40);
        let edge = Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 255, 255, 110));
        let lit = Color32::from_rgba_unmultiplied(255, 255, 255, 90);
        if let Some(j) = self.zones.jump {
            let fill = if self.jump.is_some() { lit } else { dim };
            painter.rect(j, j.width() * 0.5, fill, edge, StrokeKind::Inside);
            painter.text(
                j.center(),
                egui::Align2::CENTER_CENTER,
                "⤒",
                egui::FontId::proportional(26.0),
                Color32::from_rgba_unmultiplied(255, 255, 255, 190),
            );
        }
        if let Some(s) = self.stick {
            painter.circle(s.center, STICK_RADIUS, dim, edge);
            painter.circle(s.knob(), STICK_RADIUS * 0.45, lit, edge);
        }
    }
}

/// Whether the game gets the fingers (touch identity: the module doc,
/// src/play_game.rs "Fingers in Play"): on Android unless `RUSTUDIO_TOUCH=0`,
/// elsewhere with `RUSTUDIO_TOUCH=1` (a touch-screen laptop; the game then
/// reports UserInputService.TouchEnabled and PreferredInput Touch too).
pub fn game_touch() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_GAME_TOUCH.with(|c| c.get()) {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("RUSTUDIO_TOUCH").as_deref() {
        Ok("0") | Ok("off") | Ok("false") => false,
        Ok("1") | Ok("on") | Ok("true") => true,
        _ => cfg!(target_os = "android"),
    })
}

#[cfg(test)]
thread_local! {
    /// A test's answer for [`game_touch`] on its own thread (the process
    /// setting is read once; runtime.rs `soft_keyboard_reaches_the_scripts`).
    pub static TEST_GAME_TOUCH: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The soft keyboard over the game's view, for the scripts:
/// `UserInputService.OnScreenKeyboardVisible / Position / Size`
/// (runtime_ext.rs `device_prop`). In the game's coordinates (what
/// `InputObject.Position` and `AbsolutePosition` use: points from the
/// view's top-left, below the top inset), as Roblox's client reports
/// them. Written each frame by the viewport (`Editor::keyboard_pan`) on
/// Android while a game TextBox is typed into; `None` otherwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftKeyboard {
    /// Its top-left corner.
    pub x: f32,
    pub y: f32,
    /// Its size.
    pub w: f32,
    pub h: f32,
}

static SOFT_KEYBOARD: std::sync::Mutex<Option<SoftKeyboard>> = std::sync::Mutex::new(None);

pub fn set_soft_keyboard(k: Option<SoftKeyboard>) {
    *SOFT_KEYBOARD.lock().unwrap_or_else(|e| e.into_inner()) = k;
}

/// Where the soft keyboard lies over the game's view (`None`: it is down).
pub fn soft_keyboard() -> Option<SoftKeyboard> {
    *SOFT_KEYBOARD.lock().unwrap_or_else(|e| e.into_inner())
}

/// Speed over the last ~100 ms of samples (points / s).
fn fling_velocity(samples: &VecDeque<(Instant, Pos2)>, now: Instant) -> Vec2 {
    let Some(&(t_last, p_last)) = samples.back() else { return Vec2::ZERO };
    // A finger that stopped before lifting doesn't fling.
    if now.duration_since(t_last) > Duration::from_millis(80) {
        return Vec2::ZERO;
    }
    let first = samples.iter().find(|(t, _)| t_last.duration_since(*t) <= FLING_WINDOW).copied();
    match first {
        Some((t0, p0)) if t_last > t0 => (p_last - p0) / t_last.duration_since(t0).as_secs_f32(),
        _ => Vec2::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(id: u64, phase: TouchPhase, pos: Pos2) -> Event {
        Event::Touch { device_id: egui::TouchDeviceId(0), id: TouchId(id), phase, pos, force: None }
    }

    fn run(t: &mut Touch, ctx: &egui::Context, events: Vec<Event>) -> Vec<Event> {
        let mut raw = egui::RawInput { events, ..Default::default() };
        t.filter(ctx, &mut raw, 1);
        raw.events
    }

    #[test]
    fn a_moving_finger_scrolls_and_never_presses() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        let a = run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, Pos2::new(100.0, 100.0)), Event::PointerMoved(Pos2::new(100.0, 100.0)), button(Pos2::new(100.0, 100.0), PointerButton::Primary, true)]);
        assert!(!a.iter().any(|e| matches!(e, Event::PointerButton { .. })), "{a:?}");
        let b = run(&mut t, &ctx, vec![touch(1, TouchPhase::Move, Pos2::new(100.0, 130.0))]);
        let wheel_y: f32 = b
            .iter()
            .filter_map(|e| match e {
                Event::MouseWheel { delta, .. } => Some(delta.y),
                _ => None,
            })
            .sum();
        assert!((wheel_y - 30.0).abs() < 1e-3, "{b:?}");
        let c = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(100.0, 130.0))]);
        assert!(!c.iter().any(|e| matches!(e, Event::PointerButton { .. })), "{c:?}");
    }

    #[test]
    fn switched_off_events_pass_untouched_and_a_finger_down_at_the_switch_stays_egui_s() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        let mut raw = egui::RawInput::default();
        t.set_enabled(false, &mut raw);
        assert!(raw.events.is_empty());
        // Off: winit's stand-in pointer events reach egui as they are.
        let events = vec![touch(1, TouchPhase::Start, Pos2::new(10.0, 10.0)), Event::PointerMoved(Pos2::new(10.0, 10.0)), button(Pos2::new(10.0, 10.0), PointerButton::Primary, true)];
        let out = run(&mut t, &ctx, events.clone());
        assert_eq!(format!("{out:?}"), format!("{events:?}"));
        assert!(!t.any());
        // Switched on with that finger still down: its events (and the
        // release winit makes for it) still pass; the next finger is ours.
        t.set_enabled(true, &mut raw);
        let events = vec![touch(1, TouchPhase::End, Pos2::new(10.0, 10.0)), button(Pos2::new(10.0, 10.0), PointerButton::Primary, false), Event::PointerGone];
        let out = run(&mut t, &ctx, events.clone());
        assert_eq!(format!("{out:?}"), format!("{events:?}"));
        let out = run(&mut t, &ctx, vec![touch(2, TouchPhase::Start, Pos2::new(10.0, 10.0)), button(Pos2::new(10.0, 10.0), PointerButton::Primary, true)]);
        assert!(!out.iter().any(|e| matches!(e, Event::PointerButton { .. })), "{out:?}");
        assert!(t.any());
    }

    #[test]
    fn switching_off_lets_go_of_a_press_in_front_of_the_frame() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        t.zones = Zones { viewport: Rect::from_min_max(Pos2::ZERO, Pos2::new(500.0, 500.0)), frame: 1, ..Default::default() };
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, Pos2::new(10.0, 10.0))]);
        rt_thread::sleep(HOLD + Duration::from_millis(20));
        let out = run(&mut t, &ctx, vec![]);
        assert!(out.iter().any(|e| matches!(e, Event::PointerButton { pressed: true, .. })), "{out:?}");
        let mut raw = egui::RawInput { events: vec![touch(1, TouchPhase::Move, Pos2::new(20.0, 10.0))], ..Default::default() };
        t.set_enabled(false, &mut raw);
        assert!(matches!(raw.events[0], Event::PointerButton { button: PointerButton::Primary, pressed: false, .. }), "{:?}", raw.events);
        assert!(matches!(raw.events[1], Event::PointerGone), "{:?}", raw.events);
        assert!(matches!(raw.events.last(), Some(Event::Touch { .. })), "{:?}", raw.events);
        assert!(t.mode().is_none() && !t.any());
    }

    #[test]
    fn a_still_finger_lifted_is_a_tap() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, Pos2::new(10.0, 10.0))]);
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(12.0, 11.0))]);
        let presses: Vec<bool> = out
            .iter()
            .filter_map(|e| match e {
                Event::PointerButton { button: PointerButton::Primary, pressed, .. } => Some(*pressed),
                _ => None,
            })
            .collect();
        assert_eq!(presses, vec![true, false]);
        // The pointer stays where the release was for the rest of the frame
        // (`Response::hover_pos` for the release) and leaves at the next.
        assert!(!out.iter().any(|e| matches!(e, Event::PointerGone)), "{out:?}");
        let next = run(&mut t, &ctx, vec![]);
        assert!(matches!(next.first(), Some(Event::PointerGone)), "{next:?}");
        // Once.
        let after = run(&mut t, &ctx, vec![]);
        assert!(!after.iter().any(|e| matches!(e, Event::PointerGone)), "{after:?}");
    }

    #[test]
    fn a_tap_followed_by_a_finger_in_the_same_frame_owes_no_pointer_gone() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, Pos2::new(10.0, 10.0))]);
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(10.0, 10.0)), touch(2, TouchPhase::Start, Pos2::new(30.0, 30.0))]);
        assert_eq!(buttons(&out), vec![(PointerButton::Primary, true), (PointerButton::Primary, false)], "{out:?}");
        assert!(!out.iter().any(|e| matches!(e, Event::PointerGone)), "{out:?}");
        let next = run(&mut t, &ctx, vec![]);
        assert!(!next.iter().any(|e| matches!(e, Event::PointerGone)), "{next:?}");
        assert!(t.any());
    }

    fn buttons(out: &[Event]) -> Vec<(PointerButton, bool)> {
        out.iter()
            .filter_map(|e| match e {
                Event::PointerButton { button, pressed, .. } => Some((*button, *pressed)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn on_the_chrome_a_still_finger_is_a_right_click_and_nothing_before() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        let at = Pos2::new(10.0, 10.0);
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, at)]);
        rt_thread::sleep(HOLD + Duration::from_millis(20));
        // Held past the hold: still no press (a selection under the finger
        // stays for the menu).
        let out = run(&mut t, &ctx, vec![]);
        assert!(buttons(&out).is_empty(), "{out:?}");
        assert_eq!(t.mode(), Some(Mode::Pending));
        rt_thread::sleep(LONG_PRESS.saturating_sub(HOLD));
        let out = run(&mut t, &ctx, vec![]);
        assert_eq!(buttons(&out), vec![(PointerButton::Secondary, true), (PointerButton::Secondary, false)], "{out:?}");
        assert!(out.iter().any(|e| matches!(e, Event::PointerMoved(p) if *p == at)), "{out:?}");
        assert_eq!(t.mode(), Some(Mode::Done));
        // The lift is neither a tap nor a release; the pointer leaves at
        // the next frame.
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, at)]);
        assert!(buttons(&out).is_empty(), "{out:?}");
        assert!(!out.iter().any(|e| matches!(e, Event::PointerGone)), "{out:?}");
        assert!(t.mode().is_none());
        let next = run(&mut t, &ctx, vec![]);
        assert!(matches!(next.first(), Some(Event::PointerGone)), "{next:?}");
    }

    #[test]
    fn a_long_press_that_opened_nothing_still_drags_when_moved() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        let at = Pos2::new(10.0, 10.0);
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, at)]);
        rt_thread::sleep(LONG_PRESS + Duration::from_millis(20));
        let out = run(&mut t, &ctx, vec![]);
        assert_eq!(buttons(&out), vec![(PointerButton::Secondary, true), (PointerButton::Secondary, false)], "{out:?}");
        assert_eq!(t.mode(), Some(Mode::Done));
        // No popup opened (a plain egui context): the move is a drag.
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::Move, Pos2::new(60.0, 10.0))]);
        assert_eq!(buttons(&out), vec![(PointerButton::Primary, true)], "{out:?}");
        assert_eq!(t.mode(), Some(Mode::Press));
    }

    #[test]
    fn on_the_chrome_a_held_finger_that_moves_drags_from_where_it_landed() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        let at = Pos2::new(10.0, 10.0);
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, at)]);
        rt_thread::sleep(HOLD + Duration::from_millis(20));
        run(&mut t, &ctx, vec![]);
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::Move, Pos2::new(40.0, 10.0))]);
        assert_eq!(buttons(&out), vec![(PointerButton::Primary, true)], "{out:?}");
        assert!(matches!(out.first(), Some(Event::PointerMoved(p)) if *p == at), "{out:?}");
        assert!(out.iter().any(|e| matches!(e, Event::PointerMoved(p) if *p == Pos2::new(40.0, 10.0))), "{out:?}");
        assert!(!out.iter().any(|e| matches!(e, Event::MouseWheel { .. })), "{out:?}");
        assert_eq!(t.mode(), Some(Mode::Press));
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(40.0, 10.0))]);
        assert_eq!(buttons(&out), vec![(PointerButton::Primary, false)], "{out:?}");
        assert!(!out.iter().any(|e| matches!(e, Event::PointerGone)), "{out:?}");
        assert!(matches!(run(&mut t, &ctx, vec![]).first(), Some(Event::PointerGone)));
    }

    #[test]
    fn on_the_view_a_moving_finger_is_a_right_drag() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        t.zones = Zones { viewport: Rect::from_min_max(Pos2::ZERO, Pos2::new(500.0, 500.0)), frame: 1, ..Default::default() };
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, Pos2::new(100.0, 100.0))]);
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::Move, Pos2::new(140.0, 100.0))]);
        assert!(out.iter().any(|e| matches!(e, Event::PointerButton { button: PointerButton::Secondary, pressed: true, .. })), "{out:?}");
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(140.0, 100.0))]);
        assert!(out.iter().any(|e| matches!(e, Event::PointerButton { button: PointerButton::Secondary, pressed: false, .. })), "{out:?}");
    }

    #[test]
    fn the_stick_holds_movement_keys() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        let view = Rect::from_min_max(Pos2::ZERO, Pos2::new(800.0, 400.0));
        t.zones = Zones {
            viewport: view,
            playing: true,
            stick: Some(Rect::from_min_max(Pos2::new(0.0, 200.0), Pos2::new(300.0, 400.0))),
            jump: Some(Rect::from_min_max(Pos2::new(700.0, 300.0), Pos2::new(800.0, 400.0))),
            frame: 1,
            ..Default::default()
        };
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, Pos2::new(100.0, 300.0))]);
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::Move, Pos2::new(100.0, 240.0))]);
        assert!(out.iter().any(|e| matches!(e, Event::Key { key: Key::W, pressed: true, .. })), "{out:?}");
        let out = run(&mut t, &ctx, vec![touch(2, TouchPhase::Start, Pos2::new(750.0, 350.0))]);
        assert!(out.iter().any(|e| matches!(e, Event::Key { key: Key::Space, pressed: true, .. })), "{out:?}");
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(100.0, 240.0)), touch(2, TouchPhase::End, Pos2::new(750.0, 350.0))]);
        assert!(out.iter().any(|e| matches!(e, Event::Key { key: Key::W, pressed: false, .. })), "{out:?}");
        assert!(out.iter().any(|e| matches!(e, Event::Key { key: Key::Space, pressed: false, .. })), "{out:?}");
        assert!(t.held.is_empty());
    }

    fn play_zones_for_tests() -> Zones {
        let view = Rect::from_min_max(Pos2::ZERO, Pos2::new(800.0, 400.0));
        let (stick, jump) = play_zones(view);
        Zones { viewport: view, playing: true, stick, jump, frame: 1, ..Default::default() }
    }

    #[test]
    fn control_fingers_stay_out_of_eguis_touch_stream() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        t.zones = play_zones_for_tests();
        let jump = t.zones.jump.unwrap().center();
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, jump)]);
        assert!(!out.iter().any(|e| matches!(e, Event::Touch { .. })), "{out:?}");
        assert!(out.iter().any(|e| matches!(e, Event::Key { key: Key::Space, pressed: true, .. })), "{out:?}");
        // A look finger alongside keeps its own touch event (egui's gestures).
        let out = run(&mut t, &ctx, vec![touch(2, TouchPhase::Start, Pos2::new(600.0, 100.0))]);
        assert!(out.iter().any(|e| matches!(e, Event::Touch { .. })), "{out:?}");
    }

    #[test]
    fn a_quick_touch_in_the_stick_zone_is_a_tap() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        t.zones = play_zones_for_tests();
        let at = t.zones.stick.unwrap().center();
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, at)]);
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, at + Vec2::new(2.0, -3.0))]);
        let presses: Vec<bool> = out
            .iter()
            .filter_map(|e| match e {
                Event::PointerButton { button: PointerButton::Primary, pressed, .. } => Some(*pressed),
                _ => None,
            })
            .collect();
        assert_eq!(presses, vec![true, false], "{out:?}");
        assert!(!out.iter().any(|e| matches!(e, Event::Key { .. })), "{out:?}");
    }

    #[test]
    fn game_fingers_go_to_the_game_and_never_to_egui() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        t.zones = Zones { viewport: Rect::from_min_max(Pos2::ZERO, Pos2::new(800.0, 400.0)), playing: true, game: true, frame: 1, ..Default::default() };
        let start = vec![touch(1, TouchPhase::Start, Pos2::new(100.0, 300.0)), Event::PointerMoved(Pos2::new(100.0, 300.0)), button(Pos2::new(100.0, 300.0), PointerButton::Primary, true)];
        let out = run(&mut t, &ctx, start);
        assert!(out.is_empty(), "{out:?}");
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::Move, Pos2::new(160.0, 300.0)), touch(2, TouchPhase::Start, Pos2::new(700.0, 350.0))]);
        assert!(out.is_empty(), "{out:?}");
        assert!(t.is_game_finger(TouchId(1)) && t.is_game_finger(TouchId(2)));
        let got = t.take_game_events();
        let phases: Vec<(u64, TouchPhase)> = got.iter().map(|g| (g.id.0, g.phase)).collect();
        assert_eq!(phases, vec![(1, TouchPhase::Start), (1, TouchPhase::Move), (2, TouchPhase::Start)]);
        assert_eq!(got[1].pos, Pos2::new(160.0, 300.0));
        // No stick keys, no pointer: the game draws its own controls.
        assert!(t.held.is_empty() && t.pointer.is_none());
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(160.0, 300.0))]);
        assert!(out.is_empty(), "{out:?}");
        assert!(!t.is_game_finger(TouchId(1)) && t.is_game_finger(TouchId(2)));
        // Play ends under the second finger: it is cancelled for the game.
        t.zones.playing = false;
        let out = run(&mut t, &ctx, vec![]);
        assert!(out.is_empty(), "{out:?}");
        let got = t.take_game_events();
        let phases: Vec<(u64, TouchPhase)> = got.iter().map(|g| (g.id.0, g.phase)).collect();
        assert_eq!(phases, vec![(1, TouchPhase::End), (2, TouchPhase::Cancel)]);
        assert!(!t.is_game_finger(TouchId(2)));
    }

    #[test]
    fn a_finger_on_the_editors_widgets_stays_the_editors_with_game_fingers_on() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        t.zones = Zones {
            viewport: Rect::from_min_max(Pos2::ZERO, Pos2::new(800.0, 400.0)),
            playing: true,
            game: true,
            ui: vec![Rect::from_min_max(Pos2::new(300.0, 350.0), Pos2::new(500.0, 400.0))],
            frame: 1,
            ..Default::default()
        };
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, Pos2::new(400.0, 380.0))]);
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(400.0, 380.0))]);
        assert_eq!(buttons(&out), vec![(PointerButton::Primary, true), (PointerButton::Primary, false)], "{out:?}");
        assert!(t.take_game_events().is_empty());
    }

    #[test]
    fn a_finger_off_the_view_stays_the_editors_with_game_fingers_on() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        t.zones = Zones { viewport: Rect::from_min_max(Pos2::new(0.0, 100.0), Pos2::new(800.0, 400.0)), playing: true, game: true, frame: 1, ..Default::default() };
        // On the chrome above the view: the usual tap.
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, Pos2::new(100.0, 50.0))]);
        let out = run(&mut t, &ctx, vec![touch(1, TouchPhase::End, Pos2::new(100.0, 50.0))]);
        assert_eq!(buttons(&out), vec![(PointerButton::Primary, true), (PointerButton::Primary, false)], "{out:?}");
        assert!(t.take_game_events().is_empty());
    }

    #[test]
    fn losing_focus_lets_go_of_everything() {
        let ctx = egui::Context::default();
        let mut t = Touch::default();
        t.zones = play_zones_for_tests();
        let jump = t.zones.jump.unwrap().center();
        run(&mut t, &ctx, vec![touch(1, TouchPhase::Start, jump)]);
        run(&mut t, &ctx, vec![touch(2, TouchPhase::Start, Pos2::new(400.0, 100.0))]);
        let out = run(&mut t, &ctx, vec![Event::WindowFocused(false)]);
        assert!(out.iter().any(|e| matches!(e, Event::Key { key: Key::Space, pressed: false, .. })), "{out:?}");
        assert!(out.iter().any(|e| matches!(e, Event::PointerGone)), "{out:?}");
        assert!(out.iter().any(|e| matches!(e, Event::WindowFocused(false))), "{out:?}");
        assert!(!t.any() && t.held.is_empty() && t.pointer.is_none());
    }
}
