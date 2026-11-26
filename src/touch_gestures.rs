//! UserInputService's touch gestures from the game's fingers (the
//! `TouchOut` stream src/gui_input.rs `touch_input` leaves, in GUI
//! coordinates): TouchTap / TouchTapInWorld, TouchLongPress, TouchPan,
//! TouchPinch, TouchRotate and TouchSwipe, with the arguments the Creator
//! Hub gives them (`touchPositions: {Vector2}`, the gesture's state as
//! `Enum.UserInputState` Begin / Change / End, `gameProcessedEvent`).
//! The thresholds are src/mech.rs's `TOUCH_TAP_*`, `TOUCH_LONG_PRESS_GAME_MS`,
//! `TOUCH_PAN_SLOP_PX`, `TOUCH_SWIPE_*` (docs/MECHANICS.md, "Touch screens").
//!
//! One group of fingers at a time: the pan follows the fingers' centre
//! (re-based when a finger lands or lifts, as iOS's pan recognizer does),
//! the pinch and the rotation the first two fingers. A finger that moved
//! past the pan slop, or held past the long-press time, is no tap. The
//! gesture's `gameProcessedEvent` is whether its first finger landed on
//! the GUI (`TouchOut::sunk`).

use std::collections::VecDeque;
use std::time::Duration;
use web_time::Instant;

use crate::gui::{TouchOut, TouchState};
use crate::mech;

/// A gesture's `Enum.UserInputState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Begin,
    Change,
    End,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Begin => "Begin",
            Phase::Change => "Change",
            Phase::End => "End",
        }
    }
}

/// `Enum.SwipeDirection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swipe {
    Right,
    Left,
    Up,
    Down,
}

impl Swipe {
    pub fn name(self) -> &'static str {
        match self {
            Swipe::Right => "Right",
            Swipe::Left => "Left",
            Swipe::Up => "Up",
            Swipe::Down => "Down",
        }
    }
}

/// One UserInputService gesture event. Positions are GUI coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum Gesture {
    /// `TouchTap(touchPositions, gameProcessedEvent)`.
    Tap { positions: Vec<(f32, f32)>, gpe: bool },
    /// `TouchTapInWorld(position, processedByUI)`.
    TapInWorld { pos: (f32, f32), processed_by_ui: bool },
    /// `TouchLongPress(touchPositions, state, gameProcessedEvent)`.
    LongPress { positions: Vec<(f32, f32)>, state: Phase, gpe: bool },
    /// `TouchPan(touchPositions, totalTranslation, velocity, state, gameProcessedEvent)`.
    Pan { positions: Vec<(f32, f32)>, translation: (f32, f32), velocity: (f32, f32), state: Phase, gpe: bool },
    /// `TouchPinch(touchPositions, scale, velocity, state, gameProcessedEvent)`:
    /// the fingers' distance over the one they began at, and its rate (/s).
    Pinch { positions: Vec<(f32, f32)>, scale: f32, velocity: f32, state: Phase, gpe: bool },
    /// `TouchRotate(touchPositions, rotation, velocity, state, gameProcessedEvent)`:
    /// degrees turned since the gesture began, and the rate (°/s).
    Rotate { positions: Vec<(f32, f32)>, rotation: f32, velocity: f32, state: Phase, gpe: bool },
    /// `TouchSwipe(swipeDirection, numberOfTouches, gameProcessedEvent)`.
    Swipe { direction: Swipe, touches: u32, gpe: bool },
}

#[derive(Debug, Clone)]
struct Finger {
    id: u64,
    start: (f32, f32),
    pos: (f32, f32),
    since: Instant,
    sunk: bool,
    /// Moved past the pan slop at some point (no tap, no long press start).
    moved: bool,
    /// Recent samples (time, position) for the lift's speed.
    samples: VecDeque<(Instant, (f32, f32))>,
}

#[derive(Debug, Default)]
struct Pan {
    on: bool,
    /// The fingers' centre when the pan began / was last re-based, and the
    /// translation accumulated before that base.
    base: (f32, f32),
    carried: (f32, f32),
    last: Option<(Instant, (f32, f32))>,
    velocity: (f32, f32),
}

#[derive(Debug, Default)]
struct Pinch {
    on: bool,
    d0: f32,
    last: Option<(Instant, f32)>,
    velocity: f32,
}

#[derive(Debug, Default)]
struct Rotate {
    on: bool,
    /// Unwrapped angle so far (degrees).
    total: f32,
    prev: f32,
    last: Option<(Instant, f32)>,
    velocity: f32,
}

/// The recognizer: feed it each finger event ([`Gestures::feed`]) and
/// each frame's time ([`Gestures::tick`], for the long press).
#[derive(Debug, Default)]
pub struct Gestures {
    fingers: Vec<Finger>,
    pan: Pan,
    pinch: Pinch,
    rotate: Rotate,
    long_press: bool,
    /// The group panned, pinched or turned: none of its fingers taps.
    gestured: bool,
}

const SPEED_WINDOW: Duration = Duration::from_millis(100);
/// A rotation smaller than this (degrees) hasn't begun.
const ROTATE_SLOP_DEG: f32 = 5.0;

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn angle(a: (f32, f32), b: (f32, f32)) -> f32 {
    (b.1 - a.1).atan2(b.0 - a.0).to_degrees()
}

impl Gestures {
    fn positions(&self) -> Vec<(f32, f32)> {
        self.fingers.iter().map(|f| f.pos).collect()
    }

    fn gpe(&self) -> bool {
        self.fingers.first().is_some_and(|f| f.sunk)
    }

    fn centre(&self) -> (f32, f32) {
        let n = self.fingers.len().max(1) as f32;
        let (sx, sy) = self.fingers.iter().fold((0.0, 0.0), |(x, y), f| (x + f.pos.0, y + f.pos.1));
        (sx / n, sy / n)
    }

    /// Speed of a finger over its last samples (px/s).
    fn speed(f: &Finger, now: Instant) -> (f32, f32) {
        let Some(&(t_last, p_last)) = f.samples.back() else { return (0.0, 0.0) };
        if now.duration_since(t_last) > Duration::from_millis(80) {
            return (0.0, 0.0);
        }
        let first = f.samples.iter().find(|(t, _)| t_last.duration_since(*t) <= SPEED_WINDOW).copied();
        match first {
            Some((t0, p0)) if t_last > t0 => {
                let dt = t_last.duration_since(t0).as_secs_f32();
                ((p_last.0 - p0.0) / dt, (p_last.1 - p0.1) / dt)
            }
            _ => (0.0, 0.0),
        }
    }

    /// Pan translation now (since it began).
    fn translation(&self) -> (f32, f32) {
        let c = self.centre();
        (self.pan.carried.0 + c.0 - self.pan.base.0, self.pan.carried.1 + c.1 - self.pan.base.1)
    }

    /// Re-bases the pan when the finger count changes (the centre jumps).
    fn rebase_pan(&mut self) {
        if self.pan.on {
            self.pan.carried = self.translation();
        }
        self.pan.base = self.centre();
    }

    /// One finger event, in order. `now` is when it happened.
    pub fn feed(&mut self, ev: &TouchOut, now: Instant) -> Vec<Gesture> {
        let mut out = Vec::new();
        match ev.state {
            TouchState::Begin => {
                // The group's end before a new group starts (a lone second
                // finger after a long press is still the same group).
                self.fingers.push(Finger {
                    id: ev.id,
                    start: (ev.x, ev.y),
                    pos: (ev.x, ev.y),
                    since: now,
                    sunk: ev.sunk,
                    moved: false,
                    samples: VecDeque::from([(now, (ev.x, ev.y))]),
                });
                self.rebase_pan();
                if self.fingers.len() == 2 {
                    let (a, b) = (self.fingers[0].pos, self.fingers[1].pos);
                    self.pinch = Pinch { on: false, d0: dist(a, b).max(1.0), last: Some((now, 1.0)), velocity: 0.0 };
                    let a0 = angle(a, b);
                    self.rotate = Rotate { on: false, total: 0.0, prev: a0, last: Some((now, 0.0)), velocity: 0.0 };
                }
            }
            TouchState::Change => {
                let Some(i) = self.fingers.iter().position(|f| f.id == ev.id) else { return out };
                {
                    let f = &mut self.fingers[i];
                    f.pos = (ev.x, ev.y);
                    f.samples.push_back((now, f.pos));
                    while f.samples.len() > 16 {
                        f.samples.pop_front();
                    }
                    if !f.moved && dist(f.pos, f.start) > mech::TOUCH_PAN_SLOP_PX {
                        f.moved = true;
                    }
                }
                let any_moved = self.fingers.iter().any(|f| f.moved);
                // Long press moving: Change.
                if self.long_press {
                    out.push(Gesture::LongPress { positions: self.positions(), state: Phase::Change, gpe: self.gpe() });
                }
                // Pan.
                if any_moved {
                    let c = self.centre();
                    let t = self.translation();
                    let v = match self.pan.last {
                        Some((t0, c0)) if now > t0 => {
                            let dt = now.duration_since(t0).as_secs_f32().max(1e-3);
                            ((c.0 - c0.0) / dt, (c.1 - c0.1) / dt)
                        }
                        _ => self.pan.velocity,
                    };
                    self.pan.velocity = v;
                    self.pan.last = Some((now, c));
                    let state = if self.pan.on { Phase::Change } else { Phase::Begin };
                    self.pan.on = true;
                    self.gestured = true;
                    out.push(Gesture::Pan { positions: self.positions(), translation: t, velocity: v, state, gpe: self.gpe() });
                }
                // Pinch and rotation: the first two fingers.
                if self.fingers.len() >= 2 {
                    let (a, b) = (self.fingers[0].pos, self.fingers[1].pos);
                    let scale = dist(a, b) / self.pinch.d0.max(1.0);
                    if self.pinch.on || (scale - 1.0).abs() * self.pinch.d0 > mech::TOUCH_PAN_SLOP_PX {
                        let v = match self.pinch.last {
                            Some((t0, s0)) if now > t0 => (scale - s0) / now.duration_since(t0).as_secs_f32().max(1e-3),
                            _ => self.pinch.velocity,
                        };
                        self.pinch.velocity = v;
                        self.pinch.last = Some((now, scale));
                        let state = if self.pinch.on { Phase::Change } else { Phase::Begin };
                        self.pinch.on = true;
                        self.gestured = true;
                        out.push(Gesture::Pinch { positions: self.positions(), scale, velocity: v, state, gpe: self.gpe() });
                    }
                    let a1 = angle(a, b);
                    let mut d = a1 - self.rotate.prev;
                    while d > 180.0 {
                        d -= 360.0;
                    }
                    while d < -180.0 {
                        d += 360.0;
                    }
                    self.rotate.prev = a1;
                    self.rotate.total += d;
                    if self.rotate.on || self.rotate.total.abs() > ROTATE_SLOP_DEG {
                        let v = match self.rotate.last {
                            Some((t0, r0)) if now > t0 => (self.rotate.total - r0) / now.duration_since(t0).as_secs_f32().max(1e-3),
                            _ => self.rotate.velocity,
                        };
                        self.rotate.velocity = v;
                        self.rotate.last = Some((now, self.rotate.total));
                        let state = if self.rotate.on { Phase::Change } else { Phase::Begin };
                        self.rotate.on = true;
                        self.gestured = true;
                        out.push(Gesture::Rotate { positions: self.positions(), rotation: self.rotate.total, velocity: v, state, gpe: self.gpe() });
                    }
                }
            }
            TouchState::End | TouchState::Cancel => {
                let Some(i) = self.fingers.iter().position(|f| f.id == ev.id) else { return out };
                let mut f = self.fingers.remove(i);
                f.pos = (ev.x, ev.y);
                let positions_before: Vec<(f32, f32)> = std::iter::once(f.pos).chain(self.fingers.iter().map(|g| g.pos)).collect();
                let gpe = if i == 0 { f.sunk } else { self.gpe() };
                let count = (self.fingers.len() + 1) as u32;
                let lifted_clean = ev.state == TouchState::End;
                // A tap: quick, still, alone, no long press.
                if lifted_clean
                    && !f.moved
                    && !self.long_press
                    && !self.gestured
                    && self.fingers.is_empty()
                    && now.duration_since(f.since) <= Duration::from_millis(mech::TOUCH_TAP_MS)
                    && dist(f.pos, f.start) <= mech::TOUCH_TAP_SLOP_PX
                {
                    out.push(Gesture::Tap { positions: vec![f.pos], gpe: f.sunk });
                    out.push(Gesture::TapInWorld { pos: f.pos, processed_by_ui: f.sunk });
                }
                // The two-finger gestures end when the pair breaks.
                if count == 2 {
                    if self.pinch.on {
                        out.push(Gesture::Pinch {
                            positions: positions_before.clone(),
                            scale: dist(positions_before[0], positions_before[1]) / self.pinch.d0.max(1.0),
                            velocity: self.pinch.velocity,
                            state: Phase::End,
                            gpe,
                        });
                    }
                    if self.rotate.on {
                        out.push(Gesture::Rotate {
                            positions: positions_before.clone(),
                            rotation: self.rotate.total,
                            velocity: self.rotate.velocity,
                            state: Phase::End,
                            gpe,
                        });
                    }
                    self.pinch = Pinch::default();
                    self.rotate = Rotate::default();
                } else if count > 2 && i < 2 {
                    // One of the measured pair left: measure the next pair afresh.
                    let (a, b) = (self.fingers[0].pos, self.fingers[1].pos);
                    self.pinch = Pinch { on: self.pinch.on, d0: dist(a, b).max(1.0) / self.pinch_scale_now(a, b), last: None, velocity: 0.0 };
                    let a0 = angle(a, b);
                    self.rotate = Rotate { on: self.rotate.on, total: self.rotate.total, prev: a0, last: None, velocity: 0.0 };
                }
                if self.fingers.is_empty() {
                    if self.long_press {
                        out.push(Gesture::LongPress { positions: positions_before.clone(), state: Phase::End, gpe });
                        self.long_press = false;
                    }
                    if self.pan.on {
                        let c = f.pos;
                        let t = (self.pan.carried.0 + c.0 - self.pan.base.0, self.pan.carried.1 + c.1 - self.pan.base.1);
                        let v = Self::speed(&f, now);
                        out.push(Gesture::Pan { positions: positions_before.clone(), translation: t, velocity: v, state: Phase::End, gpe });
                        // A fast lift after a long enough travel is a swipe.
                        let speed = (v.0 * v.0 + v.1 * v.1).sqrt();
                        if lifted_clean && speed >= mech::TOUCH_SWIPE_MIN_PX_PER_S && dist(f.pos, f.start) >= mech::TOUCH_SWIPE_MIN_PX {
                            let direction = if v.0.abs() >= v.1.abs() {
                                if v.0 > 0.0 { Swipe::Right } else { Swipe::Left }
                            } else if v.1 > 0.0 {
                                Swipe::Down
                            } else {
                                Swipe::Up
                            };
                            out.push(Gesture::Swipe { direction, touches: count, gpe });
                        }
                        self.pan = Pan::default();
                    }
                    self.gestured = false;
                } else {
                    self.rebase_pan();
                }
            }
        }
        out
    }

    /// The pinch scale a new pair starts from (keeps `scale` continuous
    /// when the measured pair changes).
    fn pinch_scale_now(&self, a: (f32, f32), b: (f32, f32)) -> f32 {
        if self.pinch.on {
            let s = self.pinch.last.map(|(_, s)| s).unwrap_or(1.0);
            if s > 0.0 {
                return s;
            }
        }
        let _ = (a, b);
        1.0
    }

    /// Once a frame: a finger held still long enough begins a long press.
    pub fn tick(&mut self, now: Instant) -> Vec<Gesture> {
        let mut out = Vec::new();
        if !self.long_press && !self.fingers.is_empty() && !self.pan.on {
            let first = &self.fingers[0];
            if !first.moved
                && !self.fingers.iter().any(|f| f.moved)
                && now.duration_since(first.since) >= Duration::from_millis(mech::TOUCH_LONG_PRESS_GAME_MS)
            {
                self.long_press = true;
                out.push(Gesture::LongPress { positions: self.positions(), state: Phase::Begin, gpe: self.gpe() });
            }
        }
        out
    }

    /// Fingers down now.
    pub fn count(&self) -> usize {
        self.fingers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: u64, state: TouchState, x: f32, y: f32, sunk: bool) -> TouchOut {
        TouchOut { id, state, x, y, dx: 0.0, dy: 0.0, sunk, began_on: Vec::new() }
    }

    #[test]
    fn a_quick_still_finger_taps() {
        let mut g = Gestures::default();
        let t0 = Instant::now();
        assert!(g.feed(&ev(1, TouchState::Begin, 100.0, 100.0, false), t0).is_empty());
        let out = g.feed(&ev(1, TouchState::End, 103.0, 101.0, false), t0 + Duration::from_millis(120));
        assert_eq!(
            out,
            vec![Gesture::Tap { positions: vec![(103.0, 101.0)], gpe: false }, Gesture::TapInWorld { pos: (103.0, 101.0), processed_by_ui: false }]
        );
        // On the GUI: processed.
        g.feed(&ev(2, TouchState::Begin, 10.0, 10.0, true), t0);
        let out = g.feed(&ev(2, TouchState::End, 10.0, 10.0, true), t0 + Duration::from_millis(50));
        assert!(matches!(out[0], Gesture::Tap { gpe: true, .. }));
        assert!(matches!(out[1], Gesture::TapInWorld { processed_by_ui: true, .. }));
    }

    #[test]
    fn a_moving_finger_pans_and_a_fast_lift_swipes() {
        let mut g = Gestures::default();
        let t0 = Instant::now();
        g.feed(&ev(1, TouchState::Begin, 100.0, 100.0, false), t0);
        let a = g.feed(&ev(1, TouchState::Change, 130.0, 100.0, false), t0 + Duration::from_millis(20));
        assert!(matches!(a[0], Gesture::Pan { state: Phase::Begin, translation: (30.0, 0.0), .. }), "{a:?}");
        let b = g.feed(&ev(1, TouchState::Change, 160.0, 100.0, false), t0 + Duration::from_millis(40));
        assert!(matches!(b[0], Gesture::Pan { state: Phase::Change, translation: (60.0, 0.0), .. }), "{b:?}");
        let c = g.feed(&ev(1, TouchState::End, 190.0, 100.0, false), t0 + Duration::from_millis(60));
        assert!(matches!(c[0], Gesture::Pan { state: Phase::End, translation: (90.0, 0.0), .. }), "{c:?}");
        assert!(matches!(c[1], Gesture::Swipe { direction: Swipe::Right, touches: 1, gpe: false }), "{c:?}");
        assert!(!c.iter().any(|g| matches!(g, Gesture::Tap { .. })));
    }

    #[test]
    fn a_slow_pan_is_no_swipe_and_no_tap() {
        let mut g = Gestures::default();
        let t0 = Instant::now();
        g.feed(&ev(1, TouchState::Begin, 100.0, 100.0, false), t0);
        g.feed(&ev(1, TouchState::Change, 100.0, 140.0, false), t0 + Duration::from_millis(300));
        let c = g.feed(&ev(1, TouchState::End, 100.0, 160.0, false), t0 + Duration::from_millis(600));
        assert_eq!(c.len(), 1, "{c:?}");
        assert!(matches!(c[0], Gesture::Pan { state: Phase::End, .. }));
    }

    #[test]
    fn two_fingers_pinch_and_rotate() {
        let mut g = Gestures::default();
        let t0 = Instant::now();
        g.feed(&ev(1, TouchState::Begin, 100.0, 100.0, false), t0);
        g.feed(&ev(2, TouchState::Begin, 200.0, 100.0, false), t0);
        // Spread: 100 → 200 px.
        let a = g.feed(&ev(2, TouchState::Change, 300.0, 100.0, false), t0 + Duration::from_millis(50));
        let pinch = a.iter().find(|g| matches!(g, Gesture::Pinch { .. })).unwrap();
        assert!(matches!(pinch, Gesture::Pinch { state: Phase::Begin, scale, .. } if (*scale - 2.0).abs() < 1e-3), "{a:?}");
        assert!(!a.iter().any(|g| matches!(g, Gesture::Rotate { .. })), "{a:?}");
        // Turn the second finger a quarter around the first.
        let b = g.feed(&ev(2, TouchState::Change, 100.0, 300.0, false), t0 + Duration::from_millis(100));
        let rot = b.iter().find(|g| matches!(g, Gesture::Rotate { .. })).unwrap();
        assert!(matches!(rot, Gesture::Rotate { state: Phase::Begin, rotation, .. } if (*rotation - 90.0).abs() < 1e-2), "{b:?}");
        let c = g.feed(&ev(2, TouchState::End, 100.0, 300.0, false), t0 + Duration::from_millis(150));
        assert!(c.iter().any(|g| matches!(g, Gesture::Pinch { state: Phase::End, .. })), "{c:?}");
        assert!(c.iter().any(|g| matches!(g, Gesture::Rotate { state: Phase::End, .. })), "{c:?}");
        assert!(!c.iter().any(|g| matches!(g, Gesture::Tap { .. })), "{c:?}");
        let d = g.feed(&ev(1, TouchState::End, 100.0, 100.0, false), t0 + Duration::from_millis(200));
        // The first finger never moved, but it was part of a two-finger
        // gesture: no tap.
        assert!(!d.iter().any(|g| matches!(g, Gesture::Tap { .. })), "{d:?}");
        assert_eq!(g.count(), 0);
    }

    #[test]
    fn a_still_finger_long_presses() {
        let mut g = Gestures::default();
        let t0 = Instant::now();
        g.feed(&ev(1, TouchState::Begin, 50.0, 50.0, false), t0);
        assert!(g.tick(t0 + Duration::from_millis(100)).is_empty());
        let a = g.tick(t0 + Duration::from_millis(mech::TOUCH_LONG_PRESS_GAME_MS + 10));
        assert_eq!(a, vec![Gesture::LongPress { positions: vec![(50.0, 50.0)], state: Phase::Begin, gpe: false }]);
        assert!(g.tick(t0 + Duration::from_millis(900)).is_empty());
        let b = g.feed(&ev(1, TouchState::Change, 52.0, 50.0, false), t0 + Duration::from_millis(950));
        assert!(matches!(b[0], Gesture::LongPress { state: Phase::Change, .. }), "{b:?}");
        let c = g.feed(&ev(1, TouchState::End, 52.0, 50.0, false), t0 + Duration::from_millis(1000));
        assert!(matches!(c[0], Gesture::LongPress { state: Phase::End, .. }), "{c:?}");
        assert!(!c.iter().any(|g| matches!(g, Gesture::Tap { .. })));
        assert_eq!(g.count(), 0);
    }
}
