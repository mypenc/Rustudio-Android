//! A cancellable Team Create connection for the editor.
//!
//! Call `LiveSession::connect` only in response to the user's live-open action.
//! The four embedded Studio 0.746 templates contain flag names/build constants,
//! never player JSON, tickets, cookies, or a captured shared-string cache.
//!
//! `Ready` waits for the initial place data's end marker and a 1.5-second
//! settle period for follow-up instances. Updates continue after `Ready`;
//! callers must continue draining `events`.

use crate::error::{NetError, Result};
#[cfg(not(target_arch = "wasm32"))]
use crate::handshake::{self, ProtocolSyncTemplate, TicketTemplate};
#[cfg(not(target_arch = "wasm32"))]
use crate::matchmaker::{LiveJoinConsent, Matchmaker, PollBudget};
use crate::replicator::{packet_id, Decoder, PropValue, RepOp, Schema};
#[cfg(not(target_arch = "wasm32"))]
use crate::rtcio::{self, lane_tag, CH_CONTROL0, CH_PING, CH_REPLICATION, TAG_CTRL};
#[cfg(not(target_arch = "wasm32"))]
use crate::sessioncrypto::Hello;
#[cfg(not(target_arch = "wasm32"))]
use crate::transport::{connect_quic, QuicTarget};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use web_time::Instant;

const REP_CONTROL: [u8; 7] = [0x83, 0x23, 0, 0, 0, 0, 0];
// Studio grants an initial instance quota when the streaming join completes.
// A smaller radius on later quota reports limits the incremental stream.
const INITIAL_STREAM_QUOTA: i32 = 10_000;
const EDITOR_STREAM_REGION_RADIUS: i64 = 65_536;

fn stream_credit(instances: i32, radius: i16) -> [u8; 10] {
    let mut frame = [0x83, 0x1c, 0, 0, 0, 0, 0, 0, 0, 0];
    frame[3..7].copy_from_slice(&instances.to_le_bytes());
    frame[7..9].copy_from_slice(&radius.to_le_bytes());
    frame
}

fn initial_stream_request(schema: &Schema, player: u64) -> std::result::Result<Vec<u8>, String> {
    let mut frame = crate::writer::property(
        schema, player, "Player", "ChararacterRegionId", &PropValue::Vector3([0.0; 3]),
    )?;
    let radius = PropValue::Int(EDITOR_STREAM_REGION_RADIUS);
    let override_radii = crate::writer::event(
        schema, player, "Player", "OverrideStreamRadii", &[radius.clone(), radius],
    )?;
    frame.pop();
    frame.extend_from_slice(&stream_credit(INITIAL_STREAM_QUOTA, i16::MAX)[1..9]);
    frame.extend_from_slice(&override_radii[1..]);
    Ok(frame)
}

const BATCH_SIZE: usize = 512;
const MAX_FRAME_SIZE: usize = 64 << 20;
const QUIET_PERIOD: Duration = Duration::from_millis(1500);
/// Edits written to the replication stream per 20 ms tick, at most: a
/// plugin's burst (hundreds of creates, megabytes of Source) goes out over
/// several ticks, so pings and replication control keep their turn.
const EDITS_PER_TICK: usize = 64;
const BYTES_PER_TICK: usize = 1 << 20;
/// Written edits kept for the log line a server-ended session prints.
const RECENT_EDITS: usize = 32;

#[derive(Debug)]
pub enum LiveEvent {
    /// Join progress, for the joining dialog: a step starting.
    Stage(JoinStage),
    /// The matchmaker is still preparing the server: (attempt, its message).
    Waiting(u32, String),
    Batch(Vec<RepOp>),
    Ready,
    Identity(u64),
    Notice(String),
    /// A persistent document edit was not transmitted; saving must stay blocked.
    EditRejected(String),
    /// The edit of instance `id` (its referent as sent) was not transmitted:
    /// it could not be encoded, or it depends on an instance that was not
    /// created. Saving stays blocked, as for `EditRejected`.
    EditFailed { id: u64, error: String },
    /// Instance `id` was created without property `name`, which this server
    /// cannot take from rustudio (`reason`). The rest of it was sent.
    PropertySkipped { id: u64, name: String, reason: String },
    Error(String),
    Disconnected(String),
}

/// The steps of a join, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum JoinStage {
    SignIn,
    Matchmaking,
    Connecting,
    Handshake,
    Receiving,
}

impl JoinStage {
    pub const ALL: [JoinStage; 5] = [
        JoinStage::SignIn,
        JoinStage::Matchmaking,
        JoinStage::Connecting,
        JoinStage::Handshake,
        JoinStage::Receiving,
    ];
    pub fn label(self) -> &'static str {
        match self {
            JoinStage::SignIn => "Checking your sign-in",
            JoinStage::Matchmaking => "Asking Roblox for the Team Create server",
            JoinStage::Connecting => "Connecting to the server",
            JoinStage::Handshake => "Joining the session",
            JoinStage::Receiving => "Downloading the place",
        }
    }
}

/// Owns the lifetime of the live connection. Dropping it cancels the
/// worker; drop never joins a network thread or blocks the UI. An already
/// running synchronous account/matchmaker request can finish before noticing
/// cancellation, but cancellation prevents subsequent connection work.
enum LiveEdit {
    Event {id: u64, class: String, name: String, args: Vec<PropValue>},
    Property {
        id: u64,
        class: String,
        name: String,
        value: PropValue,
    },
    Create {
        id: u64,
        class: String,
        parent: u64,
        props: Vec<(String, PropValue)>,
    },
    Delete(u64),
    Reparent {
        id: u64,
        parent: u64,
    },
}

impl LiveEdit {
    /// One line for the log: kind, class, referent and property names.
    fn describe(&self) -> String {
        match self {
            LiveEdit::Event { id, class, name, .. } => format!("event {class}.{name} on {id:#x}"),
            LiveEdit::Property { id, class, name, .. } => format!("set {class}.{name} on {id:#x}"),
            LiveEdit::Create { id, class, parent, props } => format!(
                "create {class} {id:#x} under {parent:#x} [{}]",
                props.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")
            ),
            LiveEdit::Delete(id) => format!("delete {id:#x}"),
            LiveEdit::Reparent { id, parent } => format!("move {id:#x} under {parent:#x}"),
        }
    }
}

pub struct LiveSession {
    pub events: mpsc::Receiver<LiveEvent>,
    cancel: Arc<AtomicBool>,
    commands: mpsc::Sender<LiveEdit>,
    next_id: Arc<AtomicU64>,
}

impl LiveSession {
    pub fn connect(universe: u64, place: u64) -> Result<Self> {
        Self::connect_as(universe, place, None)
    }

    /// [`LiveSession::connect`] as a saved account (account switching's
    /// `roblox-account-<id>` keyring entry) instead of the active session.
    pub fn connect_as(universe: u64, place: u64, account: Option<u64>) -> Result<Self> {
        if universe == 0 || place == 0 {
            return Err(NetError::Other("A universe and place are required".into()));
        }
        // Bound pending model work: a large place cannot enqueue its entire
        // DataModel while the editor is applying a previous batch.
        let (tx, events) = mpsc::sync_channel(8);
        // Unbounded: every queued edit must reach the worker in order (a
        // dropped one leaves the server half-edited); the worker paces them.
        let (commands, command_rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let next_id = Arc::new(AtomicU64::new(0));
        let worker_id = next_id.clone();
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (&tx, &worker_cancel, &command_rx, &worker_id);
            return Err(NetError::Other("Team Create needs sockets, which the browser build does not have".into()));
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::Builder::new()
            .name("rustudio-team-create".into())
            .spawn(move || {
                log::info!("Team Create join started (place={place}, universe={universe})");
                let result = live_net::prepare_and_run(
                    universe,
                    place,
                    account,
                    &tx,
                    worker_cancel.clone(),
                    command_rx,
                    worker_id,
                );
                if let Err(error) = result {
                    if !worker_cancel.load(Ordering::Acquire) {
                        // Errors created by this module contain fixed context,
                        // never response bodies or secret-bearing packet bytes.
                        log::error!("Team Create worker failed (place={place}): {error}");
                        let _ = tx.send(LiveEvent::Error(error.to_string()));
                    }
                }
            })
            .map_err(|_| NetError::Other("Could not start Team Create worker".into()))?;
        Ok(Self {
            events,
            cancel,
            commands,
            next_id,
        })
    }
    pub fn send_property(
        &self,
        id: u64,
        class: String,
        name: String,
        value: PropValue,
    ) -> Result<()> {
        self.send_edit(LiveEdit::Property {
            id,
            class,
            name,
            value,
        })
    }
    pub fn send_event(&self, id: u64, class: String, name: String, args: Vec<PropValue>) -> Result<()> {
        self.send_edit(LiveEdit::Event {id,class,name,args})
    }
    pub fn allocate_id(&self) -> Result<u64> {
        self.next_id
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |id| {
                (id != 0 && id as u32 != u32::MAX).then_some(id + 1)
            })
            .map_err(|_| NetError::Other("Team Create instance namespace is unavailable".into()))
    }
    fn send_edit(&self, edit: LiveEdit) -> Result<()> {
        self.commands
            .send(edit)
            .map_err(|_| NetError::Other("Team Create edit queue is unavailable".into()))
    }
    pub fn send_create(
        &self,
        id: u64,
        class: String,
        parent: u64,
        props: Vec<(String, PropValue)>,
    ) -> Result<()> {
        self.send_edit(LiveEdit::Create {
            id,
            class,
            parent,
            props,
        })
    }
    pub fn send_delete(&self, id: u64) -> Result<()> {
        self.send_edit(LiveEdit::Delete(id))
    }
    pub fn send_reparent(&self, id: u64, parent: u64) -> Result<()> {
        self.send_edit(LiveEdit::Reparent { id, parent })
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod live_net;
