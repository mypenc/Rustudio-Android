//! Touch-friendly egui modal dialog displayed on startup when a crash from
//! the previous session was detected.
//!
//! Allows the user to inspect a scrollable monospace preview of the crash
//! report, export it as a `.txt` file (Android SAF `ACTION_CREATE_DOCUMENT`
//! via `crate::android::crash_export::export_crash_txt_saf`, or `rfd` save
//! dialog on desktop), copy the full text to the clipboard, or dismiss &
//! archive the crash report.

use crate::crash::{self, CrashReport};

#[cfg(not(target_os = "android"))]
use std::sync::mpsc::{self, Receiver, TryRecvError};

/// Banner state displayed inside the crash dialog after an action.
#[derive(Clone, Debug)]
enum StatusBanner {
    Success(String),
    Info(String),
    Error(String),
}

/// Persistent UI state for the "Previous Session Crashed" modal dialog.
pub struct CrashDialogState {
    report: Option<CrashReport>,
    open: bool,
    status: Option<StatusBanner>,
    #[cfg(not(target_os = "android"))]
    desktop_export_rx: Option<Receiver<Result<String, Option<String>>>>,
}

impl Default for CrashDialogState {
    fn default() -> Self {
        Self::from_pending_crash()
    }
}

impl CrashDialogState {
    /// Checks for a pending crash from the previous session and opens the
    /// modal automatically if one exists.
    pub fn from_pending_crash() -> Self {
        let report = crash::take_pending_crash();
        let open = report.is_some();
        Self {
            report,
            open,
            status: None,
            #[cfg(not(target_os = "android"))]
            desktop_export_rx: None,
        }
    }

    /// Creates a closed dialog without checking disk (useful for headless/tests).
    pub fn empty() -> Self {
        Self {
            report: None,
            open: false,
            status: None,
            #[cfg(not(target_os = "android"))]
            desktop_export_rx: None,
        }
    }

    /// Manually opens the dialog with a specific crash report.
    pub fn open_report(&mut self, report: CrashReport) {
        self.report = Some(report);
        self.open = true;
        self.status = None;
    }

    /// Returns whether the crash dialog is currently open.
    pub fn is_open(&self) -> bool {
        self.open && self.report.is_some()
    }

    /// Renders the crash dialog modal if open. Call once per frame in `RustudioApp::update`.
    pub fn show(&mut self, ctx: &egui::Context) {
        self.poll_export_feedback();

        if !self.open {
            return;
        }
        let Some(report) = self.report.clone() else {
            self.open = false;
            return;
        };

        let screen = ctx.content_rect();
        let modal_w = (screen.width() - 32.0).clamp(320.0, 680.0);
        let modal_h = (screen.height() - 48.0).clamp(300.0, 560.0);

        let mut dismiss_clicked = false;
        let mut export_clicked = false;
        let mut copy_clicked = false;

        egui::Window::new("Previous Session Crashed")
            .id(egui::Id::new("rustudio_crash_dialog"))
            .collapsible(false)
            .resizable(true)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .default_size(egui::vec2(modal_w, modal_h))
            .max_size(egui::vec2(screen.width() - 16.0, screen.height() - 24.0))
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(10.0, 8.0);
                ui.spacing_mut().button_padding = egui::vec2(14.0, 9.0);

                // Crash type & timestamp header
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new(&report.title)
                            .strong()
                            .size(15.0)
                            .color(egui::Color32::from_rgb(248, 113, 113)),
                    );
                    ui.label(
                        egui::RichText::new(format!("•  {}", report.timestamp))
                            .weak()
                            .size(13.0),
                    );
                });

                if !report.summary.is_empty() {
                    ui.label(
                        egui::RichText::new(&report.summary)
                            .size(13.0)
                            .color(egui::Color32::from_rgb(226, 232, 240)),
                    );
                }

                // Status feedback banner (export success / clipboard / error)
                if let Some(banner) = &self.status {
                    let (bg, fg, text) = match banner {
                        StatusBanner::Success(m) => (
                            egui::Color32::from_rgb(20, 83, 45),
                            egui::Color32::from_rgb(187, 247, 208),
                            m.as_str(),
                        ),
                        StatusBanner::Info(m) => (
                            egui::Color32::from_rgb(30, 58, 138),
                            egui::Color32::from_rgb(191, 219, 254),
                            m.as_str(),
                        ),
                        StatusBanner::Error(m) => (
                            egui::Color32::from_rgb(127, 29, 29),
                            egui::Color32::from_rgb(254, 202, 202),
                            m.as_str(),
                        ),
                    };
                    egui::Frame::NONE
                        .fill(bg)
                        .corner_radius(egui::CornerRadius::same(6))
                        .inner_margin(egui::Margin::symmetric(10, 6))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(egui::RichText::new(text).color(fg).size(12.5));
                        });
                }

                ui.separator();

                // Scrollable monospace preview of the crash report
                let preview_height = (ui.available_height() - 56.0).max(140.0);
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(18, 20, 24))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(55, 62, 74)))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        egui::ScrollArea::both()
                            .id_salt("crash_report_preview_scroll")
                            .max_height(preview_height)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&report.full_text)
                                            .monospace()
                                            .size(11.5)
                                            .color(egui::Color32::from_rgb(209, 213, 219)),
                                    )
                                    .selectable(true)
                                    .wrap_mode(egui::TextWrapMode::Extend),
                                );
                            });
                    });

                ui.add_space(4.0);

                // Touch-friendly action bar
                ui.horizontal_wrapped(|ui| {
                    let primary = egui::Button::new(
                        egui::RichText::new("Export Crash Report (.txt)")
                            .strong()
                            .color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(37, 99, 235))
                    .min_size(egui::vec2(190.0, 38.0));

                    if ui.add(primary).clicked() {
                        export_clicked = true;
                    }

                    let copy_btn =
                        egui::Button::new("Copy Report").min_size(egui::vec2(115.0, 38.0));
                    if ui.add(copy_btn).clicked() {
                        copy_clicked = true;
                    }

                    let dismiss_btn =
                        egui::Button::new("Dismiss").min_size(egui::vec2(96.0, 38.0));
                    if ui.add(dismiss_btn).clicked() {
                        dismiss_clicked = true;
                    }
                });
            });

        if export_clicked {
            self.trigger_export(ctx, &report);
        }
        if copy_clicked {
            ctx.copy_text(report.full_text.clone());
            #[cfg(target_os = "android")]
            crate::android::clipboard::set(report.full_text.clone());
            self.status = Some(StatusBanner::Success(
                "Copied full crash report to clipboard.".to_string(),
            ));
        }
        if dismiss_clicked {
            crash::dismiss_pending_crash();
            self.open = false;
            self.report = None;
            self.status = None;
        }
    }

    fn suggested_export_name(report: &CrashReport) -> String {
        let kind_tag = match report.kind {
            crash::CrashKind::Panic => "panic",
            crash::CrashKind::Signal => "signal",
        };
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
        format!("rustudio-{kind_tag}-crash-{ts}.txt")
    }

    fn trigger_export(&mut self, ctx: &egui::Context, report: &CrashReport) {
        let suggested_name = Self::suggested_export_name(report);

        #[cfg(target_os = "android")]
        {
            let _ = ctx;
            match crate::android::crash_export::export_crash_txt_saf(
                &suggested_name,
                report.full_text.clone(),
            ) {
                crate::android::crash_export::ExportStatus::Success(msg) => {
                    self.status = Some(StatusBanner::Success(msg));
                }
                crate::android::crash_export::ExportStatus::Cancelled => {
                    self.status = Some(StatusBanner::Info("Export cancelled.".to_string()));
                }
                crate::android::crash_export::ExportStatus::Error(err) => {
                    self.status = Some(StatusBanner::Error(err));
                }
            }
        }

        #[cfg(not(target_os = "android"))]
        {
            let dialog = rfd::AsyncFileDialog::new()
                .set_title("Export Crash Report")
                .set_file_name(&suggested_name)
                .add_filter("Text files", &["txt"]);
            let content = report.full_text.clone();
            let (tx, rx) = mpsc::channel();
            let repaint_ctx = ctx.clone();

            let spawned = rt_thread::Builder::new()
                .name("crash-report-export".into())
                .spawn(move || {
                    let result = match pollster::block_on(dialog.save_file()) {
                        Some(handle) => {
                            let path = crate::file_dialog::handle_path(&handle);
                            match std::fs::write(&path, content.as_bytes()) {
                                Ok(()) => Ok(format!("Saved crash report to {}", path.display())),
                                Err(e) => Err(Some(format!(
                                    "Failed to write {}: {e}",
                                    path.display()
                                ))),
                            }
                        }
                        None => Err(None),
                    };
                    let _ = tx.send(result);
                    repaint_ctx.request_repaint();
                });

            match spawned {
                Ok(_) => {
                    self.desktop_export_rx = Some(rx);
                    self.status = Some(StatusBanner::Info(
                        "Choose a location to save the crash report...".to_string(),
                    ));
                }
                Err(e) => {
                    self.status = Some(StatusBanner::Error(format!(
                        "Could not open save dialog: {e}"
                    )));
                }
            }
        }
    }

    fn poll_export_feedback(&mut self) {
        #[cfg(target_os = "android")]
        {
            if let Some(st) = crate::android::crash_export::poll_export_status() {
                self.status = Some(match st {
                    crate::android::crash_export::ExportStatus::Success(m) => {
                        StatusBanner::Success(m)
                    }
                    crate::android::crash_export::ExportStatus::Cancelled => {
                        StatusBanner::Info("Export cancelled.".to_string())
                    }
                    crate::android::crash_export::ExportStatus::Error(e) => {
                        StatusBanner::Error(e)
                    }
                });
            }
        }

        #[cfg(not(target_os = "android"))]
        {
            let Some(rx) = &self.desktop_export_rx else {
                return;
            };
            match rx.try_recv() {
                Ok(Ok(msg)) => {
                    self.status = Some(StatusBanner::Success(msg));
                    self.desktop_export_rx = None;
                }
                Ok(Err(Some(err))) => {
                    self.status = Some(StatusBanner::Error(err));
                    self.desktop_export_rx = None;
                }
                Ok(Err(None)) => {
                    self.status = Some(StatusBanner::Info("Export cancelled.".to_string()));
                    self.desktop_export_rx = None;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.desktop_export_rx = None;
                }
            }
        }
    }
}
