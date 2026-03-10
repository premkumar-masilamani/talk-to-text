import json
import logging
import os
import platform
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from transcriber.preprocessing.audio_preprocessor import (
    prepare_audio_for_transcription,
)
from transcriber.transcription.transcriber import Transcriber
from transcriber.utils.constants import (
    FFMPEG_PATH,
    WHISPER_CPP_LOCAL_BIN,
    WHISPER_CPP_LOCAL_LEGACY_BIN,
)
from transcriber.utils.file_util import (
    is_preprocessed_whisper_audio,
    supported_file_extensions,
    transcript_path_for_audio,
)
from transcriber.utils.hardware_profile import detect_hardware_profile
from transcriber.utils.model_selection import (
    min_ram_for_model,
    select_model_for_hardware,
)

logger = logging.getLogger(__name__)


class LogEmitter(QtCore.QObject):
    message = QtCore.Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.emitter.message.emit(msg)


class AnimatedProgressBar(QtWidgets.QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._animated = False
        self._stripe_offset = 0

    def setAnimated(self, animated: bool):
        self._animated = animated
        self.update()

    def advancePattern(self):
        if not self._animated:
            return
        self._stripe_offset = (self._stripe_offset + 2) % 18
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._animated:
            return

        min_val = self.minimum()
        max_val = self.maximum()
        if max_val <= min_val:
            return

        ratio = (self.value() - min_val) / (max_val - min_val)
        if ratio <= 0:
            return

        inner = self.rect().adjusted(2, 2, -2, -2)
        chunk_width = int(inner.width() * ratio)
        if chunk_width <= 0:
            return

        chunk = QtCore.QRect(inner.left(), inner.top(), chunk_width, inner.height())
        painter = QtGui.QPainter(self)
        painter.setClipRect(chunk)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 85), 2)
        painter.setPen(pen)
        spacing = 12
        x = chunk.left() - chunk.height() + self._stripe_offset
        while x < chunk.right() + chunk.height():
            painter.drawLine(
                x,
                chunk.bottom(),
                x + chunk.height(),
                chunk.top(),
            )
            x += spacing


@dataclass
class ItemState:
    path: Path
    progress: int = 0
    status: str = "Queued"


def txt_output_path_for_source(file: Path) -> Path:
    if is_preprocessed_whisper_audio(file):
        base_stem = file.stem[: -len(".whisper")]
        return file.with_name(f"{base_stem}.txt")
    return file.with_suffix(".txt")


def transcript_is_complete_for_source(file: Path) -> bool:
    txt_path = txt_output_path_for_source(file)
    if txt_path.exists():
        return True
    # Backward compatibility with previously saved naming.
    return transcript_path_for_audio(file).exists()


class DropArea(QtWidgets.QFrame):
    pathsDropped = QtCore.Signal(list)
    browseRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropArea")
        self.setProperty("dragActive", False)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setAccessibleName("File Upload Area")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)

        icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(icon.pixmap(36, 36))
        icon_label.setAlignment(QtCore.Qt.AlignCenter)

        self.title_label = QtWidgets.QLabel("Drop audio or video files here")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setObjectName("dropTitle")

        sub_label = QtWidgets.QLabel("or click to choose files")
        sub_label.setAlignment(QtCore.Qt.AlignCenter)
        sub_label.setObjectName("dropSubtitle")

        self.browse_btn = QtWidgets.QPushButton("Choose Files")
        self.browse_btn.setObjectName("secondaryButton")
        self.browse_btn.clicked.connect(self.browseRequested.emit)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.browse_btn)
        button_row.addStretch(1)

        layout.addWidget(icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(sub_label)
        layout.addLayout(button_row)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.browseRequested.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.setProperty("dragActive", True)
            self.style().unpolish(self)
            self.style().polish(self)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        urls = event.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        if paths:
            self.pathsDropped.emit(paths)
        event.acceptProposedAction()


class Worker(QtCore.QThread):
    itemStatus = QtCore.Signal(Path, int, str, int, int)
    itemDone = QtCore.Signal(Path)
    allDone = QtCore.Signal()

    def __init__(
            self,
            items: list[Path],
            language: str,
            include_timestamps: bool,
            parent=None,
    ):
        super().__init__(parent)
        self.items = items
        self.language = language
        self.include_timestamps = include_timestamps
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _save_txt(self, transcript_path: Path, text_content: str):
        with open(transcript_path, "w", encoding="utf-8") as handle:
            handle.write(text_content)

    def _save_transcript_output(self, transcript_path: Path, transcribed_json: str):
        payload = json.loads(transcribed_json)
        raw_transcript = str(payload.get("raw_transcript", "")).strip()
        text_content = raw_transcript
        self._save_txt(transcript_path, text_content)

    def run(self):
        transcriber = Transcriber()
        total = len(self.items)
        logger.info("Worker started. Items queued: %s", total)

        for index, path in enumerate(self.items, start=1):
            logger.info("Processing item %s/%s: %s", index, total, path)
            if self._stop_event.is_set():
                self.itemStatus.emit(path, 0, "Canceled", index, total)
                self.itemDone.emit(path)
                logger.info("Item canceled before start: %s", path)
                continue

            transcript_path = txt_output_path_for_source(path)
            try:
                if transcript_is_complete_for_source(path):
                    self.itemStatus.emit(
                        path, 100, "Skipped (transcript exists)", index, total
                    )
                    logger.info(
                        "Skipping transcription, transcript exists: %s", transcript_path
                    )
                    continue

                self.itemStatus.emit(path, 20, "Preprocessing", index, total)
                logger.info("Preprocessing started: %s", path)
                processed = prepare_audio_for_transcription(
                    path, stop_event=self._stop_event
                )
                logger.info("Preprocessing ready: %s -> %s", path, processed)

                self.itemStatus.emit(path, 65, "Transcribing", index, total)
                logger.info("Transcription started: %s", processed)
                transcribed_json = transcriber.transcribe(
                    processed, stop_event=self._stop_event
                )
                if transcribed_json:
                    self.itemStatus.emit(path, 90, "Saving", index, total)
                    self._save_transcript_output(transcript_path, transcribed_json)
                    logger.info("Transcript saved: %s", transcript_path)
                    self.itemStatus.emit(path, 100, "Done", index, total)
                else:
                    self.itemStatus.emit(path, 100, "Error", index, total)
                    logger.warning("Transcription produced no output: %s", path)
            except InterruptedError:
                self.itemStatus.emit(path, 0, "Canceled", index, total)
                logger.info("Canceled: %s", path)
            except Exception as exc:
                self.itemStatus.emit(path, 100, f"Error: {exc}", index, total)
                logger.exception("Unhandled processing error for %s", path)
            finally:
                self.itemDone.emit(path)

        logger.info("Worker finished all queued items.")
        self.allDone.emit()


class SetupWorker(QtCore.QThread):
    statusUpdate = QtCore.Signal(str, str, str)
    setupDone = QtCore.Signal(bool, str)

    def run(self):
        try:
            transcriber = Transcriber(progress_cb=self._emit_progress)
            if not transcriber.binary_path:
                self.setupDone.emit(False, "Transcriber engine is unavailable.")
                return
            if not transcriber.model_path.is_file():
                self.setupDone.emit(False, "Language model is unavailable.")
                return
            self.setupDone.emit(True, "Setup complete.")
        except Exception as exc:
            self.setupDone.emit(False, str(exc))

    def _emit_progress(self, item_id: str, status: str, path_text: str):
        self.statusUpdate.emit(item_id, status, path_text)


class TranscriberWindow(QtWidgets.QMainWindow):
    QUEUE_HEADERS = ("File", "Status", "Details")
    PLACEHOLDER_ROLE = QtCore.Qt.UserRole + 1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Talk to Text")
        self.resize(960, 640)
        self.setMinimumSize(920, 600)
        self.hardware_profile = None
        self.model_spec = select_model_for_hardware(None)

        self.items: dict[Path, ItemState] = {}
        self.rows: dict[Path, QtWidgets.QTreeWidgetItem] = {}
        self.row_progress: dict[Path, QtWidgets.QProgressBar] = {}
        self.row_visual_progress: dict[Path, int] = {}

        self.worker: Worker | None = None
        self.total_items = 0
        self.completed_items = 0
        self._processing = False

        self._log_emitter: LogEmitter | None = None
        self._log_handler: QtLogHandler | None = None
        self._log_messages: list[str] = []

        self._close_requested = False
        self.setup_worker: SetupWorker | None = None
        self.setup_in_progress = True

        self._pulse_timer = QtCore.QTimer(self)
        self._pulse_timer.setInterval(120)
        self._pulse_timer.timeout.connect(self._animate_progress)

        self._build_ui()
        self._setup_logging()
        QtCore.QTimer.singleShot(0, self._start_initial_setup)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        content = QtWidgets.QWidget()
        scroll.setWidget(content)

        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        root.setAlignment(QtCore.Qt.AlignTop)
        # 1) Header
        self.header_card = QtWidgets.QFrame()
        self.header_card.setObjectName("sectionCard")
        header_layout = QtWidgets.QVBoxLayout(self.header_card)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(4)

        title = QtWidgets.QLabel("Talk to Text")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Upload audio or video files and get accurate text automatically."
        )
        subtitle.setObjectName("subtitle")
        save_location_note = QtWidgets.QLabel(
            "Your notes will be saved next to the original file."
        )
        save_location_note.setObjectName("subtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(save_location_note)
        root.addWidget(self.header_card)

        self.copy_diagnostics_btn = QtWidgets.QToolButton(self.header_card)
        self.copy_diagnostics_btn.setObjectName("headerCopyButton")
        self.copy_diagnostics_btn.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
        )
        self.copy_diagnostics_btn.setToolTip("Copy Support Info")
        self.copy_diagnostics_btn.setAutoRaise(True)
        self.copy_diagnostics_btn.setFixedSize(22, 22)
        self.copy_diagnostics_btn.clicked.connect(self._copy_diagnostics_to_clipboard)
        self.header_card.installEventFilter(self)
        self._position_header_copy_button()

        # 2 + 3) Upload and Queue in one row
        upload_queue_row = QtWidgets.QHBoxLayout()
        upload_queue_row.setSpacing(14)

        self.drop_area = DropArea()
        self.drop_area.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.drop_area.pathsDropped.connect(self._add_paths)
        self.drop_area.browseRequested.connect(self._add_files)
        upload_queue_row.addWidget(self.drop_area, 1)

        self.queue_card = QtWidgets.QFrame()
        self.queue_card.setObjectName("sectionCard")
        self.queue_card.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        queue_layout = QtWidgets.QVBoxLayout(self.queue_card)
        queue_layout.setContentsMargins(16, 12, 16, 16)
        queue_layout.setSpacing(8)
        queue_title = QtWidgets.QLabel("Files to Process")
        queue_title.setObjectName("sectionTitle")
        queue_layout.addWidget(queue_title)
        self.list_widget = QtWidgets.QTreeWidget()
        self.list_widget.setColumnCount(3)
        self.list_widget.setHeaderLabels(list(self.QUEUE_HEADERS))
        self.list_widget.setRootIsDecorated(False)
        self.list_widget.setUniformRowHeights(True)
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.list_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.list_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list_widget.setAccessibleName("Files to Process")
        self.list_widget.itemClicked.connect(self._on_queue_row_clicked)
        header = self.list_widget.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header.setStretchLastSection(False)
        self.list_widget.setColumnWidth(1, 270)
        self.list_widget.setColumnWidth(2, 130)
        queue_layout.addWidget(self.list_widget)
        upload_queue_row.addWidget(self.queue_card, 3)
        root.addLayout(upload_queue_row)

        # 5) Primary Action Area
        action_card = QtWidgets.QFrame()
        action_layout = QtWidgets.QHBoxLayout(action_card)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.clear_btn = QtWidgets.QPushButton("Remove All")
        self.clear_btn.setObjectName("secondaryButton")
        action_layout.addWidget(self.clear_btn)
        action_layout.addStretch(1)
        self.start_btn = QtWidgets.QPushButton("Create Notes")
        self.start_btn.setObjectName("primaryButton")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setVisible(False)
        action_layout.addWidget(self.stop_btn)
        action_layout.addWidget(self.start_btn)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.clear_btn.clicked.connect(self._clear)
        root.addWidget(action_card)

        # 6) Progress Area
        progress_card = QtWidgets.QFrame()
        progress_card.setObjectName("sectionCard")
        progress_layout = QtWidgets.QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(16, 12, 16, 12)
        progress_layout.setSpacing(8)
        self.overall_label = QtWidgets.QLabel("No files added yet")
        self.progress_status_label = QtWidgets.QLabel("Add recordings to begin.")
        self.progress_status_label.setObjectName("statusMessage")
        self.overall_progress = AnimatedProgressBar()
        self.overall_progress.setRange(0, 100)
        progress_layout.addWidget(self.overall_label)
        progress_layout.addWidget(self.overall_progress)
        progress_layout.addWidget(self.progress_status_label)
        root.addWidget(progress_card)

        self._update_queue_height()
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget {
                color: #1f1f1f;
                font-size: 13px;
            }
            QMainWindow {
                background: #f3f5f8;
            }
            QFrame#sectionCard {
                background: #ffffff;
                border: 1px solid #dce2ea;
                border-radius: 12px;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: 700;
                color: #101828;
            }
            QLabel#subtitle {
                color: #475467;
                font-size: 14px;
            }
            QLabel#sectionTitle {
                font-size: 15px;
                font-weight: 600;
                color: #101828;
            }
            QLabel#statusMessage {
                color: #344054;
                font-weight: 500;
            }
            QLabel#dropTitle {
                font-size: 18px;
                font-weight: 600;
                color: #0f172a;
            }
            QLabel#dropSubtitle {
                color: #475467;
            }
            QFrame#dropArea {
                background: #ffffff;
                border: 2px dashed #b8c4d6;
                border-radius: 14px;
            }
            QFrame#dropArea[dragActive="true"] {
                border: 2px solid #1d4ed8;
                background: #eff6ff;
            }
            QPushButton {
                border: 1px solid #c6d0dc;
                background: #ffffff;
                color: #1f2937;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f8fafc;
            }
            QPushButton#primaryButton {
                background: #0f62fe;
                border: 1px solid #0f62fe;
                color: #ffffff;
            }
            QPushButton#primaryButton:hover {
                background: #0b53d6;
            }
            QPushButton#secondaryButton {
                background: #ffffff;
            }
            QPushButton#dangerButton {
                background: #d92d20;
                border: 1px solid #d92d20;
                color: #ffffff;
            }
            QPushButton#dangerButton:hover {
                background: #b42318;
                border: 1px solid #b42318;
            }
            QPushButton:disabled {
                background: #f1f5f9;
                color: #98a2b3;
                border: 1px solid #d0d5dd;
            }
            QToolButton {
                border: none;
                color: #0f172a;
                font-weight: 600;
                text-align: left;
                padding: 3px;
            }
            QToolButton#headerCopyButton {
                padding: 0px;
                border-radius: 6px;
            }
            QToolButton#headerCopyButton:hover {
                background: #eef2f7;
            }
            QComboBox, QLineEdit {
                border: 1px solid #d0d5dd;
                border-radius: 8px;
                padding: 6px 8px;
                background: #ffffff;
                min-height: 24px;
            }
            QTreeWidget {
                border: 1px solid #dce2ea;
                border-radius: 10px;
                background: #ffffff;
            }
            QTreeWidget::item {
                padding-top: 6px;
                padding-bottom: 6px;
            }
            QHeaderView::section {
                background: #f8fafc;
                border-top: none;
                border-left: none;
                border-right: 1px solid #e4e7ec;
                border-bottom: 1px solid #e4e7ec;
                padding: 8px;
                font-weight: 600;
                color: #344054;
            }
            QProgressBar {
                border: 1px solid #d0d5dd;
                border-radius: 8px;
                text-align: center;
                height: 14px;
                background: #ffffff;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #0f62fe,
                    stop: 1 #1d4ed8
                );
                border-radius: 8px;
            }
            """
        )

    def _friendly_status(self, raw: str) -> str:
        if raw.startswith("Error"):
            return "Error"
        if raw == "Queued":
            return "Waiting"
        if raw == "Preprocessing":
            return "Waiting in queue"
        if raw == "Processing":
            return "Transcribing"
        if raw == "Transcribing":
            return "Transcribing"
        if raw == "Saving":
            return "Finalizing"
        if raw == "Completed":
            return "Done"
        if raw == "Done":
            return "Done"
        if raw == "Skipped (transcript exists)":
            return "Done"
        if raw == "Canceled":
            return "Stopped"
        return raw

    def _is_complete_status(self, status: str) -> bool:
        return status in {"Done", "Completed", "Skipped (transcript exists)"}

    def _has_preconverted_wav(self, path: Path) -> bool:
        if path.suffix.lower() == ".wav":
            return True
        return path.with_suffix(".wav").exists()

    def _is_intermediate_wav(self, path: Path) -> bool:
        if path.suffix.lower() != ".wav":
            return False
        # Only canonical preprocessing artifact is treated as intermediate.
        return is_preprocessed_whisper_audio(path)

    def _initial_progress_and_status_for_path(self, path: Path) -> tuple[int, str]:
        transcript_exists = transcript_is_complete_for_source(path)
        if transcript_exists:
            return 100, "Skipped (transcript exists)"
        if self._has_preconverted_wav(path):
            return 50, "Queued"
        return 0, "Queued"

    def _status_color(self, friendly: str) -> str:
        if friendly == "Done":
            return "#137333"
        if friendly == "Error":
            return "#b42318"
        if friendly in {"Transcribing", "Finalizing"}:
            return "#0f62fe"
        return "#475467"

    def _setup_logging(self):
        self._log_emitter = LogEmitter()
        self._log_emitter.message.connect(self._append_log)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        self._log_handler = QtLogHandler(self._log_emitter)
        self._log_handler.setFormatter(formatter)

        root = logging.getLogger()
        requested_level = os.environ.get("TALKTOTEXT_LOG_LEVEL", "INFO").upper()
        resolved_level = getattr(logging, requested_level, logging.INFO)
        root.setLevel(resolved_level)
        root.addHandler(self._log_handler)

        has_stream = any(getattr(h, "_ui_stream_handler", False) for h in root.handlers)
        if not has_stream:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler._ui_stream_handler = True
            stream_handler.setFormatter(formatter)
            root.addHandler(stream_handler)

        logger.info(
            "UI initialized. Log level: %s", logging.getLevelName(resolved_level)
        )

    def _update_queue_height(self):
        row_hint = 34
        if self.list_widget.topLevelItemCount() > 0:
            row_hint = max(34, self.list_widget.sizeHintForRow(0))
        header_h = self.list_widget.header().height()
        visible_rows = 3
        table_h = header_h + (row_hint * visible_rows) + 8
        self.list_widget.setMinimumHeight(table_h)
        self.list_widget.setMaximumHeight(table_h)
        card_h = table_h + 52
        if hasattr(self, "queue_card"):
            self.queue_card.setMinimumHeight(card_h)
            self.queue_card.setMaximumHeight(card_h)
        if hasattr(self, "drop_area"):
            self.drop_area.setMinimumHeight(card_h)
            self.drop_area.setMaximumHeight(card_h)

    def _position_header_copy_button(self):
        if not hasattr(self, "header_card") or not hasattr(
                self, "copy_diagnostics_btn"
        ):
            return
        x = self.header_card.width() - self.copy_diagnostics_btn.width() - 10
        y = 10
        self.copy_diagnostics_btn.move(max(0, x), y)

    def eventFilter(self, obj, event):
        if (
                obj is getattr(self, "header_card", None)
                and event.type() == QtCore.QEvent.Resize
        ):
            self._position_header_copy_button()
        return super().eventFilter(obj, event)

    def _clear_placeholder_rows(self):
        for row_idx in range(self.list_widget.topLevelItemCount() - 1, -1, -1):
            row = self.list_widget.topLevelItem(row_idx)
            if row.data(0, self.PLACEHOLDER_ROLE):
                self.list_widget.takeTopLevelItem(row_idx)

    def _add_placeholder_rows(self):
        if self.items:
            return
        self._clear_placeholder_rows()
        for _ in range(3):
            row = QtWidgets.QTreeWidgetItem(["", "", ""])
            row.setFlags(QtCore.Qt.NoItemFlags)
            row.setData(0, self.PLACEHOLDER_ROLE, True)
            self._apply_row_padding(row)
            self.list_widget.addTopLevelItem(row)

    def _system_profile_rows(self) -> list[tuple[str, str]]:
        self._ensure_hardware_profile()
        min_ram_gb = min_ram_for_model(self.model_spec.model_id)
        return [
            ("OS", self.hardware_profile.system),
            ("Architecture", self.hardware_profile.architecture),
            ("RAM", f"{self.hardware_profile.ram_gb} GB"),
            ("CPU Cores", str(self.hardware_profile.cpu_cores)),
            ("Accelerator", self.hardware_profile.accelerator),
            ("Whisper Model", self.model_spec.model_id),
            ("Model Minimum RAM", f"{min_ram_gb} GB"),
            ("Whisper CLI", self._resolve_whisper_binary_for_profile()),
            ("ffmpeg", self._resolve_ffmpeg_for_profile()),
        ]

    def _ensure_hardware_profile(self):
        if self.hardware_profile is not None:
            return
        self.hardware_profile = detect_hardware_profile()
        self.model_spec = select_model_for_hardware(self.hardware_profile)

    def _copy_diagnostics_to_clipboard(self):
        system_lines = [
            f"{field}: {value}" for field, value in self._system_profile_rows()
        ]
        payload = [
            "System Profile",
            *system_lines,
            "",
            "Processing Log",
            *(self._log_messages if self._log_messages else ["No logs captured yet."]),
        ]
        QtWidgets.QApplication.clipboard().setText("\n".join(payload))
        self._show_copied_tooltip(self.copy_diagnostics_btn)

    def _show_copied_tooltip(self, source_button: QtWidgets.QWidget):
        point = source_button.mapToGlobal(
            QtCore.QPoint(source_button.width() // 2, source_button.height() + 6)
        )
        QtWidgets.QToolTip.showText(
            point,
            "Copied Support Info",
            source_button,
            source_button.rect(),
            1800,
        )

    def _resolve_whisper_binary_for_profile(self) -> str:
        if WHISPER_CPP_LOCAL_BIN.is_file():
            return str(WHISPER_CPP_LOCAL_BIN)
        if WHISPER_CPP_LOCAL_LEGACY_BIN.is_file():
            return str(WHISPER_CPP_LOCAL_LEGACY_BIN)
        for binary_name in ("whisper-cli", "whisper-cli.exe", "main", "main.exe"):
            from_path = shutil.which(binary_name)
            if from_path:
                return from_path
        return "Unavailable"

    def _resolve_ffmpeg_for_profile(self) -> str:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return str(Path(system_ffmpeg).resolve())

        binary_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
        bundled = FFMPEG_PATH / binary_name
        if bundled.is_file():
            return str(bundled)

        return "Unavailable"

    def _start_initial_setup(self):
        self.setup_in_progress = True
        self._set_processing_state(False)
        self._set_controls_enabled(False)
        self.drop_area.setEnabled(False)
        self._set_progress_message("Preparing files...")
        logger.info(
            "Preparing first-run tools and model. Controls are disabled until setup completes."
        )

        self.setup_worker = SetupWorker()
        self.setup_worker.statusUpdate.connect(self._on_setup_status)
        self.setup_worker.setupDone.connect(self._on_setup_done)
        self.setup_worker.finished.connect(self._on_setup_worker_finished)
        self.setup_worker.start()

    @QtCore.Slot(str, str, str)
    def _on_setup_status(self, item_id: str, status: str, path_text: str):
        del path_text
        if item_id.startswith("model"):
            self._set_progress_message("Detecting language...")
        elif item_id.startswith("tool"):
            self._set_progress_message("Preparing files...")
        logger.info("Setup status: %s=%s", item_id, status)

    @QtCore.Slot(bool, str)
    def _on_setup_done(self, success: bool, message: str):
        if success:
            self.setup_in_progress = False
            self.drop_area.setEnabled(True)
            self._set_controls_enabled(True)
            self._set_progress_message("Add recordings to begin.")
            logger.info("First-run setup complete. You can now add files.")
            return

        self.setup_in_progress = True
        self._set_controls_enabled(False)
        self.drop_area.setEnabled(False)
        self._set_progress_message("Unable to prepare the app.")
        logger.error("Setup failed: %s", message)
        QtWidgets.QMessageBox.critical(
            self,
            "Setup failed",
            "The app could not finish setup. Please restart and try again.",
        )

    @QtCore.Slot()
    def _on_setup_worker_finished(self):
        if self.setup_worker:
            self.setup_worker.deleteLater()
            self.setup_worker = None

    @QtCore.Slot(str)
    def _append_log(self, message: str):
        self._log_messages.append(message)

    def _set_progress_message(self, text: str):
        self.progress_status_label.setText(text)

    def _add_files(self):
        if self.setup_in_progress:
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select audio files",
            self._default_browse_dir(),
        )
        if paths:
            self._add_paths([Path(p) for p in paths])

    def _default_browse_dir(self) -> str:
        home = Path.home()
        downloads = home / "Downloads"
        return str(downloads if downloads.is_dir() else home)

    def _add_paths(self, paths: list[Path]):
        if self.setup_in_progress:
            return

        added_files: list[Path] = []
        skipped_existing: list[Path] = []
        for path in paths:
            if path.is_dir():
                for file_path in path.rglob("*"):
                    if (
                            file_path.is_file()
                            and file_path.suffix.lower() in supported_file_extensions
                    ):
                        if self._is_intermediate_wav(file_path):
                            continue
                        if file_path in self.items:
                            skipped_existing.append(file_path)
                            continue
                        if self._add_item(file_path):
                            added_files.append(file_path)
            elif path.is_file() and path.suffix.lower() in supported_file_extensions:
                if self._is_intermediate_wav(path):
                    continue
                if path in self.items:
                    skipped_existing.append(path)
                    continue
                if self._add_item(path):
                    added_files.append(path)

        self._update_empty_state()

        if added_files or skipped_existing:
            added_unique = list(dict.fromkeys(added_files))
            skipped_unique = list(dict.fromkeys(skipped_existing))
            message_lines: list[str] = []
            if added_unique:
                message_lines.append(f"Added ({len(added_unique)}):")
                message_lines.extend(f"- {file.name}" for file in added_unique)
            if skipped_unique:
                if message_lines:
                    message_lines.append("")
                message_lines.append(f"Already in list ({len(skipped_unique)}):")
                message_lines.extend(f"- {file.name}" for file in skipped_unique)
            QtWidgets.QMessageBox.information(
                self, "Files added", "\n".join(message_lines)
            )
            return

        if paths:
            QtWidgets.QMessageBox.information(
                self,
                "No audio files",
                "Unable to process this file.\nPlease check the file format and try again.",
            )

    def _add_item(self, path: Path) -> int:
        if path in self.items:
            return 0
        self._clear_placeholder_rows()

        initial_progress, initial_status = self._initial_progress_and_status_for_path(
            path
        )
        state = ItemState(path=path, progress=initial_progress, status=initial_status)
        row = QtWidgets.QTreeWidgetItem(
            [path.name, "", self._friendly_status(state.status)]
        )
        row.setData(0, QtCore.Qt.UserRole, str(path))
        row.setToolTip(0, str(path))
        self._apply_row_padding(row)

        self.list_widget.addTopLevelItem(row)
        self.items[path] = state
        self.rows[path] = row

        self._build_progress_cell(path, row)
        self._refresh_row(path)
        return 1

    def _build_progress_cell(self, path: Path, row: QtWidgets.QTreeWidgetItem):
        cell = QtWidgets.QWidget()
        cell_layout = QtWidgets.QHBoxLayout(cell)
        cell_layout.setContentsMargins(10, 3, 10, 3)
        cell_layout.setSpacing(0)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat("0%")
        bar.setTextVisible(True)
        bar.setFixedWidth(220)
        bar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        cell_layout.addWidget(bar, 0, QtCore.Qt.AlignCenter)
        self.list_widget.setItemWidget(row, 1, cell)
        self.row_progress[path] = bar
        self.row_visual_progress[path] = self.items[path].progress

    def _refresh_row(self, path: Path):
        state = self.items.get(path)
        row = self.rows.get(path)
        if not state or not row:
            return

        friendly = self._friendly_status(state.status)
        row.setText(2, friendly)
        row.setForeground(2, QtGui.QBrush(QtGui.QColor(self._status_color(friendly))))
        self._refresh_progress_cell(path)

    def _refresh_progress_cell(self, path: Path):
        state = self.items.get(path)
        bar = self.row_progress.get(path)
        if not state or not bar:
            return

        value = max(0, min(100, state.progress))
        visual = max(self.row_visual_progress.get(path, 0), value)
        self.row_visual_progress[path] = visual

        bar.setRange(0, 100)
        bar.setValue(visual)
        bar.setFormat(f"{visual}%")

    def _reveal_file(self, path: Path):
        target = path.parent if path.exists() else path
        self._open_in_default_app(target)

    def _clear(self):
        if self.setup_in_progress or self._processing:
            return
        if not self.items:
            return

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Remove All")
        box.setText("Remove all files from the list?")
        cancel_btn = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        remove_btn = box.addButton("Remove", QtWidgets.QMessageBox.AcceptRole)
        box.setDefaultButton(cancel_btn)
        box.exec()

        if box.clickedButton() != remove_btn:
            return

        self.items.clear()
        self.rows.clear()
        self.row_progress.clear()
        self.row_visual_progress.clear()
        self.list_widget.clear()
        self._add_placeholder_rows()
        self.total_items = 0
        self.completed_items = 0
        self._update_overall(0, 0)
        self._set_progress_message("Add recordings to begin.")
        self._update_empty_state()

    def _start(self):
        if self.setup_in_progress or self._processing:
            return
        if not self.items:
            QtWidgets.QMessageBox.information(self, "No files", "No audio files added")
            return

        pending_items: list[Path] = []
        for path, state in self.items.items():
            initial_progress, initial_status = (
                self._initial_progress_and_status_for_path(path)
            )
            state.progress = initial_progress
            state.status = initial_status
            self.row_visual_progress[path] = initial_progress
            if not self._is_complete_status(state.status):
                pending_items.append(path)
            self._refresh_row(path)

        self.total_items = len(self.items)
        self.completed_items = sum(
            1 for state in self.items.values() if self._is_complete_status(state.status)
        )
        self._update_overall(self.completed_items, self.total_items)

        if not pending_items:
            self._set_progress_message("All files are already complete.")
            return

        self._set_controls_enabled(False)

        self.worker = Worker(
            pending_items,
            language="auto",
            include_timestamps=True,
        )
        self.worker.itemStatus.connect(self._on_item_status)
        self.worker.itemDone.connect(self._on_item_done)
        self.worker.allDone.connect(self._on_all_done)
        self.worker.finished.connect(self._on_worker_finished)

        self._set_processing_state(True)
        self._set_progress_message("Preparing files...")
        logger.info("Starting processing for %s item(s).", self.total_items)
        self.overall_progress.setAnimated(True)
        self._pulse_timer.start()
        self.worker.start()

    def _stop(self):
        if self.setup_in_progress:
            return
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._set_progress_message("Stopping transcription...")
            logger.info("Stop requested. Pending items will be marked as canceled.")

    def _set_processing_state(self, processing: bool):
        self._processing = processing
        self.stop_btn.setVisible(processing)
        self.start_btn.setVisible(not processing)

    def _set_controls_enabled(self, enabled: bool):
        if self.setup_in_progress:
            self.start_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            return

        self.start_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)

    @QtCore.Slot(Path, int, str, int, int)
    def _on_item_status(
            self, path: Path, progress: int, status: str, index: int, total: int
    ):
        state = self.items.get(path)
        if state:
            state.progress = progress
            state.status = status

        self._refresh_row(path)

        if status == "Preprocessing":
            self._set_progress_message("Preparing files...")
        elif status == "Transcribing":
            self._set_progress_message(f"Transcribing file {index} of {total}")
        elif status == "Saving":
            self._set_progress_message("Finalizing transcript...")
        elif status.startswith("Error"):
            self._set_progress_message("Unable to process this file.")

        logger.debug("Item status: %s progress=%s status=%s", path, progress, status)

    @QtCore.Slot(Path)
    def _on_item_done(self, path: Path):
        state = self.items.get(path)
        if state and self._is_complete_status(state.status):
            self.completed_items += 1
        self._update_overall(self.completed_items, self.total_items)

    @QtCore.Slot()
    def _on_all_done(self):
        self._pulse_timer.stop()
        self.overall_progress.setAnimated(False)
        self._set_controls_enabled(True)
        self._set_processing_state(False)

        has_canceled = any(state.status == "Canceled" for state in self.items.values())
        has_errors = any(
            state.status.startswith("Error") for state in self.items.values()
        )
        if has_canceled:
            self._set_progress_message("Transcription stopped.")
        elif has_errors:
            self._set_progress_message("Please check the file format and try again.")
        else:
            self._set_progress_message("Transcription complete.")
            QtWidgets.QMessageBox.information(
                self, "Complete", "Your transcript is ready."
            )

        logger.info(
            "Processing finished. Completed: %s/%s",
            self.completed_items,
            self.total_items,
        )
        self._update_overall(self.completed_items, self.total_items)

    @QtCore.Slot()
    def _on_worker_finished(self):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self._close_requested:
            self._close_requested = False
            QtCore.QTimer.singleShot(0, self.close)

    def _update_overall(self, completed: int, total: int):
        self.overall_label.setText(f"Overall progress: {completed}/{total}")
        percent = int((completed / total) * 100) if total else 0
        self.overall_progress.setValue(percent)

    @QtCore.Slot()
    def _animate_progress(self):
        self.overall_progress.advancePattern()
        if not self._processing:
            return
        for path, state in self.items.items():
            if state.status != "Transcribing":
                continue
            current = self.row_visual_progress.get(path, 0)
            target_floor = max(10, state.progress)
            if current < target_floor:
                current = target_floor
            elif current < 89:
                current += 1
            self.row_visual_progress[path] = min(current, 89)
            self._refresh_progress_cell(path)

    def _open_in_default_app(self, path: Path):
        if not path.exists():
            logger.warning("Path does not exist: %s", path)
            return
        url = QtCore.QUrl.fromLocalFile(str(path))
        QtGui.QDesktopServices.openUrl(url)

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _on_queue_row_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        del column
        raw = item.data(0, QtCore.Qt.UserRole)
        if not raw:
            return
        self._reveal_file(Path(raw))

    def _apply_row_padding(self, item: QtWidgets.QTreeWidgetItem):
        size = QtCore.QSize(0, 34)
        item.setSizeHint(0, size)
        item.setSizeHint(1, size)
        item.setSizeHint(2, size)

    def _update_empty_state(self):
        self._add_placeholder_rows()
        self._update_queue_height()

    def closeEvent(self, event):
        if self.setup_worker and self.setup_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Setup in progress",
                "The app is still preparing. Please wait before closing.",
            )
            event.ignore()
            return

        if self.worker and self.worker.isRunning():
            if not self._close_requested:
                answer = QtWidgets.QMessageBox.question(
                    self,
                    "Transcription in progress",
                    "Transcription is still running. Stop and close?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes,
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    event.ignore()
                    return
                self._close_requested = True
                self._set_controls_enabled(False)
                self.worker.stop()
                logger.info(
                    "Close requested while processing. Waiting for worker to stop."
                )
            event.ignore()
            return

        root = logging.getLogger()
        if self._log_handler and self._log_handler in root.handlers:
            root.removeHandler(self._log_handler)
            self._log_handler = None
        super().closeEvent(event)


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#f3f5f8"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#f2f4f7"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#1f1f1f"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#1f1f1f"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#1f2937"))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#1f2937"))
    app.setPalette(palette)

    icon_path = resource_path(os.path.join("files", "talk-to-text-icon.png"))
    if os.path.exists(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))

    window = TranscriberWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
