"""
VoiceToText Desktop Application - PyQt5 GUI
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QLineEdit, QPushButton, QProgressBar,
    QTextEdit, QMessageBox, QGroupBox, QFormLayout, QFileDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from transcribe import Transcriber


LANGUAGE_OPTIONS = [
    ("ja", "ja - Japanese"),
    ("auto", "auto - Automatic"),
    ("en", "en - English"),
    ("ms", "ms - Malay"),
    ("id", "id - Indonesian"),
    ("vi", "vi - Vietnamese"),
    ("th", "th - Thai"),
    ("ru", "ru - Russian"),
]

MODEL_SIZE_OPTIONS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
DEVICE_OPTIONS = [("cpu", "cpu"), ("cuda", "cuda")]
COMPUTE_TYPE_OPTIONS = [("int8", "int8"), ("int16", "int16"), ("float16", "float16")]


class TranscribeThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, audio_path, model_size, device, compute_type, language, beam_size):
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size

    def run(self):
        try:
            self.log.emit(f"Initializing model: {self.model_size} ({self.device})...")
            self.progress.emit(10)

            transcriber = Transcriber(
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language,
                beam_size=self.beam_size
            )

            self.log.emit(f"Transcribing: {self.audio_path}")
            self.progress.emit(30)

            result = transcriber.transcribe(self.audio_path)

            self.progress.emit(100)
            self.log.emit(f"Complete! Saved to: {result['output_path']}")

            if result.get('detected_language'):
                self.log.emit(f"Detected language: {result['detected_language']} ({result['language_probability']:.2%})")

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class VoiceToTextWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.transcribe_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("VoiceToText - Whisper Transcription")
        self.setMinimumSize(600, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        param_group = QGroupBox("Transcription Parameters")
        param_layout = QFormLayout()

        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(MODEL_SIZE_OPTIONS)
        self.model_size_combo.setCurrentText("medium")
        param_layout.addRow("Model Size:", self.model_size_combo)

        self.device_combo = QComboBox()
        for value, text in DEVICE_OPTIONS:
            self.device_combo.addItem(text, value)
        self.device_combo.setCurrentIndex(0)
        param_layout.addRow("Device:", self.device_combo)

        self.compute_type_combo = QComboBox()
        for value, text in COMPUTE_TYPE_OPTIONS:
            self.compute_type_combo.addItem(text, value)
        self.compute_type_combo.setCurrentText("int16")
        param_layout.addRow("Compute Type:", self.compute_type_combo)

        self.language_combo = QComboBox()
        for value, text in LANGUAGE_OPTIONS:
            self.language_combo.addItem(text, value)
        self.language_combo.setCurrentIndex(0)
        param_layout.addRow("Language:", self.language_combo)

        self.beam_size_spin = QSpinBox()
        self.beam_size_spin.setRange(1, 10)
        self.beam_size_spin.setValue(5)
        param_layout.addRow("Beam Size:", self.beam_size_spin)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        file_group = QGroupBox("Audio File")
        file_layout = QHBoxLayout()

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select an audio file (mp3, wav, m4a, etc.)")
        file_layout.addWidget(self.file_edit)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        self.start_btn = QPushButton("Start Transcription")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_transcribe)
        layout.addWidget(self.start_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        log_group = QGroupBox("Log / Status")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.wma);;All Files (*)"
        )
        if file_path:
            self.file_edit.setText(file_path)

    def start_transcribe(self):
        audio_path = self.file_edit.text().strip()

        if not audio_path:
            QMessageBox.warning(self, "Warning", "Please select an audio file!")
            return

        if not Path(audio_path).exists():
            QMessageBox.warning(self, "Warning", "Audio file not found!")
            return

        model_size = self.model_size_combo.currentText()
        device = self.device_combo.currentData()
        compute_type = self.compute_type_combo.currentData()
        language = self.language_combo.currentData()
        beam_size = self.beam_size_spin.value()

        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.transcribe_thread = TranscribeThread(
            audio_path, model_size, device, compute_type, language, beam_size
        )
        self.transcribe_thread.progress.connect(self.progress_bar.setValue)
        self.transcribe_thread.log.connect(self.log_message)
        self.transcribe_thread.finished.connect(self.transcribe_finished)
        self.transcribe_thread.error.connect(self.transcribe_error)
        self.transcribe_thread.start()

    def log_message(self, message):
        self.log_text.append(message)

    def transcribe_finished(self, result):
        self.start_btn.setEnabled(True)
        output_path = result.get('output_path', '')

        msg = QMessageBox(self)
        msg.setWindowTitle("Success")
        msg.setIcon(QMessageBox.Information)
        msg.setText("Transcription completed successfully!")
        msg.setDetailedText(f"Output file:\n{output_path}")
        msg.exec_()

    def transcribe_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        QMessageBox.critical(self, "Error", f"Transcription failed:\n{error_msg}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VoiceToTextWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
