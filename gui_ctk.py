import os
import sys
import threading
import traceback
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox

# --- Environment Fixes for Stability ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["QT_QPA_PLATFORM"] = "windows" # Just in case any other lib tries to use Qt

from transcribe import Transcriber

# Options
LANGUAGE_OPTIONS = ["ja", "auto", "en", "ms", "id", "vi", "th", "ru"]
MODEL_SIZE_OPTIONS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"]
DEVICE_OPTIONS = ["cpu", "cuda"]
COMPUTE_TYPE_OPTIONS = ["int8", "int16", "float16"]

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VoiceToText - Whisper Transcription (Modern)")
        self.geometry("700x600")
        
        # Appearance
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # 1. Configuration Frame
        self.config_frame = ctk.CTkFrame(self)
        self.config_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        self.config_frame.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(self.config_frame, text="Model Size:").grid(row=0, column=0, padx=10, pady=5)
        self.model_combo = ctk.CTkComboBox(self.config_frame, values=MODEL_SIZE_OPTIONS)
        self.model_combo.set("large-v3-turbo")
        self.model_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(self.config_frame, text="Device:").grid(row=0, column=2, padx=10, pady=5)
        self.device_combo = ctk.CTkComboBox(self.config_frame, values=DEVICE_OPTIONS)
        self.device_combo.set("cpu")
        self.device_combo.grid(row=0, column=3, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(self.config_frame, text="Compute Type:").grid(row=1, column=0, padx=10, pady=5)
        self.compute_combo = ctk.CTkComboBox(self.config_frame, values=COMPUTE_TYPE_OPTIONS)
        self.compute_combo.set("int8")
        self.compute_combo.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(self.config_frame, text="Language:").grid(row=1, column=2, padx=10, pady=5)
        self.lang_combo = ctk.CTkComboBox(self.config_frame, values=LANGUAGE_OPTIONS)
        self.lang_combo.set("ja")
        self.lang_combo.grid(row=1, column=3, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(self.config_frame, text="Beam Size:").grid(row=2, column=0, padx=10, pady=5)
        self.beam_spin = ctk.CTkEntry(self.config_frame)
        self.beam_spin.insert(0, "5")
        self.beam_spin.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Audio Preprocessing Options Frame
        self.audio_frame = ctk.CTkFrame(self)
        self.audio_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        ctk.CTkLabel(self.audio_frame, text="Audio Preprocessing (Optional):").pack(anchor="w", padx=10, pady=(5, 0))
        
        self.normalize_var = ctk.BooleanVar(value=False)
        self.normalize_check = ctk.CTkCheckBox(self.audio_frame, text="Normalize Volume (提高音量)", variable=self.normalize_var)
        self.normalize_check.pack(anchor="w", padx=20, pady=2)
        
        self.denoise_var = ctk.BooleanVar(value=False)
        self.denoise_check = ctk.CTkCheckBox(self.audio_frame, text="Noise Reduction (降噪)", variable=self.denoise_var)
        self.denoise_check.pack(anchor="w", padx=20, pady=2)
        
        self.highpass_var = ctk.BooleanVar(value=False)
        self.highpass_check = ctk.CTkCheckBox(self.audio_frame, text="High-Pass Filter (去除低频噪音)", variable=self.highpass_var)
        self.highpass_check.pack(anchor="w", padx=20, pady=2)

        # 2. File Selection Frame
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.file_frame.grid_columnconfigure(0, weight=1)

        self.file_entry = ctk.CTkEntry(self.file_frame, placeholder_text="Select a folder containing MP3 files...")
        self.file_entry.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")
        
        self.browse_btn = ctk.CTkButton(self.file_frame, text="Browse Folder", width=120, command=self.browse_file)
        self.browse_btn.grid(row=0, column=1, padx=(5, 10), pady=10)

        # 3. Action Frame
        self.start_btn = ctk.CTkButton(self, text="Start Transcription", height=40, command=self.start_task)
        self.start_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=4, column=0, padx=20, pady=5, sticky="ew")

        # 4. Log Area
        self.log_text = ctk.CTkTextbox(self)
        self.log_text.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")

    def browse_file(self):
        folder_path = filedialog.askdirectory(title="Select folder containing MP3 files")
        if folder_path:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, folder_path)

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def start_task(self):
        folder_path = self.file_entry.get().strip()
        if not folder_path or not Path(folder_path).exists():
            messagebox.showwarning("Warning", "Please select a valid folder!")
            return

        self.start_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.log_text.delete("1.0", "end")
        self.log("Scanning folder for MP3 files...")

        # Find all MP3 files in folder
        mp3_files = list(Path(folder_path).glob("*.mp3"))
        
        if not mp3_files:
            messagebox.showwarning("Warning", "No MP3 files found in the selected folder!")
            self.start_btn.configure(state="normal")
            return

        self.log(f"Found {len(mp3_files)} MP3 file(s)")

        # Gather params
        params = {
            "model_size": self.model_combo.get(),
            "device": self.device_combo.get(),
            "compute_type": self.compute_combo.get(),
            "language": self.lang_combo.get(),
            "beam_size": int(self.beam_spin.get()),
            "normalize": self.normalize_var.get(),
            "denoise": self.denoise_var.get(),
            "highpass": self.highpass_var.get()
        }

        # Run in thread
        threading.Thread(target=self.run_batch_transcription, args=(folder_path, mp3_files, params), daemon=True).start()

    def run_batch_transcription(self, folder_path, mp3_files, params):
        total = len(mp3_files)
        success_count = 0
        skip_count = 0
        error_count = 0
        
        try:
            self.log(f"Initializing model: {params['model_size']} ({params['device']})...")
            
            transcriber = Transcriber(
                model_size=params['model_size'],
                device=params['device'],
                compute_type=params['compute_type'],
                language=params['language'],
                beam_size=params['beam_size']
            )

            for idx, mp3_file in enumerate(mp3_files, 1):
                json_file = mp3_file.with_suffix('.json')
                
                # Skip if JSON already exists
                if json_file.exists():
                    self.log(f"[{idx}/{total}] SKIP: {mp3_file.name} (JSON already exists)")
                    skip_count += 1
                    continue

                audio_opts = ""
                if params.get("normalize"):
                    audio_opts += " +normalize"
                if params.get("denoise"):
                    audio_opts += " +denoise"
                if params.get("highpass"):
                    audio_opts += " +highpass"
                
                if audio_opts:
                    self.log(f"[{idx}/{total}] Processing: {mp3_file.name} [Audio:{audio_opts}]...")
                else:
                    self.log(f"[{idx}/{total}] Processing: {mp3_file.name}...")
                
                try:
                    result = transcriber.transcribe(
                        str(mp3_file),
                        normalize=params.get("normalize", False),
                        denoise=params.get("denoise", False),
                        highpass=params.get("highpass", False)
                    )
                    self.log(f"[{idx}/{total}] SUCCESS: {mp3_file.name} -> {result['output_path']}")
                    success_count += 1
                except Exception as e:
                    self.log(f"[{idx}/{total}] FAILED: {mp3_file.name} - {str(e)}")
                    error_count += 1

                # Update progress
                progress = idx / total
                self.after(0, lambda p=progress: self.progress_bar.set(p))

            # All done
            summary = f"Complete!\n\nTotal: {total}\nSuccess: {success_count}\nSkipped: {skip_count}\nFailed: {error_count}"
            self.log("\n" + summary)
            self.after(0, lambda: messagebox.showinfo("Batch Transcription Complete", summary))

        except Exception as e:
            err_msg = traceback.format_exc()
            self.log(f"ERROR: {err_msg}")
            self.after(0, lambda: messagebox.showerror("Error", f"Batch transcription failed:\n{str(e)}"))
        
        finally:
            self.after(0, lambda: self.start_btn.configure(state="normal"))

if __name__ == "__main__":
    app = App()
    app.mainloop()
