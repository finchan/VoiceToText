import os
import threading
import multiprocessing
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox
from transcribe import Transcriber

LANG_MAP = {"日语 (ja)": "ja", "自动识别 (auto)": "auto", "中文 (zh)": "zh", "英语 (en)": "en"}

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Whisper 教材精调工具 - 解决丢字/幻觉")
        self.geometry("900x880")

        # --- 1. 性能设置 ---
        self.perf_frame = ctk.CTkFrame(self)
        self.perf_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.perf_frame, text="核数:").pack(side="left", padx=5)
        self.cpu_spin = ctk.CTkEntry(self.perf_frame, width=50)
        self.cpu_spin.insert(0, "4")
        self.cpu_spin.pack(side="left", padx=5)
        self.device_var = ctk.StringVar(value="cpu")
        ctk.CTkOptionMenu(self.perf_frame, values=["cpu", "cuda"], variable=self.device_var, width=80).pack(side="left", padx=10)

        # --- 2. 核心调试参数 (改用 CTkFrame 替代 CTkLabelFrame) ---
        self.debug_outer = ctk.CTkFrame(self)
        self.debug_outer.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.debug_outer, text="—— 深度调试参数 (针对教材优化) ——", font=("Arial", 13, "bold")).pack(pady=5)

        # 第一行：断句 Gap 与 VAD 开关
        row1 = ctk.CTkFrame(self.debug_outer, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row1, text="断句间隙 (Gap 秒):").pack(side="left", padx=5)
        self.gap_val = ctk.CTkEntry(row1, width=60)
        self.gap_val.insert(0, "0.7") 
        self.gap_val.pack(side="left", padx=5)
        
        def toggle_vad_settings():
            state = "normal" if self.vad_enabled.get() else "disabled"
            self.vad_ms.configure(state=state)

        self.vad_enabled = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(row1, text="启用 VAD (易丢开头)", variable=self.vad_enabled, command=toggle_vad_settings).pack(side="left", padx=20)

        # 第二行：VAD 细节控制 (增加静音时长)
        row2 = ctk.CTkFrame(self.debug_outer, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row2, text="VAD 静音时长 (ms):").pack(side="left", padx=5)
        self.vad_ms = ctk.CTkEntry(row2, width=80)
        self.vad_ms.insert(0, "3000") # 教材建议设为 3 秒
        self.vad_ms.configure(state="disabled")
        self.vad_ms.pack(side="left", padx=5)
        ctk.CTkLabel(row2, text="（已绑定：仅启用VAD时生效并可编辑）", font=("Arial", 11), text_color="gray").pack(side="left", padx=5)

        # 第三行：防幻觉与推理
        row3 = ctk.CTkFrame(self.debug_outer, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row3, text="无声阈值 (0.6):").pack(side="left", padx=5)
        self.no_speech_val = ctk.CTkEntry(row3, width=60)
        self.no_speech_val.insert(0, "0.6")
        self.no_speech_val.pack(side="left", padx=5)
        
        ctk.CTkLabel(row3, text="推理深度 (Beam):").pack(side="left", padx=20)
        self.beam_val = ctk.CTkEntry(row3, width=60)
        self.beam_val.insert(0, "7")
        self.beam_val.pack(side="left", padx=5)

        # --- 3. 业务设置 ---
        self.biz_frame = ctk.CTkFrame(self)
        self.biz_frame.pack(fill="x", padx=20, pady=10)
        self.lang_var = ctk.StringVar(value="日语 (ja)")
        ctk.CTkOptionMenu(self.biz_frame, values=list(LANG_MAP.keys()), variable=self.lang_var).pack(side="left", padx=10)
        self.prompt_entry = ctk.CTkEntry(self.biz_frame, placeholder_text="起始提示词 (Initial Prompt)...", width=400)
        self.prompt_entry.pack(side="left", padx=10, fill="x", expand=True)

        # --- 4. 操作区 ---
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkButton(self.btn_frame, text="📁 选取音频", command=self.select_files).pack(side="left", padx=10)
        self.start_btn = ctk.CTkButton(self.btn_frame, text="🚀 开始解析", fg_color="#1f538d", command=self.start_transcription)
        self.start_btn.pack(side="right", padx=10)

        self.log_text = ctk.CTkTextbox(self, height=300)
        self.log_text.pack(fill="both", expand=True, padx=20, pady=10)
        self.selected_files = []

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio", "*.mp3 *.wav *.m4a")])
        if files:
            self.selected_files = [Path(f) for f in files]
            self.log(f"已选取: {len(self.selected_files)} 个音频")

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def start_transcription(self):
        if not self.selected_files: return
        self.start_btn.configure(state="disabled")
        try:
            options = {
                "language": LANG_MAP[self.lang_var.get()],
                "initial_prompt": self.prompt_entry.get(),
                "vad_filter": self.vad_enabled.get(),
                "min_silence_ms": int(self.vad_ms.get()),
                "gap_threshold": float(self.gap_val.get()),
                "no_speech_threshold": float(self.no_speech_val.get()),
                "beam_size": int(self.beam_val.get()),
                "cpu_threads": int(self.cpu_spin.get()),
                "device": self.device_var.get(),
                "compression_threshold": 2.4 # 内部固定
            }
            threading.Thread(target=self.run_process, args=(options,), daemon=True).start()
        except Exception as e:
            self.log(f"参数错误: {e}")
            self.start_btn.configure(state="normal")

    def run_process(self, options):
        try:
            ts = Transcriber(device=options["device"], cpu_threads=options["cpu_threads"])
            for f in self.selected_files:
                self.log(f"正在处理: {f.name}...")
                ts.transcribe(str(f), options)
                self.log(f"✅ 完成: {f.stem}.json")
            messagebox.showinfo("完成", "任务处理完毕")
        except Exception as e:
            self.log(f"❌ 运行错误: {str(e)}")
        finally:
            self.start_btn.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()