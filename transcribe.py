"""
Japanese Speech Transcription Module (Standard Version)
"""

import os
import sys
import tempfile
import shutil
from faster_whisper import WhisperModel
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import COMPUTE_TYPE

def get_cache_dir():
    # 获取模型存放的根目录
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), 'models')
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def preprocess_audio(audio_path: str, normalize: bool = False, denoise: bool = False, highpass: bool = False) -> str:
    """
    Preprocess audio file with various filters.
    Returns path to processed audio file (or original if no processing needed).
    """
    if not normalize and not denoise and not highpass:
        return audio_path

    try:
        import numpy as np
        import scipy.signal as signal
        import soundfile as sf

        audio, sample_rate = sf.read(audio_path)

        # Handle stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # High-pass filter (remove low frequency noise below 80Hz)
        if highpass:
            nyquist = sample_rate / 2
            high = 80 / nyquist
            b, a = signal.butter(4, high, btype='high')
            audio = signal.filtfilt(b, a, audio)

        # Simple noise reduction (spectral gating - basic version)
        if denoise:
            # Calculate noise profile from first 0.5 seconds (assumed silence)
            noise_samples = min(int(0.5 * sample_rate), len(audio) // 4)
            noise_profile = np.mean(audio[:noise_samples]**2)
            threshold = np.sqrt(noise_profile) * 1.5
            audio = np.where(np.abs(audio) > threshold, audio, audio * 0.1)

        # Normalize volume
        if normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            audio = audio * 0.9  # Target 90% max amplitude

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio, sample_rate)
        return temp_file.name

    except Exception as e:
        # If preprocessing fails, return original file
        return audio_path


class Transcriber:
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = COMPUTE_TYPE,
        language: str = "ja",
        beam_size: int = 5,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language if language != "auto" else None
        self.beam_size = beam_size
        self._model: Optional[WhisperModel] = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            user_home = os.path.expanduser("~")
            hf_hub_path = os.path.join(user_home, ".cache", "huggingface", "hub")

            # --- 核心修改：匹配你截图中的文件夹名 ---
            # 你的截图中文件夹名就是这个，没有任何前缀
            local_folder_name = "faster-whisper-large-v3-turbo-ct2"

            target_model = self.model_size

            if self.model_size == "large-v3-turbo":
                # 尝试拼接路径：C:\Users\Administrator\.cache\huggingface\hub\faster-whisper-large-v3-turbo-ct2
                potential_path = os.path.join(hf_hub_path, local_folder_name)

                if os.path.exists(potential_path):
                    # 如果该路径存在，强制使用该文件夹
                    target_model = potential_path
                    print(f"--- 成功定位本地模型: {target_model} ---")
                else:
                    print(f"--- 未找到本地模型文件夹，将尝试联网下载 ---")

            # 初始化模型
            self._model = WhisperModel(
                target_model,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,  # 维持 4 线程，适合 16GB 内存
                num_workers=1
            )
        return self._model

    def transcribe(self, audio_path: str, output_path: str = None, normalize: bool = False, denoise: bool = False, highpass: bool = False) -> List[Dict[str, Any]]:
        import sys
        import os
        write_log = lambda msg: None
        if getattr(sys, 'frozen', False):
            log_file = os.path.join(os.path.dirname(sys.executable), 'app.log')
            def write_log(msg):
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')

        write_log(f"transcribe() called with audio_path: {audio_path}")

        if output_path is None:
            audio_file = Path(audio_path)
            output_path = str(audio_file.with_suffix('.json'))

        # Preprocess audio if needed
        temp_file = None
        try:
            processed_audio = preprocess_audio(audio_path, normalize, denoise, highpass)
            if processed_audio != audio_path:
                write_log(f"Audio preprocessed, using: {processed_audio}")
                temp_file = processed_audio
                audio_to_transcribe = temp_file
            else:
                audio_to_transcribe = audio_path

            write_log("About to call model.transcribe()...")

            try:
                # 重新启用时间戳，并保留 None 检查
                segments, info = self.model.transcribe(
                    audio_to_transcribe,
                    beam_size=self.beam_size,
                    language=self.language,
                    word_timestamps=True,
                    initial_prompt="こんにちは。"
                )
                write_log("model.transcribe() returned successfully")
            except Exception as e:
                write_log(f"Exception during transcribe: {type(e).__name__}: {e}")
                raise

            detected_lang = info.language if self.language is None else self.language
            lang_prob = info.language_probability

            results: List[Dict[str, Any]] = []
            for segment in segments:
                line_data = {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text,
                    "words": [
                        {
                            "word": w.word,
                            "start": round(w.start, 2),
                            "end": round(w.end, 2)
                        } for w in segment.words
                    ] if segment.words else []
                }
                results.append(line_data)

            self._save_json(results, output_path)
            return {
                "results": results,
                "detected_language": detected_lang,
                "language_probability": lang_prob,
                "output_path": output_path
            }

        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    @staticmethod
    def _save_json(data: List[Dict[str, Any]], output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_data(json_path: str = "data.json") -> List[Dict[str, Any]]:
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)