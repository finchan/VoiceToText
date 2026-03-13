"""
Japanese Speech Transcription Module (Standard Version)
"""

from faster_whisper import WhisperModel
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import COMPUTE_TYPE

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
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str, output_path: str = None) -> List[Dict[str, Any]]:
        if output_path is None:
            audio_file = Path(audio_path)
            output_path = str(audio_file.with_suffix('.json'))

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=self.language,
            word_timestamps=True,
            initial_prompt="こんにちは。今日は漢字とかなを使って日本語で話します。"
        )

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
                ]
            }
            results.append(line_data)

        self._save_json(results, output_path)
        return {
            "results": results,
            "detected_language": detected_lang,
            "language_probability": lang_prob,
            "output_path": output_path
        }

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
