import os
import json
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any
from faster_whisper import WhisperModel

def get_model_path():
    # 动态获取当前用户的 .cache 路径
    home = os.path.expanduser("~")
    local_path = os.path.join(home, ".cache", "huggingface", "hub", "faster-whisper-large-v3-turbo-ct2")
    
    if os.path.exists(local_path): 
        return local_path
    return "deepdml/faster-whisper-large-v3-turbo-ct2"

class Transcriber:
    def __init__(self, device="cpu", compute_type="int8", cpu_threads=0):
        model_path = get_model_path()
        threads = cpu_threads if cpu_threads > 0 else multiprocessing.cpu_count()
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type, cpu_threads=threads, num_workers=1)

    def transcribe(self, audio_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        output_path = str(Path(audio_path).with_suffix(".json"))
        results = self._perform_transcription(audio_path, options)
        
        self._save_json(results, output_path)
        return {"results": results, "output_path": output_path}

    def transcribe_text_only(self, audio_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        output_path = str(Path(audio_path).with_suffix(".json"))
        results = self._perform_transcription(audio_path, options)
        
        # 移除单词级别的信息，只保留句子级别
        for entry in results:
            if "words" in entry:
                del entry["words"]
                
        self._save_json(results, output_path)
        return {"results": results, "output_path": output_path}

    def _perform_transcription(self, audio_path: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        segments, _ = self.model.transcribe(
            audio_path,
            language=options.get("language", "ja"),
            beam_size=options.get("beam_size", 7),
            word_timestamps=True,
            initial_prompt=options.get("initial_prompt", ""),
            vad_filter=options.get("vad_filter", False),
            vad_parameters={
                "min_silence_duration_ms": options.get("min_silence_ms", 2000),
                "threshold": 0.35
            },
            no_speech_threshold=options.get("no_speech_threshold", 0.6),
            compression_ratio_threshold=options.get("compression_threshold", 2.4),
            condition_on_previous_text=True
        )

        results = []
        gap_threshold = options.get("gap_threshold", 2.5)

        for segment in segments:
            if not segment.words: continue
            current_line_words = []
            for w in segment.words:
                if current_line_words and (w.start - current_line_words[-1]["end"]) > gap_threshold:
                    results.append(self._build_line(current_line_words))
                    current_line_words = []
                current_line_words.append({"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)})
            if current_line_words:
                results.append(self._build_line(current_line_words))
        return results

    def _build_line(self, words_list: List[Dict]) -> Dict:
        return {
            "start": words_list[0]["start"], "end": words_list[-1]["end"],
            "text": "".join([w["word"] for w in words_list]).strip(),
            "words": words_list
        }

    @staticmethod
    def _save_json(data: List[Dict[str, Any]], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)