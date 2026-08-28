"""Data structures and schemas for transcription and diarization."""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class WordInfo(BaseModel):
    """Word-level timing and speaker annotation."""

    word: str
    start: float
    end: float
    probability: float = 1.0
    speaker: Optional[str] = None


class TranscriptSegment(BaseModel):
    """Raw transcription segment from ASR."""

    id: int
    start: float
    end: float
    text: str
    words: List[WordInfo] = Field(default_factory=list)
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    emotion: Optional[str] = None
    events: List[str] = Field(default_factory=list)


class SpeakerSegment(BaseModel):
    """Time interval identified for a specific speaker."""

    speaker: str
    start: float
    end: float


class DiarizedSegment(BaseModel):
    """Aligned transcript segment with speaker attribution."""

    id: int
    speaker: str
    start: float
    end: float
    text: str
    words: List[WordInfo] = Field(default_factory=list)
    emotion: Optional[str] = None
    events: List[str] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Complete result container."""

    language: str
    language_probability: float = 1.0
    duration: float
    segments: List[DiarizedSegment] = Field(default_factory=list)
    speakers: List[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Formatted full text with speaker tags and emotion/event annotations."""
        lines = []
        for seg in self.segments:
            badge = ""
            if seg.emotion and seg.emotion != "NEUTRAL":
                badge += f" [{seg.emotion}]"
            if seg.events:
                badge += f" [{' '.join(seg.events)}]"
            lines.append(f"[{seg.speaker}]{badge} ({seg.start:.2f}s - {seg.end:.2f}s): {seg.text.strip()}")
        return "\n".join(lines)


class ASRModelInfo(BaseModel):
    """Metadata schema for available ASR/STT models."""

    name: str
    family: str
    display_name: str
    params: str
    vram: str
    speed_factor: str
    languages: str
    description: str
    quantization_options: List[str] = ["float16", "int8", "int8_float16"]
    default_compute_type: str = "default"
    capabilities: List[str] = ["local", "gpu"]
    implemented: bool = True
    is_local: bool = True
    is_cached: bool = True


def _is_dir_non_empty(p: Optional[Path]) -> bool:
    """Return True if path is a non-empty directory."""
    if not p or not p.exists() or not p.is_dir():
        return False
    try:
        return any(p.iterdir())
    except Exception:
        return False


def _check_hf_cache(name: str) -> bool:
    """Check if model name matches any cached HuggingFace model repo."""
    hf_cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    if not hf_cache_dir.exists():
        return False
    target = name.lower()
    for entry in hf_cache_dir.glob("models--*"):
        c = entry.name.lower()
        if target in c or c.endswith(f"faster-whisper-{target}") or c.endswith(f"faster-distil-whisper-{target}"):
            return True
    return False


def _check_modelscope_cache(name: str) -> bool:
    """Check if model exists in local ModelScope cache."""
    ms_cache_dir = Path(os.path.expanduser("~/.cache/modelscope/hub"))
    if not ms_cache_dir.exists():
        return False
    target = name.lower().replace("-", "").replace("_", "")
    for entry in ms_cache_dir.glob("*/*"):
        if target in entry.name.lower().replace("-", "").replace("_", ""):
            return True
    return False


def check_model_cached(name: str) -> bool:
    """Check if model weights exist in local data/models directory, Hugging Face or ModelScope cache."""
    candidates = [
        Path("data/models") / name,
        Path("data/models") / f"{name}-ct2",
        Path("data/models/moonshine") / name.replace("moonshine-", ""),
        Path("data/models/mms-1b-all") if name == "meta-omnilingual-asr" else None,
    ]
    if any(_is_dir_non_empty(p) for p in candidates):
        return True
    return _check_hf_cache(name) or _check_modelscope_cache(name)


MODEL_CATALOG: List[ASRModelInfo] = [
    ASRModelInfo(name="tiny", family="Faster-Whisper", display_name="Whisper Tiny", params="39M", vram="~1 GB", speed_factor="~32x", languages="Multilingual (99+)", description="Fastest multilingual model with minimal resource requirements", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="tiny.en", family="Faster-Whisper", display_name="Whisper Tiny English", params="39M", vram="~1 GB", speed_factor="~32x", languages="English Only", description="Optimized English-only tiny model", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="base", family="Faster-Whisper", display_name="Whisper Base", params="74M", vram="~1 GB", speed_factor="~16x", languages="Multilingual (99+)", description="Default balanced standard model for fast general-purpose transcription", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="base.en", family="Faster-Whisper", display_name="Whisper Base English", params="74M", vram="~1 GB", speed_factor="~16x", languages="English Only", description="Optimized English-only base model", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="small", family="Faster-Whisper", display_name="Whisper Small", params="244M", vram="~2 GB", speed_factor="~6x", languages="Multilingual (99+)", description="High accuracy model balanced for moderate hardware", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="small.en", family="Faster-Whisper", display_name="Whisper Small English", params="244M", vram="~2 GB", speed_factor="~6x", languages="English Only", description="Optimized English-only small model", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="distil-small.en", family="Faster-Whisper", display_name="Distil-Whisper Small (EN)", params="166M", vram="~1.5 GB", speed_factor="~10x", languages="English Only", description="Knowledge-distilled small model, 5x faster than small with near-zero WER loss", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="medium", family="Faster-Whisper", display_name="Whisper Medium", params="769M", vram="~5 GB", speed_factor="~2x", languages="Multilingual (99+)", description="Near production-grade multilingual transcription accuracy", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="medium.en", family="Faster-Whisper", display_name="Whisper Medium English", params="769M", vram="~5 GB", speed_factor="~2x", languages="English Only", description="Optimized English-only medium model", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="distil-medium.en", family="Faster-Whisper", display_name="Distil-Whisper Medium (EN)", params="394M", vram="~3 GB", speed_factor="~5x", languages="English Only", description="Knowledge-distilled medium model for high-speed English ASR", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="large-v1", family="Faster-Whisper", display_name="Whisper Large-v1", params="1550M", vram="~10 GB", speed_factor="~1x", languages="Multilingual (99+)", description="First-generation large Whisper checkpoint", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="large-v2", family="Faster-Whisper", display_name="Whisper Large-v2", params="1550M", vram="~10 GB", speed_factor="~1x", languages="Multilingual (99+)", description="Second-generation large Whisper checkpoint with improved audio encoding", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="large-v3", family="Faster-Whisper", display_name="Whisper Large-v3", params="1550M", vram="~10 GB", speed_factor="~1x", languages="Multilingual (99+)", description="State-of-the-art multilingual Whisper model with 128 Mel frequency bins", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="large-v3-turbo", family="Faster-Whisper", display_name="Whisper Large-v3 Turbo", params="809M", vram="~6 GB", speed_factor="~8x", languages="Multilingual (99+)", description="Pruned 4-decoder large-v3 architecture; 8x faster with large-v3 accuracy", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="turbo", family="Faster-Whisper", display_name="Whisper Turbo", params="809M", vram="~6 GB", speed_factor="~8x", languages="Multilingual (99+)", description="Alias for large-v3-turbo", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="distil-large-v2", family="Faster-Whisper", display_name="Distil-Whisper Large-v2", params="756M", vram="~5 GB", speed_factor="~6x", languages="English & Multi", description="Distilled large-v2 model with 6x real-time speedup", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="distil-large-v3", family="Faster-Whisper", display_name="Distil-Whisper Large-v3", params="756M", vram="~5 GB", speed_factor="~6x", languages="English & Multi", description="Distilled large-v3 model with state-of-the-art distilled accuracy", capabilities=["local", "gpu", "vad"]),
    # Indonesian Fine-tunes
    ASRModelInfo(name="cahya-whisper-tiny-id", family="Indonesian Fine-tune", display_name="Cahya Whisper Tiny (ID)", params="39M", vram="~1 GB", speed_factor="~32x", languages="Indonesian (id)", description="Specialized Indonesian fine-tune (cahya/whisper-tiny-id)", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="cahya-whisper-base-id", family="Indonesian Fine-tune", display_name="Cahya Whisper Base (ID)", params="74M", vram="~1 GB", speed_factor="~16x", languages="Indonesian (id)", description="Specialized Indonesian fine-tune (cahya/whisper-base-id)", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="cahya-whisper-small-id", family="Indonesian Fine-tune", display_name="Cahya Whisper Small (ID)", params="244M", vram="~2 GB", speed_factor="~6x", languages="Indonesian (id)", description="Specialized Indonesian fine-tune (cahya/whisper-small-id)", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="cahya-whisper-medium-id", family="Indonesian Fine-tune", display_name="Cahya Whisper Medium (ID)", params="769M", vram="~5 GB", speed_factor="~2x", languages="Indonesian (id)", description="Specialized Indonesian fine-tune (cahya/whisper-medium-id)", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="cahya-faster-whisper-medium-id", family="Indonesian Fine-tune", display_name="Cahya Faster-Whisper Medium (ID)", params="769M", vram="~5 GB", speed_factor="~3x", languages="Indonesian (id)", description="CTranslate2 converted Indonesian checkpoint for faster inference", capabilities=["local", "gpu", "vad"]),
    ASRModelInfo(name="indonesian-wav2vec2-regional", family="Indonesian CTC", display_name="Wav2Vec2 Regional (ID/JV/SU)", params="317M", vram="~2 GB", speed_factor="~15x", languages="ID / JV / SU", description="indonesian-nlp/wav2vec2-indonesian-javanese-sundanese acoustic CTC", capabilities=["local", "gpu", "ctc"]),
    ASRModelInfo(name="indonesian-wav2vec2-large-xlsr", family="Indonesian CTC", display_name="Wav2Vec2 Large XLSR (ID)", params="317M", vram="~2 GB", speed_factor="~15x", languages="Indonesian (id)", description="cahya/wav2vec2-large-xlsr-indonesian acoustic CTC", capabilities=["local", "gpu", "ctc"]),
    # Next-Gen Open ASR & Edge Architectures
    ASRModelInfo(name="moonshine-tiny", family="UsefulSensors Moonshine", display_name="Moonshine Tiny Edge", params="27M", vram="~1 GB", speed_factor="~20x", languages="English Only", description="UsefulSensors/moonshine-tiny zero-overhead edge ONNX model", capabilities=["local", "edge", "onnx"]),
    ASRModelInfo(name="moonshine-base", family="UsefulSensors Moonshine", display_name="Moonshine Base Edge", params="61M", vram="~2 GB", speed_factor="~10x", languages="English Only", description="UsefulSensors/moonshine-base accurate edge ONNX model", capabilities=["local", "edge", "onnx"]),
    ASRModelInfo(name="sensevoice-small", family="Alibaba SenseVoice", display_name="SenseVoice-Small", params="230M", vram="~1.5 GB", speed_factor="~50x", languages="ZH / EN / JA / KO / YUE", description="FunAudioLLM/SenseVoiceSmall rich audio transcription (50x real-time) with SER & AED", capabilities=["local", "gpu", "ser", "aed"]),
    ASRModelInfo(name="meta-omnilingual-asr", family="Meta MMS", display_name="Meta MMS-1B Omnilingual", params="1B", vram="~2 GB", speed_factor="~8x", languages="Multilingual (100+)", description="facebook/mms-1b-all universal speech recognition with adapter switching", capabilities=["local", "gpu", "ctc", "mms"]),
    ASRModelInfo(name="meta-mms-1b-fl102", family="Meta MMS", display_name="Meta MMS-1B FLORES-102", params="1B", vram="~2 GB", speed_factor="~8x", languages="Top 102 World Langs", description="facebook/mms-1b-fl102 optimized for 102 major languages", capabilities=["local", "gpu", "ctc", "mms"]),
    ASRModelInfo(name="meta-mms-300m", family="Meta MMS", display_name="Meta MMS-300M Edge", params="300M", vram="~800 MB", speed_factor="~15x", languages="Multilingual (1400+)", description="facebook/mms-300m-1400 lightweight multilingual acoustic CTC model", capabilities=["local", "gpu", "ctc", "mms"]),
    ASRModelInfo(name="omniasr-ctc-300m", family="Meta OmniASR", display_name="OmniASR CTC 300M (1600+)", params="300M", vram="~1 GB", speed_factor="~18x", languages="Multilingual (1600+)", description="Meta Omnilingual ASR (bezzam/omniasr-ctc-300m-v2) supporting 1600+ languages", capabilities=["local", "gpu", "ctc"]),
    ASRModelInfo(name="omniasr-ctc-1b", family="Meta OmniASR", display_name="OmniASR CTC 1B (1600+)", params="1B", vram="~2.5 GB", speed_factor="~10x", languages="Multilingual (1600+)", description="Meta Omnilingual ASR (bezzam/omniasr-ctc-1b-v2) 1B parameter flagship", capabilities=["local", "gpu", "ctc"]),
    # FireRed Team Industrial ASR & Audio-LLM
    ASRModelInfo(name="fireredasr-aed-l", family="FireRed Team", display_name="FireRedASR AED Large", params="1.1B", vram="~2 GB", speed_factor="~15x", languages="Mandarin / Dialects / EN", description="FireRedTeam/FireRedASR-AED-L industrial Conformer-AED speech recognition", capabilities=["local", "gpu", "aed"]),
    ASRModelInfo(name="fireredasr-llm-l", family="FireRed Team", display_name="FireRedASR LLM Large", params="7.8B", vram="~8 GB", speed_factor="~5x", languages="Mandarin / Dialects / EN", description="FireRedTeam/FireRedASR-LLM-L Qwen2-7B encoder-adapter-LLM speech interaction", capabilities=["local", "gpu", "llm"]),
    ASRModelInfo(name="fireredaudio-9b", family="FireRed Team", display_name="FireRedAudio 9B", params="9B", vram="~10 GB", speed_factor="~4x", languages="Multilingual / Audio Q&A", description="FireRedTeam/FireRedAudio unified 9B audio-language foundation model", capabilities=["local", "gpu", "llm", "multimodal"]),
    # Tsinghua VoiceMem Dual-Brain Cognitive Voice Memory
    ASRModelInfo(name="voicemem-normal", family="Tsinghua VoiceMem", display_name="VoiceMem Normal (SER+Voiceprint)", params="Dual-Brain", vram="~2.5 GB", speed_factor="~20x", languages="Multilingual", description="Tsinghua xzf-thu/VoiceMem cognitive voice memory & emotion perception", capabilities=["local", "gpu", "ser", "memory"]),
    ASRModelInfo(name="voicemem-realtime", family="Tsinghua VoiceMem", display_name="VoiceMem Streaming Realtime", params="Dual-Brain", vram="~2.5 GB", speed_factor="~30x", languages="Multilingual", description="Tsinghua xzf-thu/VoiceMem low-latency (~134ms) streaming voice memory", capabilities=["local", "gpu", "ser", "memory", "streaming"]),
    # Whisper.cpp (GGML / GGUF High-Efficiency C++ Engine)
    ASRModelInfo(name="whispercpp-tiny", family="Whisper.cpp (GGML)", display_name="Whisper.cpp Tiny (GGML)", params="39M", vram="~100 MB", speed_factor="~35x", languages="Multilingual (99+)", description="Whisper.cpp ultra-lightweight GGML quantized runtime", capabilities=["local", "cpu", "ggml", "edge"]),
    ASRModelInfo(name="whispercpp-base", family="Whisper.cpp (GGML)", display_name="Whisper.cpp Base (GGML)", params="74M", vram="~200 MB", speed_factor="~25x", languages="Multilingual (99+)", description="Whisper.cpp base GGML runtime with multi-threaded CPU SIMD", capabilities=["local", "cpu", "ggml", "edge"]),
    ASRModelInfo(name="whispercpp-small", family="Whisper.cpp (GGML)", display_name="Whisper.cpp Small (GGML)", params="244M", vram="~600 MB", speed_factor="~15x", languages="Multilingual (99+)", description="Whisper.cpp small GGML runtime for balance of speed and precision", capabilities=["local", "cpu", "ggml"]),
    ASRModelInfo(name="whispercpp-medium", family="Whisper.cpp (GGML)", display_name="Whisper.cpp Medium (GGML)", params="769M", vram="~1.5 GB", speed_factor="~8x", languages="Multilingual (99+)", description="Whisper.cpp medium GGML runtime", capabilities=["local", "cpu", "ggml"]),
    ASRModelInfo(name="whispercpp-turbo", family="Whisper.cpp (GGML)", display_name="Whisper.cpp Turbo (GGML)", params="809M", vram="~1.5 GB", speed_factor="~12x", languages="Multilingual (99+)", description="Whisper.cpp large-v3-turbo GGML runtime", capabilities=["local", "cpu", "ggml"]),
    ASRModelInfo(name="whispercpp-large-v3", family="Whisper.cpp (GGML)", display_name="Whisper.cpp Large-v3 (GGML)", params="1550M", vram="~3.0 GB", speed_factor="~5x", languages="Multilingual (99+)", description="Whisper.cpp flagship large-v3 GGML runtime", capabilities=["local", "cpu", "ggml"]),
    # NVIDIA NeMo / Parakeet & Nemotron Speech Family
    ASRModelInfo(name="nvidia-parakeet-tdt-1.1b", family="NVIDIA NeMo", display_name="NVIDIA Parakeet-TDT 1.1B", params="1.1B", vram="~2.5 GB", speed_factor="~25x", languages="English (en)", description="NVIDIA Fast-Conformer Transducer (TDT) high-accuracy conversational ASR", capabilities=["local", "gpu", "tdt", "nemo"]),
    ASRModelInfo(name="nvidia-parakeet-tdt-0.6b", family="NVIDIA NeMo", display_name="NVIDIA Parakeet-TDT 0.6B", params="600M", vram="~1.5 GB", speed_factor="~35x", languages="English (en)", description="NVIDIA Fast-Conformer Transducer (TDT) lightweight streaming ASR", capabilities=["local", "gpu", "tdt", "nemo"]),
    ASRModelInfo(name="nvidia-parakeet-ctc-1.1b", family="NVIDIA NeMo", display_name="NVIDIA Parakeet-CTC 1.1B", params="1.1B", vram="~2.5 GB", speed_factor="~25x", languages="English (en)", description="NVIDIA Fast-Conformer CTC 1.1B speech recognition model", capabilities=["local", "gpu", "ctc", "nemo"]),
    ASRModelInfo(name="nvidia-nemotron-speech-3.5", family="NVIDIA NeMo", display_name="NVIDIA Nemotron Speech 3.5", params="3.5B", vram="~5.0 GB", speed_factor="~15x", languages="English / Multilingual", description="NVIDIA Nemotron Speech 3.5 cache-aware real-time streaming ASR", capabilities=["local", "gpu", "streaming", "nemo"]),
    ASRModelInfo(name="nemotron-audex-2b", family="NVIDIA NeMo", display_name="NVIDIA Audex-2B Audio-LLM", params="2B", vram="~4.5 GB", speed_factor="~10x", languages="English / Multilingual", description="NVIDIA Nemotron-Labs-Audex-2B compact unified audio-text LLM (Thinking & Instruct)", capabilities=["local", "gpu", "llm", "multimodal", "nemo"]),
]

CLOUD_MODEL_CATALOG: List[ASRModelInfo] = [
    ASRModelInfo(name="openai-whisper-api", family="Cloud Hosted API", display_name="OpenAI Whisper API", params="Cloud (1550M)", vram="Cloud", speed_factor="~20x", languages="Multilingual (99+)", description="OpenAI Hosted Whisper API (whisper-1 / gpt-4o-transcribe)", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
    ASRModelInfo(name="google-gemini-audio", family="Cloud Hosted API", display_name="Google Gemini 2.5 Audio", params="Cloud (Gemini)", vram="Cloud", speed_factor="~25x", languages="Multilingual (100+)", description="Google Gemini 2.5 Flash Audio Understanding API", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
    ASRModelInfo(name="deepgram-nova-2", family="Cloud Hosted API", display_name="Deepgram Nova-2", params="Cloud", vram="Cloud", speed_factor="~40x", languages="Multilingual (30+)", description="Deepgram Nova-2 speech-to-text API", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
    ASRModelInfo(name="elevenlabs-scribe-v2", family="Cloud Hosted API", display_name="ElevenLabs Scribe v2", params="Cloud", vram="Cloud", speed_factor="~35x", languages="Multilingual (99+)", description="ElevenLabs Scribe v2 diarized transcription API", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
    ASRModelInfo(name="amazon-transcribe", family="Cloud Hosted API", display_name="AWS Amazon Transcribe", params="Cloud", vram="Cloud", speed_factor="~20x", languages="Multilingual (100+)", description="AWS Amazon Transcribe batch & real-time ASR API", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
    ASRModelInfo(name="azure-speech-to-text", family="Cloud Hosted API", display_name="Microsoft Azure Speech", params="Cloud", vram="Cloud", speed_factor="~25x", languages="Multilingual (100+)", description="Microsoft Azure Cognitive Services Speech-to-Text API", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
    ASRModelInfo(name="groq-whisper-cloud", family="Cloud Hosted API", display_name="Groq Whisper Cloud (LPU)", params="Cloud (LPU)", vram="Cloud", speed_factor="~100x", languages="Multilingual (99+)", description="Groq Cloud LPU accelerated Whisper inference API", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
    ASRModelInfo(name="assemblyai-conformer-2", family="Cloud Hosted API", display_name="AssemblyAI Universal-2", params="Cloud", vram="Cloud", speed_factor="~30x", languages="Multilingual (99+)", description="AssemblyAI Universal-2 / Conformer-2 speech API", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
    ASRModelInfo(name="revai-speech-api", family="Cloud Hosted API", display_name="Rev.ai Speech API", params="Cloud", vram="Cloud", speed_factor="~25x", languages="Multilingual (30+)", description="Rev.ai enterprise speech recognition API", capabilities=["cloud"], implemented=False, is_local=False, is_cached=False),
]


def get_enriched_model_catalog() -> List[ASRModelInfo]:
    """Return model catalog with dynamic local server cache status evaluated at runtime."""
    catalog: List[ASRModelInfo] = []
    for m in MODEL_CATALOG:
        m_copy = m.model_copy()
        m_copy.is_cached = check_model_cached(m.name)
        catalog.append(m_copy)
    return catalog


