"""Data structures and schemas for transcription and diarization."""

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


class TranscriptionResult(BaseModel):
    """Complete result container."""

    language: str
    language_probability: float = 1.0
    duration: float
    segments: List[DiarizedSegment] = Field(default_factory=list)
    speakers: List[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Formatted full text with speaker tags."""
        return "\n".join(
            f"[{seg.speaker}] ({seg.start:.2f}s - {seg.end:.2f}s): {seg.text.strip()}"
            for seg in self.segments
        )


class ASRModelInfo(BaseModel):
    """Metadata schema for available ASR/STT models."""

    name: str
    family: str
    params: str
    vram: str
    speed_factor: str
    languages: str
    description: str


MODEL_CATALOG: List[ASRModelInfo] = [
    ASRModelInfo(name="tiny", family="Whisper Standard", params="39M", vram="~1 GB", speed_factor="~32x", languages="Multilingual (99+)", description="Fastest multilingual model with minimal resource requirements"),
    ASRModelInfo(name="tiny.en", family="Whisper English", params="39M", vram="~1 GB", speed_factor="~32x", languages="English Only", description="Optimized English-only tiny model"),
    ASRModelInfo(name="base", family="Whisper Standard", params="74M", vram="~1 GB", speed_factor="~16x", languages="Multilingual (99+)", description="Default balanced standard model for fast general-purpose transcription"),
    ASRModelInfo(name="base.en", family="Whisper English", params="74M", vram="~1 GB", speed_factor="~16x", languages="English Only", description="Optimized English-only base model"),
    ASRModelInfo(name="small", family="Whisper Standard", params="244M", vram="~2 GB", speed_factor="~6x", languages="Multilingual (99+)", description="High accuracy model balanced for moderate hardware"),
    ASRModelInfo(name="small.en", family="Whisper English", params="244M", vram="~2 GB", speed_factor="~6x", languages="English Only", description="Optimized English-only small model"),
    ASRModelInfo(name="distil-small.en", family="Distil-Whisper", params="166M", vram="~1.5 GB", speed_factor="~10x", languages="English Only", description="Knowledge-distilled small model, 5x faster than small with near-zero WER loss"),
    ASRModelInfo(name="medium", family="Whisper Standard", params="769M", vram="~5 GB", speed_factor="~2x", languages="Multilingual (99+)", description="Near production-grade multilingual transcription accuracy"),
    ASRModelInfo(name="medium.en", family="Whisper English", params="769M", vram="~5 GB", speed_factor="~2x", languages="English Only", description="Optimized English-only medium model"),
    ASRModelInfo(name="distil-medium.en", family="Distil-Whisper", params="394M", vram="~3 GB", speed_factor="~5x", languages="English Only", description="Knowledge-distilled medium model for high-speed English ASR"),
    ASRModelInfo(name="large-v1", family="Whisper Standard", params="1550M", vram="~10 GB", speed_factor="~1x", languages="Multilingual (99+)", description="First-generation large Whisper checkpoint"),
    ASRModelInfo(name="large-v2", family="Whisper Standard", params="1550M", vram="~10 GB", speed_factor="~1x", languages="Multilingual (99+)", description="Second-generation large Whisper checkpoint with improved audio encoding"),
    ASRModelInfo(name="large-v3", family="Whisper Standard", params="1550M", vram="~10 GB", speed_factor="~1x", languages="Multilingual (99+)", description="State-of-the-art multilingual Whisper model with 128 Mel frequency bins"),
    ASRModelInfo(name="large-v3-turbo", family="Whisper Turbo", params="809M", vram="~6 GB", speed_factor="~8x", languages="Multilingual (99+)", description="Pruned 4-decoder large-v3 architecture; 8x faster with large-v3 accuracy"),
    ASRModelInfo(name="turbo", family="Whisper Turbo", params="809M", vram="~6 GB", speed_factor="~8x", languages="Multilingual (99+)", description="Alias for large-v3-turbo"),
    ASRModelInfo(name="distil-large-v2", family="Distil-Whisper", params="756M", vram="~5 GB", speed_factor="~6x", languages="English & Multi", description="Distilled large-v2 model with 6x real-time speedup"),
    ASRModelInfo(name="distil-large-v3", family="Distil-Whisper", params="756M", vram="~5 GB", speed_factor="~6x", languages="English & Multi", description="Distilled large-v3 model with state-of-the-art distilled accuracy"),
    # Indonesian Specialized Fine-tunes (from stt-arena-demo-2026)
    ASRModelInfo(name="cahya-whisper-tiny-id", family="Indonesian Fine-tune", params="39M", vram="~1 GB", speed_factor="~32x", languages="Indonesian (id)", description="Specialized Indonesian fine-tune (cahya/whisper-tiny-id)"),
    ASRModelInfo(name="cahya-whisper-base-id", family="Indonesian Fine-tune", params="74M", vram="~1 GB", speed_factor="~16x", languages="Indonesian (id)", description="Specialized Indonesian fine-tune (cahya/whisper-base-id)"),
    ASRModelInfo(name="cahya-whisper-small-id", family="Indonesian Fine-tune", params="244M", vram="~2 GB", speed_factor="~6x", languages="Indonesian (id)", description="Specialized Indonesian fine-tune (cahya/whisper-small-id)"),
    ASRModelInfo(name="cahya-whisper-medium-id", family="Indonesian Fine-tune", params="769M", vram="~5 GB", speed_factor="~2x", languages="Indonesian (id)", description="Specialized Indonesian fine-tune (cahya/whisper-medium-id)"),
    ASRModelInfo(name="cahya-faster-whisper-medium-id", family="Indonesian Fine-tune", params="769M", vram="~5 GB", speed_factor="~3x", languages="Indonesian (id)", description="CTranslate2 converted Indonesian checkpoint for faster inference"),
    ASRModelInfo(name="indonesian-wav2vec2-regional", family="Indonesian Wav2Vec2", params="317M", vram="~2 GB", speed_factor="~15x", languages="ID / JV / SU", description="indonesian-nlp/wav2vec2-indonesian-javanese-sundanese acoustic CTC"),
    ASRModelInfo(name="indonesian-wav2vec2-large-xlsr", family="Indonesian Wav2Vec2", params="317M", vram="~2 GB", speed_factor="~15x", languages="Indonesian (id)", description="indonesian-nlp/wav2vec2-large-xlsr-indonesian acoustic CTC"),
    # Next-Gen Open ASR Architectures (from stt-arena-demo-2026)
    ASRModelInfo(name="moonshine-base", family="UsefulSensors Moonshine", params="400M", vram="~2 GB", speed_factor="~10x", languages="English Only", description="UsefulSensors/moonshine-base ultra-fast edge speech recognition"),
    ASRModelInfo(name="nvidia-parakeet-tdt-v3", family="NVIDIA NeMo", params="600M", vram="~4 GB", speed_factor="~12x", languages="Multilingual (25 EU)", description="nvidia/parakeet-tdt-0.6b-v3 FastConformer-TDT architecture"),
    ASRModelInfo(name="kyutai-stt", family="Kyutai Moshi", params="1000M", vram="~6 GB", speed_factor="~4x", languages="EN / FR", description="kyutai/stt-1b-en_fr real-time duplex speech recognition"),
    ASRModelInfo(name="meta-omnilingual-asr", family="Meta AI", params="300M", vram="~2 GB", speed_factor="~8x", languages="Multilingual (100+)", description="facebook/omnilingual-asr universal speech recognition"),
    ASRModelInfo(name="voxtral-mini-3b", family="Mistral Voxtral", params="3000M", vram="~12 GB", speed_factor="~1x", languages="Multilingual (EN/FR/ES/DE/IT)", description="mistralai/Voxtral-Mini-3B-2507 edge speech model"),
    ASRModelInfo(name="gemma-3n-audio", family="Google Gemma", params="2000M", vram="~8 GB", speed_factor="~2x", languages="Multilingual", description="google/gemma-3n-E2B-it edge audio understanding (gated on HF)"),
    # Cloud STT APIs (Reference Integrations)
    ASRModelInfo(name="openai-whisper-api", family="Cloud Hosted API", params="Cloud (1550M)", vram="Cloud", speed_factor="~20x", languages="Multilingual (99+)", description="OpenAI Hosted Whisper API (whisper-1 / gpt-4o-transcribe)"),
    ASRModelInfo(name="google-omni-api", family="Cloud Hosted API", params="Cloud (Gemini)", vram="Cloud", speed_factor="~25x", languages="Multilingual (100+)", description="Google Gemini 2.5 Flash Audio Understanding API"),
    ASRModelInfo(name="deepgram-nova-2", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~40x", languages="Multilingual (30+)", description="Deepgram Nova-2 high-speed speech-to-text API"),
    ASRModelInfo(name="elevenlabs-scribe-v2", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~35x", languages="Multilingual (99+)", description="ElevenLabs Scribe v2 SOTA diarized transcription API"),
    ASRModelInfo(name="amazon-transcribe", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~20x", languages="Multilingual (100+)", description="AWS Amazon Transcribe real-time & batch ASR API"),
    # Newly Discovered Open Architectures (YouTube Playlist Analysis)
    ASRModelInfo(name="sensevoice-small", family="Alibaba FunAudioLLM", params="230M", vram="~1.5 GB", speed_factor="~50x", languages="ZH / EN / JA / KO / YUE", description="FunAudioLLM/SenseVoiceSmall rich audio transcription (50x real-time)"),
    ASRModelInfo(name="nvidia-nemotron-speech-asr", family="NVIDIA NeMo", params="600M", vram="~4 GB", speed_factor="~15x", languages="Multilingual (25+)", description="NVIDIA Cache-Aware low-latency streaming RNN-T ASR"),
    ASRModelInfo(name="microsoft-vibevoice-asr", family="Microsoft", params="1200M", vram="~6 GB", speed_factor="~8x", languages="Multilingual (50+)", description="Microsoft VibeVoice 60-minute long-form single pass acoustic ASR"),
    ASRModelInfo(name="moss-sats-diarized-asr", family="Fudan OpenMOSS", params="800M", vram="~5 GB", speed_factor="~6x", languages="Multilingual", description="Speech-Aware Target-Speaker ASR with integrated speaker diarization"),
    ASRModelInfo(name="qwen3-audio-stt", family="Alibaba Qwen", params="3000M", vram="~10 GB", speed_factor="~3x", languages="Multilingual (100+)", description="Alibaba Qwen3 Audio speech understanding and ASR"),
    ASRModelInfo(name="tencent-covo-audio-7b", family="Tencent", params="7000M", vram="~16 GB", speed_factor="~1x", languages="ZH / EN", description="Tencent Covo-Audio 7B end-to-end open voice AI model"),
]


