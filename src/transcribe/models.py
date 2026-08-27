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
    params: str
    vram: str
    speed_factor: str
    languages: str
    description: str
    implemented: bool = True
    is_local: bool = True


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
    # Next-Gen Open ASR & Edge Architectures
    ASRModelInfo(name="moonshine-tiny", family="UsefulSensors Moonshine", params="150M", vram="~1 GB", speed_factor="~20x", languages="English Only", description="UsefulSensors/moonshine-tiny ultra-lightweight edge ONNX model"),
    ASRModelInfo(name="moonshine-base", family="UsefulSensors Moonshine", params="400M", vram="~2 GB", speed_factor="~10x", languages="English Only", description="UsefulSensors/moonshine-base high-accuracy edge ONNX model"),
    ASRModelInfo(name="sensevoice-small", family="Alibaba FunAudioLLM", params="230M", vram="~1.5 GB", speed_factor="~50x", languages="ZH / EN / JA / KO / YUE", description="FunAudioLLM/SenseVoiceSmall rich audio transcription (50x real-time) with SER & AED"),
    ASRModelInfo(name="meta-omnilingual-asr", family="Meta AI", params="300M", vram="~2 GB", speed_factor="~8x", languages="Multilingual (100+)", description="facebook/mms-1b-all universal speech recognition with adapter switching"),
]

# Deferred / Unimplemented Cloud-Hosted STT Models Section
CLOUD_MODEL_CATALOG: List[ASRModelInfo] = [
    ASRModelInfo(name="openai-whisper-api", family="Cloud Hosted API", params="Cloud (1550M)", vram="Cloud", speed_factor="~20x", languages="Multilingual (99+)", description="OpenAI Hosted Whisper API (whisper-1 / gpt-4o-transcribe)", implemented=False, is_local=False),
    ASRModelInfo(name="google-gemini-audio", family="Cloud Hosted API", params="Cloud (Gemini)", vram="Cloud", speed_factor="~25x", languages="Multilingual (100+)", description="Google Gemini 2.5 Flash Audio Understanding API", implemented=False, is_local=False),
    ASRModelInfo(name="deepgram-nova-2", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~40x", languages="Multilingual (30+)", description="Deepgram Nova-2 speech-to-text API", implemented=False, is_local=False),
    ASRModelInfo(name="elevenlabs-scribe-v2", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~35x", languages="Multilingual (99+)", description="ElevenLabs Scribe v2 diarized transcription API", implemented=False, is_local=False),
    ASRModelInfo(name="amazon-transcribe", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~20x", languages="Multilingual (100+)", description="AWS Amazon Transcribe batch & real-time ASR API", implemented=False, is_local=False),
    ASRModelInfo(name="azure-speech-to-text", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~25x", languages="Multilingual (100+)", description="Microsoft Azure Cognitive Services Speech-to-Text API", implemented=False, is_local=False),
    ASRModelInfo(name="groq-whisper-cloud", family="Cloud Hosted API", params="Cloud (LPU)", vram="Cloud", speed_factor="~100x", languages="Multilingual (99+)", description="Groq Cloud LPU accelerated Whisper inference API", implemented=False, is_local=False),
    ASRModelInfo(name="assemblyai-conformer-2", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~30x", languages="Multilingual (99+)", description="AssemblyAI Universal-2 / Conformer-2 speech API", implemented=False, is_local=False),
    ASRModelInfo(name="revai-speech-api", family="Cloud Hosted API", params="Cloud", vram="Cloud", speed_factor="~25x", languages="Multilingual (30+)", description="Rev.ai enterprise speech recognition API", implemented=False, is_local=False),
]


