"""Pydantic models for request/response bodies — power the Swagger schemas."""
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# ------------------------------- request -------------------------------
class TranscribeRequest(BaseModel):
    """Body for enqueueing a recording."""
    model_config = ConfigDict(
        populate_by_name=True, extra="ignore",
        json_schema_extra={"example": {
            "Audio Path": "https://bucket.s3.amazonaws.com/call.wav",
            "engine": "auto", "language": "auto",
            "translate_to_english": True, "speakers": 2,
            "callback_url": "https://your-server.example.com/webhook",
        }},
    )
    audio_path: str = Field(
        validation_alias=AliasChoices("Audio Path", "audio_path", "audio_url", "AudioPath", "url"),
        description="Audio URL — presigned/public S3 or any https link. "
                    "Accepted keys: `Audio Path`, `audio_path`, `audio_url`, `url`.")
    callback_url: str | None = Field(
        None, description="Optional webhook. When the transcribe pass finishes we POST the "
                          "full result here (you can also just poll).")
    engine: str = Field(
        "auto", description="ASR model. `auto` → large-v3 (best). Or a size: "
                            "`small` | `medium` | `large-v3` | `large-v3-turbo`.")
    language: str = Field(
        "auto", description="Source language. `auto` → auto-detect. Or an ISO code (`hi`, `ml`, `en`, …).")
    translate_to_english: bool = Field(
        False, description="If true, add an English translation per turn (auto-skipped if the call is already English).")
    speakers: int | None = Field(
        None, description="Pin the number of speakers (e.g. `2` for a 1:1 call). Omit to auto-detect.")


# ------------------------------- responses -------------------------------
class CreateTaskResponse(BaseModel):
    task_id: str = Field(description="Use this id in every follow-up call.")
    status: str = Field(examples=["queued"])
    status_url: str = Field(description="Relative URL to poll for status.")
    message: str


class TaskStatus(BaseModel):
    """Polling payload."""
    task_id: str
    status: str = Field(description="queued | running | done | error | interrupted")
    stage: str | None = Field(None, description="Current stage, e.g. Transcribing, Diarizing, Translating, Done.")
    detail: str | None = Field(None, description="Sub-progress, e.g. '2:31 / 5:02 (50%)' or 'category 3/5'.")
    summary_status: str | None = Field(None, description="null | queued | running | done | error")
    analytics_status: str | None = Field(None, description="null | queued | running | done | error")
    error: str | None = None
    language: str | None = None
    duration_seconds: float | None = None
    transcript_ready: bool = False
    summary_ready: bool = False
    analytics_ready: bool = False


class SpeakerTurn(BaseModel):
    speaker: str = Field(examples=["Speaker 1"])
    role: str | None = Field(None, description="agent | customer | null (set after summarize infers roles).")
    start: float = Field(description="Turn start, seconds.")
    end: float = Field(description="Turn end, seconds.")
    text: str
    text_translated: str = Field("", description="Empty unless translation was requested.")


class TranscriptResponse(BaseModel):
    task_id: str
    status: str | None = None
    filename: str | None = None
    language: str | None = None
    translated_to: str | None = None
    duration_seconds: float | None = None
    roles: dict = Field(default_factory=dict, description="{agent, customer} once known.")
    speakers: dict = Field(default_factory=dict, description="Per-speaker talk time & share.")
    speaker_wise: list[SpeakerTurn] = Field(default_factory=list)
    full_text: str = ""
    full_text_translated: str = ""


class Sentiment(BaseModel):
    model_config = ConfigDict(extra="allow")
    overall: str | None = Field(None, examples=["positive", "neutral", "negative"])
    trend: str | None = None
    frustration_points: list[str] = Field(default_factory=list)


class Summary(BaseModel):
    model_config = ConfigDict(extra="allow")
    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    outcome: str | None = Field(None, examples=["Resolved", "Escalated", "Follow-up", "Unresolved"])
    customer_sentiment: Sentiment | None = None
    agent_tone: str | None = None
    roles: dict = Field(default_factory=dict)


class SummaryResponse(BaseModel):
    task_id: str
    summary_status: str | None = None
    summary: Summary = Field(default_factory=Summary)


class ScorecardItem(BaseModel):
    category: str | None = None
    checkpoint: str | None = None
    score: int | None = Field(None, description="1–5, or 0 for N/A.")
    verdict: str | None = Field(None, examples=["Met", "Partially Met", "Not Met", "N/A"])
    evidence: str | None = Field(None, description="Quote + [mm:ss] from the transcript.")
    suggestion: str | None = None


class Analytics(BaseModel):
    model_config = ConfigDict(extra="allow")
    overall_score: int | None = Field(None, description="Weighted 0–100. Compliance failures cap it.")
    raw_score: int | None = None
    compliance_fail: bool | None = None
    checkpoints_scored: int | None = None
    scorecard: list[ScorecardItem] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    task_id: str
    analytics_status: str | None = None
    analytics: Analytics = Field(default_factory=Analytics)


class TaskListItem(BaseModel):
    task_id: str
    filename: str | None = None
    status: str | None = None
    stage: str | None = None
    summary_status: str | None = None
    analytics_status: str | None = None
    created_at: str | None = None


class StageQueued(BaseModel):
    task_id: str
    summary_status: str | None = None
    analytics_status: str | None = None
    result_url: str


class ErrorResponse(BaseModel):
    detail: str
