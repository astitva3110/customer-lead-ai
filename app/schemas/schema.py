from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.helpers.channel import ALL_ORIGINS, CHANNELS, normalize_channel, normalize_origin


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    country: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    channel: str | None = None
    origin: str | None = None
    source: str | None = None
    phone: str | None = None
    user_name: str | None = None

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        channel = normalize_channel(value)
        if not channel:
            raise ValueError(f"channel must be one of: {', '.join(sorted(CHANNELS))}")
        return channel

    @field_validator("origin", "source")
    @classmethod
    def validate_origin(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        origin = normalize_origin(value)
        if not origin:
            raise ValueError(f"origin must be one of: {', '.join(sorted(ALL_ORIGINS))}")
        return origin


class SourceChunk(BaseModel):
    url: str
    title: str
    score: float


class ChatResponse(BaseModel):
    model_config = ConfigDict(exclude_none=True)

    conversation_id: str
    mode: str
    response: str
    answer: str
    sources: list[SourceChunk]
    debug_trace_id: str | None = None


class ChannelInboundResponse(BaseModel):
    accepted: bool = True
    results: list[ChatResponse] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str


class DocumentProgress(BaseModel):
    pages: int = 0
    chunks: int = 0
    embedded: int = 0


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    error: str | None = None
    progress: DocumentProgress = Field(default_factory=DocumentProgress)


class ChatLatency(BaseModel):
    guardrail_ms: float | None = None
    routing_ms: float | None = None
    rewrite_ms: float | None = None
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    tool_ms: float | None = None
    total_ms: float | None = None
    vector_ms: float | None = None
    keyword_ms: float | None = None
    merge_ms: float | None = None
    rerank_ms: float | None = None
    threshold_ms: float | None = None


class ChatLayerHit(BaseModel):
    rank: int
    chunk_id: str
    document_id: str = ""
    title: str = ""
    section_path: str = ""
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    combined_score: float | None = None
    original_retrieval_rank: int | None = None
    text_preview: str = ""


class ChatConversationSummary(BaseModel):
    conversation_id: str
    last_trace_id: str
    last_user_message: str
    last_response: str
    last_intent: str
    last_total_ms: float | None = None
    turn_count: int
    updated_at: str | None = None


class ChatListResponse(BaseModel):
    items: list[ChatConversationSummary]


class ChatTurnDetail(BaseModel):
    trace_id: str
    conversation_id: str
    user_message: str
    rewritten_query: str | None = None
    intent: str
    response: str
    tool_name: str | None = None
    tool_success: bool | None = None
    lead_id: str | None = None
    ticket_id: str | None = None
    backend: str | None = None
    reranker_name: str | None = None
    latency: ChatLatency
    layers: dict[str, list[ChatLayerHit]]
    created_at: str | None = None


class ChatDetailResponse(BaseModel):
    conversation_id: str
    turns: list[ChatTurnDetail]


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    user_id: str
    email: str
    role: str
    is_active: bool


class CreateUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    role: str = "user"


class UpdateUserRequest(BaseModel):
    email: str | None = Field(default=None, min_length=3, max_length=320)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserStatusUpdateRequest(BaseModel):
    is_active: bool


class UserRoleUpdateRequest(BaseModel):
    role: str


class LeadSummary(BaseModel):
    id: str
    name: str
    phone: str
    city: str
    product: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None


class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: str | None = None


class LeadDetail(LeadSummary):
    country: str
    conversation_id: str
    conversation: list[ConversationMessage] = Field(default_factory=list)


class LeadListResponse(BaseModel):
    items: list[LeadSummary]


class LeadUpdateRequest(BaseModel):
    status: str


class SupportSummary(BaseModel):
    id: str
    name: str
    phone: str
    product: str
    issue: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None


class SupportDetail(SupportSummary):
    conversation_id: str
    conversation: list[ConversationMessage] = Field(default_factory=list)


class SupportListResponse(BaseModel):
    items: list[SupportSummary]


class SupportUpdateRequest(BaseModel):
    status: str
