"""Common API DTO concepts only. RAG/HITL engines are explicitly unsupported."""
from typing import Annotated, Literal
from pydantic import Field
from .models import CamelModel


class ChatMessage(CamelModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=12000)


class BaseRequest(CamelModel):
    stream: bool = False
    user: str | None = None  # display-only, never an authenticated identity
    messages: list[ChatMessage] = Field(default_factory=list, max_length=100)


class FileInfo(CamelModel):
    data: str = Field(max_length=2_000_000)
    file_name: str = Field(max_length=200)


class Selectitem(CamelModel):
    type: Literal["select"] = "select"
    label: str
    sensitive: bool = False
    options: list[str] = Field(default_factory=list)


class Textbox(CamelModel):
    type: Literal["textbox"] = "textbox"
    placeholder: str = ""
    pattern: str | None = None


class Report(CamelModel):
    type: Literal["report"] = "report"
    title: str
    content: str


class HITL(CamelModel):
    content: Annotated[Selectitem | Textbox | Report, Field(discriminator="type")] | None = None


class RetrievalEvidence(CamelModel):
    source_id: str
    excerpt: str
    uri: str | None = None
    version: str | None = None


class RAGActionResponse(CamelModel):
    augmented_query: str = ""
    hyde_query: str = ""
    results: list[RetrievalEvidence] = Field(default_factory=list)


class ActionsResponseProtocol(CamelModel):
    actions: list[str] = Field(default_factory=list)
    references: list[RetrievalEvidence] = Field(default_factory=list)
    recommend_queries: list[str] = Field(default_factory=list)
