from pydantic import BaseModel


class AnalysisSchema(BaseModel):
    category: str | None
    confidence: float | None
    explanation: str | None
    keywords: list[str] | None
    ambiguities: list[dict[str, str]] | None
