from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Package lives in evolve/; repo root is parent (RaiderEvolve/.env).
_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    LLM credentials: OPENAI_API_KEY or LLM_API_KEY.

    Loads ``.env`` from the current working directory first, then from the repo root
    (so keys still work if ``uvicorn`` is started from another directory).
    Later files override earlier ones.
    """

    model_config = SettingsConfigDict(
        env_file=(Path.cwd() / ".env", _REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_API_KEY"),
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL"),
    )
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    @model_validator(mode="after")
    def _normalize_api_key(self) -> Settings:
        k = (self.openai_api_key or "").strip()
        if not k or k.upper() == "YOUR_API_KEY":
            self.openai_api_key = ""
        else:
            self.openai_api_key = k
        self.openai_base_url = (self.openai_base_url or "").strip().rstrip("/") or "https://api.openai.com/v1"
        return self


settings = Settings()
