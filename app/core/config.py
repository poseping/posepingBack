from typing import Annotated, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "척추Ping API"
    app_env: str = "dev" # dev || production
    cors_origins: Annotated[List[str], NoDecode] = ["http://localhost:5173", "http://localhost:3000"]
    database_url: str = ""
    database_url_direct: str = ""
    secret_key: str = "change-me"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


settings = Settings()
