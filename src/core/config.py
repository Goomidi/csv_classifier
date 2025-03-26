from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LITE_LLM_BASE_URL: str

    OPENAI_API_KEY: str

    MODEL: str

    model_config = SettingsConfigDict(
        env_file=[
            ".env",
            ".env.example",
        ],
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
