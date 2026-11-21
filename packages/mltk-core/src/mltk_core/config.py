"""Configuration base using pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class MltkBaseSettings(BaseSettings):
    """
    Base settings class for MLTK components.

    Supports:
    - Default attributes defined in class
    - YAML configuration loading
    - .env file support
    - Environment variables with prefix override
    """

    model_config = SettingsConfigDict(
        frozen=True,
        extra="ignore",
        env_prefix="MLTK_",
        env_file=".env",
        env_file_encoding="utf-8",
    )
