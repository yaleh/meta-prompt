# config.py
from confz import BaseConfig
from pydantic import BaseModel, Extra
from typing import Optional

class LLMConfig(BaseModel):
    type: str

    class Config:
        extra = Extra.allow

class MetaPromptConfig(BaseConfig):
    llms: Optional[dict[str, LLMConfig]]
    examples_path: Optional[str]
    server_name: Optional[str] = '127.0.0.1'
    server_port: Optional[int] = 7878