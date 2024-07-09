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
    server_name: Optional[str] = None
    server_port: Optional[int] = None
    recursion_limit: Optional[int] = 25
    recursion_limit_max: Optional[int] = 50
    allow_flagging: Optional[bool] = False
    verbose: Optional[bool] = False
    max_output_age: Optional[int] = 3
    max_output_age_max: Optional[int] = 8