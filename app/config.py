# config.py
from confz import BaseConfig
from pydantic import BaseModel
from typing import Optional, Dict, List

class RoleMessage(BaseModel):
    role: str
    message: str

class LLMConfig(BaseModel):
    type: str

    class Config:
        extra = 'allow'

class PromptGroup(BaseModel):
    class Config:
        extra = 'allow'

class MetaPromptConfig(BaseConfig):
    llms: Optional[dict[str, LLMConfig]]
    aggressive_exploration: Optional[bool] = False
    examples_path: Optional[str]
    server_name: Optional[str] = None
    server_port: Optional[int] = None
    recursion_limit: Optional[int] = 25
    recursion_limit_max: Optional[int] = 50
    allow_flagging: Optional[bool] = False
    verbose: Optional[bool] = False
    max_output_age: Optional[int] = 3
    max_output_age_max: Optional[int] = 8
    prompt_templates: Optional[Dict[str, Dict[str, List[RoleMessage]]]] = None

    class Config:
        extra = 'allow'