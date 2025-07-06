# types_/state.py

from typing import TypedDict, Optional, Dict, Any

class GraphState(TypedDict):
    frame_path: str
    frame_data: Dict[str, Any]           # base64 image + file name
    frame_features: Dict[str, Any]       # brightness, contrast, face count, etc.
    importance: str                      # "important" or "not_important"
    reason: str                          # reason from the LLM
    description: Optional[Dict[str, Any]]  # final output with extracted text + visual description
    next_step: str                       # used to navigate graph transitions
