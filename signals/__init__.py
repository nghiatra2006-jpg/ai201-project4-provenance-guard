# Signals package — exports the two classification functions
from .llm_signal import groq_classify
from .stylometric_signal import stylometric_classify

__all__ = ["groq_classify", "stylometric_classify"]
