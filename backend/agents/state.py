"""
Stating definition for the Mental Health Chatbot Langgraph.

This file defines the state structure used by the Langgraph orchestration.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    """A message in the conversation."""
    role: Literal["user", "assistant", "system"] = Field(...)
    content: str = Field(...)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class UserInfo(BaseModel):
    """Information about the user."""
    user_id: str = Field(...)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    risk_factors: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)

class EmotionAnalysis(BaseModel):
    """Emotion analysis of the user message."""
    primary_emotion: str = Field(...)
    confidence: float = Field(...)
    secondary_emotions: Dict[str, float] = Field(default_factory=dict)

class ResourceInfo(BaseModel):
    """Information about mental health resources."""
    type: str = Field(...)
    content: str = Field(...)
    source: Optional[str] = None
    url: Optional[str] = None

class SafetyCheck(BaseModel):
    """Safety check results."""
    is_safe: bool = Field(...)
    toxicity_score: float = Field(default=0.0)
    sensitive_topics: List[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = Field(default="low")
    needs_human_intervention: bool = Field(default=False)

class ChatbotState(BaseModel):
    """The complete state of the chatbot system."""
    conversation: List[Message] = Field(default_factory=list)
    current_user_input: Optional[str] = None
    current_agent: Optional[str] = None
    user_info: Optional[UserInfo] = None
    emotion_analysis: Optional[EmotionAnalysis] = None
    safety_check: Optional[SafetyCheck] = None
    suggested_resources: List[ResourceInfo] = Field(default_factory=list)
    agent_responses: Dict[str, str] = Field(default_factory=dict)
    final_response: Optional[str] = None 