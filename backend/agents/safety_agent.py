"""
Safety Agent for the Mental Health Chatbot.

This agent is responsible for safety checks on user input.
"""

from typing import Dict, Any, Optional, List
import sys
import os

# Add the project root to sys.path to import ml_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base_agent import BaseAgent
from .state import ChatbotState, SafetyCheck
from ml_models.toxicity_moderator import ToxicityModerator  # This would be your toxicity model

class SafetyAgent(BaseAgent):
    """Agent responsible for safety checks on user input."""
    
    def __init__(self, model_name: str = "toxicity-moderator", parameters: Optional[Dict[str, Any]] = None):
        """Initialize the Safety Agent.
        
        Args:
            model_name: The name of the toxicity model to use
            parameters: Optional parameters for the toxicity model
        """
        super().__init__(model_name, parameters)
        
    def initialize(self):
        """Initialize the toxicity moderator."""
        # Initialize toxicity moderator
        self.toxicity_moderator = ToxicityModerator()
        
        # Define sensitive topics and risk patterns
        self.sensitive_topics = [
            "suicide", "self-harm", "violence", "abuse", 
            "drugs", "alcohol", "eating disorders"
        ]
        
        # Define high-risk keywords that might need intervention
        self.high_risk_keywords = [
            "kill myself", "end my life", "want to die", 
            "suicide plan", "hurt myself", "self-harm"
        ]
        
    def detect_sensitive_topics(self, text: str) -> List[str]:
        """Detect sensitive topics in the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A list of detected sensitive topics
        """
        return [topic for topic in self.sensitive_topics if topic.lower() in text.lower()]
    
    def assess_risk_level(self, text: str, toxicity_score: float) -> Dict[str, Any]:
        """Assess the risk level of the user input.
        
        Args:
            text: The user input text
            toxicity_score: The toxicity score from the moderator
            
        Returns:
            A dictionary with risk assessment information
        """
        needs_intervention = any(keyword.lower() in text.lower() for keyword in self.high_risk_keywords)
        
        if needs_intervention or toxicity_score > 0.8:
            risk_level = "high"
        elif toxicity_score > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
            
        return {
            "risk_level": risk_level,
            "needs_human_intervention": needs_intervention
        }
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user input for safety concerns.
        
        Args:
            inputs: The input state containing user message
            
        Returns:
            Updated state with safety check results
        """
        state = ChatbotState.model_validate(inputs)
        user_input = state.current_user_input
        
        if not user_input:
            # No user input to check
            state.safety_check = SafetyCheck(is_safe=True, toxicity_score=0.0)
            return state.model_dump()
        
        # Check toxicity
        toxicity_score = self.toxicity_moderator.check_toxicity(user_input)
        
        # Detect sensitive topics
        sensitive_topics = self.detect_sensitive_topics(user_input)
        
        # Assess risk
        risk_assessment = self.assess_risk_level(user_input, toxicity_score)
        
        # Create safety check
        safety_check = SafetyCheck(
            is_safe=toxicity_score < 0.7 and risk_assessment["risk_level"] != "high",
            toxicity_score=toxicity_score,
            sensitive_topics=sensitive_topics,
            risk_level=risk_assessment["risk_level"],
            needs_human_intervention=risk_assessment["needs_human_intervention"]
        )
        
        # Update the state
        state.safety_check = safety_check
        
        return state.model_dump() 