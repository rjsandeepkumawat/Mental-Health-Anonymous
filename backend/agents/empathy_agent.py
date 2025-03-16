"""
Empathy Agent for the Mental Health Chatbot.

This agent is responsible for providing empathetic responses to users.
"""

from typing import Dict, Any, Optional, List
import re
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base_agent import BaseAgent
from .state import ChatbotState

class EmpathyAgent(BaseAgent):
    """Agent responsible for providing empathetic responses."""
    
    def __init__(self, model_name: str = "empathy-llm", parameters: Optional[Dict[str, Any]] = None):
        """Initialize the Empathy Agent.
        
        Args:
            model_name: The name of the language model to use
            parameters: Optional parameters for the language model
        """
        super().__init__(model_name, parameters)
        
    def initialize(self):
        """Initialize resources for the empathy agent."""
        # Define empathetic templates based on emotions
        self.empathy_templates = {
            "sadness": [
                "I'm sorry you're feeling sad. It's okay to feel this way, and I'm here to listen.",
                "That sounds really difficult. I can understand why you might feel sad about that."
            ],
            "anxiety": [
                "I hear that you're feeling anxious. That's a natural response, though I know it's not easy.",
                "Anxiety can be really challenging to deal with. I'm here to support you through this."
            ],
            "anger": [
                "I can sense your frustration. It's completely valid to feel this way given what you've shared.",
                "That situation would make many people feel angry. Thank you for sharing how you feel."
            ],
            "fear": [
                "Being afraid in this situation is completely understandable. You're not alone in feeling this way.",
                "I can imagine that must be scary to deal with. I'm here to listen and support you."
            ],
            "joy": [
                "I'm really happy to hear that! It's wonderful that you've had this positive experience.",
                "That's great news! Those moments of joy are worth celebrating."
            ],
            "default": [
                "Thank you for sharing that with me. I'm here to listen and support you.",
                "I appreciate you opening up about this. How else can I support you right now?"
            ]
        }
        
    def create_empathetic_response(self, emotion: str, user_input: str) -> str:
        """Create an empathetic response based on detected emotion.
        
        Args:
            emotion: The primary emotion detected
            user_input: The user's message
            
        Returns:
            An empathetic response
        """
        import random
        
        # Get templates for the emotion or use default
        templates = self.empathy_templates.get(emotion.lower(), self.empathy_templates["default"])
        
        # Choose a random template
        response_template = random.choice(templates)
        
        # Personalize response if possible
        name_match = re.search(r"My name is (\w+)", user_input)
        if name_match:
            name = name_match.group(1)
            response_template = f"Hi {name}, " + response_template
            
        # Add follow-up question to encourage continued conversation
        follow_ups = [
            "Would you like to tell me more about how you're feeling?",
            "Is there anything specific about this that's been on your mind?",
            "What do you think might help you feel better right now?",
            "Have you tried any coping strategies that have worked for you in the past?"
        ]
        
        response = response_template + " " + random.choice(follow_ups)
        
        return response
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user input to generate an empathetic response.
        
        Args:
            inputs: The input state containing user message and emotion analysis
            
        Returns:
            Updated state with empathetic response
        """
        state = ChatbotState.model_validate(inputs)
        user_input = state.current_user_input
        
        if not user_input or not state.emotion_analysis:
            # Cannot generate empathetic response without input and emotion analysis
            return state.model_dump()
            
        # Get the primary emotion
        primary_emotion = state.emotion_analysis.primary_emotion
        
        # Create empathetic response
        response = self.create_empathetic_response(primary_emotion, user_input)
        
        # Update the state
        state.agent_responses["empathy"] = response
        
        return state.model_dump() 