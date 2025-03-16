"""
Triage Agent for the Mental Health Chatbot.

This agent is responsible for determining which specialized agent
should handle the user query.
"""

from typing import Dict, Any, Optional, List
import sys
import os

# Add the project root to sys.path to import ml_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base_agent import BaseAgent
from .state import ChatbotState, EmotionAnalysis
from ml_models.emotion_classifier import EmotionClassifier  # This would be your emotion model

class TriageAgent(BaseAgent):
    """Agent responsible for triaging user queries to the appropriate agent."""
    
    def __init__(self, model_name: str = "llm-triage", parameters: Optional[Dict[str, Any]] = None):
        """Initialize the Triage Agent.
        
        Args:
            model_name: The name of the language model to use
            parameters: Optional parameters for the language model
        """
        super().__init__(model_name, parameters)
        
    def initialize(self):
        """Initialize the emotion classifier and other resources."""
        # Initialize emotion classifier
        self.emotion_classifier = EmotionClassifier()
        
        # Define categories of queries
        self.info_seeking_keywords = [
            "what is", "how do I", "resources", "help for", 
            "information", "advice", "symptoms", "coping with"
        ]
        
        self.emotional_support_keywords = [
            "feeling", "sad", "anxious", "depressed", "worried",
            "stressed", "overwhelmed", "lonely", "afraid", "scared"
        ]
        
    def classify_emotion(self, text: str) -> EmotionAnalysis:
        """Classify the emotion in the user's text.
        
        Args:
            text: The user input text
            
        Returns:
            Emotion analysis result
        """
        emotion_results = self.emotion_classifier.classify(text)
        
        return EmotionAnalysis(
            primary_emotion=emotion_results["primary_emotion"],
            confidence=emotion_results["confidence"],
            secondary_emotions=emotion_results["secondary_emotions"]
        )
        
    def determine_agent(self, text: str, emotion_analysis: EmotionAnalysis) -> str:
        """Determine which agent should handle the query.
        
        Args:
            text: The user input text
            emotion_analysis: The emotion analysis results
            
        Returns:
            The name of the agent that should handle the query
        """
        # Check if this is an information-seeking query
        if any(keyword.lower() in text.lower() for keyword in self.info_seeking_keywords):
            return "resource"
            
        # If strong emotional content, route to empathy agent
        if emotion_analysis.confidence > 0.7 and emotion_analysis.primary_emotion in [
            "sadness", "anxiety", "fear", "anger", "grief"
        ]:
            return "empathy"
            
        # Default to empathy agent for most conversations
        return "empathy"
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user input to determine routing.
        
        Args:
            inputs: The input state containing user message
            
        Returns:
            Updated state with routing information
        """
        state = ChatbotState.model_validate(inputs)
        user_input = state.current_user_input
        
        if not user_input:
            # No user input to process
            return state.model_dump()
            
        # Analyze emotions
        emotion_analysis = self.classify_emotion(user_input)
        state.emotion_analysis = emotion_analysis
        
        # Determine which agent should handle the query
        agent = self.determine_agent(user_input, emotion_analysis)
        state.current_agent = agent
        
        return state.model_dump() 