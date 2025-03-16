"""
Memory Agent for the Mental Health Chatbot.

This agent is responsible for maintaining conversation context
and user information over time.
"""

from typing import Dict, Any, Optional, List
import datetime
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base_agent import BaseAgent
from .state import ChatbotState, Message, UserInfo

class MemoryAgent(BaseAgent):
    """Agent responsible for maintaining conversation context and user information."""
    
    def __init__(self, model_name: str = "memory-manager", parameters: Optional[Dict[str, Any]] = None):
        """Initialize the Memory Agent.
        
        Args:
            model_name: The name of the model to use
            parameters: Optional parameters for configuration
        """
        super().__init__(model_name, parameters)
        
    def initialize(self):
        """Initialize memory storage."""
        # In a real implementation, this might connect to a database
        pass
        
    def update_conversation_history(self, state: ChatbotState) -> ChatbotState:
        """Update the conversation history in the state.
        
        Args:
            state: The current chatbot state
            
        Returns:
            Updated state with conversation history
        """
        # Extract user input and create a user message
        if state.current_user_input:
            user_message = Message(
                role="user",
                content=state.current_user_input,
                metadata={
                    "timestamp": datetime.datetime.now().isoformat(),
                    "emotion": state.emotion_analysis.primary_emotion if state.emotion_analysis else None
                }
            )
            state.conversation.append(user_message)
            
        # Combine agent responses into a single assistant message
        response_content = ""
        
        # Add safety warning if present
        if "safety_warning" in state.agent_responses:
            response_content += state.agent_responses["safety_warning"] + "\n\n"
            
        # Add main response content
        if "empathy" in state.agent_responses:
            response_content += state.agent_responses["empathy"] + "\n\n"
            
        if "resource" in state.agent_responses:
            response_content += state.agent_responses["resource"]
            
        # If we have a response, add it to conversation
        if response_content:
            assistant_message = Message(
                role="assistant",
                content=response_content.strip(),
                metadata={
                    "timestamp": datetime.datetime.now().isoformat(),
                    "agents_used": list(state.agent_responses.keys())
                }
            )
            state.conversation.append(assistant_message)
            state.final_response = response_content.strip()
            
        return state
        
    def update_user_info(self, state: ChatbotState) -> ChatbotState:
        """Update user information based on the conversation.
        
        Args:
            state: The current chatbot state
            
        Returns:
            Updated state with user information
        """
        # Initialize user info if it doesn't exist
        if not state.user_info:
            state.user_info = UserInfo(user_id=f"user_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
            
        # Update user preferences based on conversation
        if state.emotion_analysis:
            # Track emotion trends over time
            current_emotion = state.emotion_analysis.primary_emotion
            if "emotion_history" not in state.user_info.preferences:
                state.user_info.preferences["emotion_history"] = []
                
            state.user_info.preferences["emotion_history"].append({
                "emotion": current_emotion,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        # Update risk factors if safety check was performed
        if state.safety_check:
            # Track risk level
            state.user_info.risk_factors["latest_risk_level"] = state.safety_check.risk_level
            state.user_info.risk_factors["latest_assessment"] = datetime.datetime.now().isoformat()
            
            # Track sensitive topics mentioned
            if state.safety_check.sensitive_topics:
                if "mentioned_topics" not in state.user_info.risk_factors:
                    state.user_info.risk_factors["mentioned_topics"] = set()
                    
                for topic in state.safety_check.sensitive_topics:
                    state.user_info.risk_factors["mentioned_topics"].add(topic)
                    
        return state
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state to update memory.
        
        Args:
            inputs: The input state to process
            
        Returns:
            Updated state with memory information
        """
        state = ChatbotState.model_validate(inputs)
        
        # Update conversation history
        state = self.update_conversation_history(state)
        
        # Update user information
        state = self.update_user_info(state)
        
        # Clean up the state for the next iteration
        if state.current_user_input:
            state.current_user_input = None
            
        state.safety_check = None
        state.emotion_analysis = None
        state.agent_responses = {}
        
        return state.model_dump() 