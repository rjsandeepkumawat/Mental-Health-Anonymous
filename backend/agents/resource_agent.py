"""
Resource Agent for the Mental Health Chatbot.

This agent is responsible for providing helpful mental health
resources and information.
"""

from typing import Dict, Any, Optional, List
import re
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base_agent import BaseAgent
from .state import ChatbotState, ResourceInfo

class ResourceAgent(BaseAgent):
    """Agent responsible for providing mental health resources."""
    
    def __init__(self, model_name: str = "resource-llm", parameters: Optional[Dict[str, Any]] = None):
        """Initialize the Resource Agent.
        
        Args:
            model_name: The name of the language model to use
            parameters: Optional parameters for the language model
        """
        super().__init__(model_name, parameters)
        
    def initialize(self):
        """Initialize the resource database."""
        # Define mental health resources
        self.resources = {
            "anxiety": [
                ResourceInfo(
                    type="article",
                    content="Understanding and Managing Anxiety",
                    source="National Institute of Mental Health",
                    url="https://www.nimh.nih.gov/health/topics/anxiety-disorders"
                ),
                ResourceInfo(
                    type="technique",
                    content="Deep breathing: Breathe in for 4 counts, hold for 2, and exhale for 6 counts. Repeat 5-10 times.",
                    source="Anxiety and Depression Association of America",
                    url="https://adaa.org/tips"
                )
            ],
            "depression": [
                ResourceInfo(
                    type="article",
                    content="Depression Basics and Treatment Options",
                    source="National Institute of Mental Health",
                    url="https://www.nimh.nih.gov/health/topics/depression"
                ),
                ResourceInfo(
                    type="technique",
                    content="Activity scheduling: Plan enjoyable activities throughout your week, even if you don't feel motivated initially.",
                    source="American Psychological Association",
                    url="https://www.apa.org/depression"
                )
            ],
            "stress": [
                ResourceInfo(
                    type="article",
                    content="Stress Management Techniques",
                    source="Mayo Clinic",
                    url="https://www.mayoclinic.org/healthy-lifestyle/stress-management/basics/stress-basics/hlv-20049495"
                ),
                ResourceInfo(
                    type="technique",
                    content="Progressive muscle relaxation: Tense and then relax each muscle group in your body, starting from your toes and working upward.",
                    source="American Psychological Association",
                    url="https://www.apa.org/topics/stress"
                )
            ],
            "crisis": [
                ResourceInfo(
                    type="hotline",
                    content="National Suicide Prevention Lifeline: 988 or 1-800-273-8255",
                    source="SAMHSA",
                    url="https://988lifeline.org/"
                ),
                ResourceInfo(
                    type="text_line",
                    content="Crisis Text Line: Text HOME to 741741",
                    source="Crisis Text Line",
                    url="https://www.crisistextline.org/"
                )
            ],
            "general": [
                ResourceInfo(
                    type="article",
                    content="Taking Care of Your Mental Health",
                    source="Mental Health America",
                    url="https://www.mhanational.org/taking-good-care-yourself"
                ),
                ResourceInfo(
                    type="app",
                    content="Mindfulness and meditation apps like Headspace, Calm, or Insight Timer",
                    source="Various",
                    url="https://www.mindful.org/free-mindfulness-apps-worthy-of-your-attention/"
                )
            ]
        }
        
        # Define keywords for each category
        self.keywords = {
            "anxiety": ["anxiety", "anxious", "worry", "panic", "fear", "stressed", "nervous"],
            "depression": ["depression", "depressed", "sad", "hopeless", "unmotivated", "exhausted", "worthless"],
            "stress": ["stress", "overwhelmed", "burnout", "pressure", "overworked", "tense"],
            "crisis": ["suicidal", "crisis", "emergency", "harm", "unsafe", "danger"],
            "general": []  # Fallback category
        }
        
    def match_category(self, text: str) -> str:
        """Match the user's query to a resource category.
        
        Args:
            text: The user's query text
            
        Returns:
            The matched category
        """
        text = text.lower()
        
        # Check each category for keyword matches
        for category, keywords in self.keywords.items():
            if any(keyword in text for keyword in keywords):
                return category
                
        # Default to general resources
        return "general"
        
    def get_resources(self, category: str, max_resources: int = 2) -> List[ResourceInfo]:
        """Get resources for a specific category.
        
        Args:
            category: The resource category
            max_resources: Maximum number of resources to return
            
        Returns:
            A list of ResourceInfo objects
        """
        # Get resources for the category or default to general
        resources = self.resources.get(category, self.resources["general"])
        
        # Limit the number of resources
        return resources[:max_resources]
        
    def format_response(self, resources: List[ResourceInfo]) -> str:
        """Format resources into a user-friendly response.
        
        Args:
            resources: List of ResourceInfo objects
            
        Returns:
            Formatted response text
        """
        if not resources:
            return "I don't have specific resources for that topic right now, but I'm here to listen and support you."
            
        response = "Here are some resources that might help:\n\n"
        
        for i, resource in enumerate(resources, 1):
            response += f"{i}. {resource.content}"
            if resource.source:
                response += f" (Source: {resource.source})"
            if resource.url:
                response += f"\n   {resource.url}"
            response += "\n\n"
            
        response += "Would you like more information on any of these resources, or is there something else I can help with?"
        
        return response
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user input to provide relevant resources.
        
        Args:
            inputs: The input state containing user message
            
        Returns:
            Updated state with resource information
        """
        state = ChatbotState.model_validate(inputs)
        user_input = state.current_user_input
        
        if not user_input:
            # Cannot provide resources without input
            return state.model_dump()
            
        # Match the category based on user input
        category = self.match_category(user_input)
        
        # Get relevant resources
        resources = self.get_resources(category)
        
        # Format the response
        response = self.format_response(resources)
        
        # Update the state
        state.agent_responses["resource"] = response
        state.suggested_resources = resources
        
        return state.model_dump() 