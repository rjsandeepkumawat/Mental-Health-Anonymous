"""
Base Agent class for the Mental Health Chatbot

This serves as the foundation for all specialized agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseAgent(ABC):
    """Base agent class that all specialized agents must inherit from."""
    
    def __init__(self, model_name: str, parameters: Optional[Dict[str, Any]] = None):
        """Initialize the base agent.
        
        Args:
            model_name: The name of the model to use for this agent
            parameters: Optional parameters for model configuration
        """
        self.model_name = model_name
        self.parameters = parameters or {}
        self.initialize()
        
    def initialize(self):
        """Initialize any resources needed by the agent."""
        pass
        
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the inputs and generate a response.
        
        Args:
            inputs: A dictionary containing the input data
            
        Returns:
            A dictionary containing the processed output
        """
        pass
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make the agent callable.
        
        Args:
            inputs: A dictionary containing the input data
            
        Returns:
            A dictionary containing the processed output
        """
        return self.process(inputs) 