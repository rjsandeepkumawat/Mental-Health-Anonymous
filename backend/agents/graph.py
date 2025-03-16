"""
Graph orchestration for the Mental Health Chatbot using Langgraph.

This module defines the flow logic for connecting the specialized agents.
"""
from typing import Dict, Any, Annotated, TypeVar, Literal
from langgraph.graph import StateGraph, END
from .state import ChatbotState

# Import the agents
from .triage_agent import TriageAgent
from .empathy_agent import EmpathyAgent
from .resource_agent import ResourceAgent
from .safety_agent import SafetyAgent
from .memory_agent import MemoryAgent

# Type for agent decisions
AgentDecision = Literal["triage", "empathy", "resource", "safety", "memory", "end"]

def should_run_safety_check(state: ChatbotState) -> AgentDecision:
    """Decide if safety check should be run."""
    # Always run safety check for new user inputs
    if state.current_user_input and not state.safety_check:
        return "safety"
    return "triage"

def route_based_on_triage(state: ChatbotState) -> AgentDecision:
    """Route to appropriate agent based on triage assessment."""
    # If safety check failed, end the conversation
    if state.safety_check and not state.safety_check.is_safe:
        if state.safety_check.needs_human_intervention:
            # End with a message about human intervention
            state.final_response = "I'm connecting you with a mental health professional who can better assist you."
            return "end"
        # Otherwise continue with a warning
        state.agent_responses["safety_warning"] = "Please note that I'm an AI assistant and not a substitute for professional mental health support."
    
    # Default routing logic based on current_agent from triage
    current_agent = state.current_agent
    if not current_agent:
        return "end"
        
    if current_agent == "empathy":
        return "empathy"
    elif current_agent == "resource":
        return "resource"
    elif current_agent == "memory":
        return "memory"
    else:
        return "end"

def should_end_conversation(state: ChatbotState) -> bool:
    """Determine if the conversation should end."""
    # End if we have a final response
    return state.final_response is not None

def create_agent_graph(
    triage_agent: TriageAgent,
    empathy_agent: EmpathyAgent,
    resource_agent: ResourceAgent,
    safety_agent: SafetyAgent,
    memory_agent: MemoryAgent,
) -> StateGraph:
    """Create the agent graph for the mental health chatbot.
    
    Args:
        triage_agent: The agent for triaging user inputs
        empathy_agent: The agent for providing empathetic responses
        resource_agent: The agent for providing mental health resources
        safety_agent: The agent for safety checks
        memory_agent: The agent for managing conversation context
        
    Returns:
        A Langgraph StateGraph for orchestrating the agents
    """
    # Create the graph
    graph = StateGraph(ChatbotState)
    
    # Add the nodes (agent functions)
    graph.add_node("safety", safety_agent)
    graph.add_node("triage", triage_agent)
    graph.add_node("empathy", empathy_agent)
    graph.add_node("resource", resource_agent)
    graph.add_node("memory", memory_agent)
    
    # Define the edges
    graph.add_edge("safety", "triage")
    graph.add_conditional_edges(
        "triage",
        route_based_on_triage,
        {
            "empathy": "empathy",
            "resource": "resource",
            "memory": "memory",
            "end": END
        }
    )
    graph.add_edge("empathy", "memory")
    graph.add_edge("resource", "memory")
    graph.add_edge("memory", END)
    
    # Set the entry point
    graph.set_entry_point(should_run_safety_check)
    
    return graph.compile() 