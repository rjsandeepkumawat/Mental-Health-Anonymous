"""
Mental Health Chatbot API

This is the main FastAPI application that serves the mental health chatbot.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import json
from dotenv import load_dotenv

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent classes
from agents import (
    TriageAgent, 
    EmpathyAgent, 
    ResourceAgent, 
    SafetyAgent, 
    MemoryAgent, 
    create_agent_graph,
    ChatbotState
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Chatbot API",
    description="API for a mental health chatbot using Langgraph for agent orchestration",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agents
triage_agent = TriageAgent(model_name="llm-triage")
empathy_agent = EmpathyAgent(model_name="empathy-llm")
resource_agent = ResourceAgent(model_name="resource-llm")
safety_agent = SafetyAgent(model_name="toxicity-moderator")
memory_agent = MemoryAgent(model_name="memory-manager")

# Create the agent graph
agent_graph = create_agent_graph(
    triage_agent=triage_agent,
    empathy_agent=empathy_agent,
    resource_agent=resource_agent,
    safety_agent=safety_agent,
    memory_agent=memory_agent
)

# Session storage (in-memory for demonstration)
# In production, use a proper database
sessions = {}

# Input/Output models
class ChatInput(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")

class ChatOutput(BaseModel):
    response: str = Field(..., description="Chatbot response")
    session_id: str = Field(..., description="Session ID for the conversation")
    resources: Optional[List[Dict[str, Any]]] = Field(None, description="Suggested resources")

@app.post("/chat", response_model=ChatOutput)
async def chat(chat_input: ChatInput):
    """Process a chat message and return a response."""
    message = chat_input.message
    session_id = chat_input.session_id
    
    # Get or create session state
    if session_id and session_id in sessions:
        # Continue existing conversation
        state = ChatbotState.model_validate(sessions[session_id])
    else:
        # Start new conversation
        state = ChatbotState()
        session_id = f"session_{len(sessions) + 1}"
    
    # Update state with user input
    state.current_user_input = message
    
    # Run the agent graph
    try:
        result = agent_graph.invoke(state.model_dump())
        result_state = ChatbotState.model_validate(result)
        
        # Store updated state
        sessions[session_id] = result_state.model_dump()
        
        # Prepare response
        response = result_state.final_response or "I'm not sure how to respond to that."
        
        # Extract resources
        resources = None
        if result_state.suggested_resources:
            resources = [resource.model_dump() for resource in result_state.suggested_resources]
            
        return {
            "response": response,
            "session_id": session_id,
            "resources": resources
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
