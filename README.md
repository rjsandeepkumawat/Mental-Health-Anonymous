# Mental-Health-Anonymous

A mental health chatbot built using Langgraph for orchestrating multiple specialized agents.

## Project Structure

- `backend/`: Backend API and server logic
  - `agents/`: Contains all the Langgraph agents and their configurations
  - `database/`: Database models and connection handlers
  - `NLP/`: Natural Language Processing utilities
  - `utils/`: Helper functions and utilities
  - `app.py`: FastAPI application entry point

- `fronted/`: User interface components
  - `src/`: Source code for the frontend

- `ml_models/`: Machine learning models for specific tasks
  - `emotion_classifier/`: ML model for classifying emotions
  - `toxicity_moderator/`: ML model for moderating toxic content

- `deployment/`: Deployment configuration files

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file

## Usage

To run the backend server:
```
cd backend
python app.py
```

To run the frontend (instructions for the specific frontend framework will be added)

## Architecture

This project uses Langgraph to orchestrate multiple specialized agents:
- Triage Agent: Determines the best route for user queries
- Empathy Agent: Provides empathetic responses
- Resource Agent: Provides mental health resources and information
- Safety Agent: Ensures safety guidelines and monitors for risk indicators
- Memory Agent: Maintains conversation context and user preferences

The agents collaborate via Langgraph to provide a comprehensive mental health support system while maintaining user privacy and security.