import torch
from transformers import (AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer)
import torch.nn.functional as F
from textblob import TextBlob  # Sentiment analysis for refined responses
import random  # For diverse conversational patterns 

# Load models and tokenizers
try:
    emotion_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    context_model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    dialogue_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

tokenizer_emotion = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
tokenizer_context = AutoTokenizer.from_pretrained("roberta-base")
tokenizer_dialogue = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Memory buffer to track conversation history
conversation_history = []

# Emotion detection with priority weighting
def detect_emotion(user_input):
    try:
        inputs = tokenizer_emotion(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = emotion_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
        emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        emotion_weights = {"sadness": 1.7, "anger": 1.4, "fear": 1.3, "joy": 1.2}  # Strengthened weight scaling
        weighted_probs = [probs[i] * emotion_weights.get(emotions[i], 1) for i in range(len(emotions))]
        return emotions[weighted_probs.index(max(weighted_probs))]
    except Exception:
        return "neutral"

# Sentiment analysis for enhanced emotional detection
def detect_sentiment(user_input):
    analysis = TextBlob(user_input)
    return "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"

# Context understanding with enhanced feedback
def analyze_context(user_input):
    try:
        inputs = tokenizer_context(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = context_model(**inputs)
        return random.choice([
            "I hear you.",
            "I understand where you're coming from.",
            "That's completely understandable. I'm here for you.",
            "Your feelings are valid, and I'm here to listen.",
            "I'm here with you through this."  # Enhanced variation
        ])
    except Exception:
        return "I understand. "

# Conversational response with improved coherence
def generate_response(user_input):
    global conversation_history
    conversation_history.append(user_input)
    conversation_history = conversation_history[-10:]  # Expanded conversation memory for better context

    combined_input = " ".join(conversation_history) + tokenizer_dialogue.eos_token
    try:
        inputs = tokenizer_dialogue.encode(combined_input, return_tensors="pt")
        response_ids = dialogue_model.generate(inputs, max_length=150, pad_token_id=tokenizer_dialogue.eos_token_id)
        return tokenizer_dialogue.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    except Exception:
        return "I'm here for you. Let's talk about what’s on your mind."

# Final response blending with refined templates
def get_response(user_input):
    emotion = detect_emotion(user_input)
    sentiment = detect_sentiment(user_input)
    context = analyze_context(user_input)
    dialogue_response = generate_response(user_input)

    emotion_responses = {
        "sadness": "I'm really sorry you're feeling this way. Let's take it one step at a time. Maybe talking it out can help?",
        "joy": "That's wonderful! Keep embracing those positive vibes.",
        "love": "Love is beautiful. Staying connected to those you care about can make all the difference.",
        "anger": "I get that you're upset. Taking a few deep breaths or stepping away might help clear your mind.",
        "fear": "Fear can feel overwhelming, but you’re stronger than you realize. Let's talk through it.",
        "surprise": "That sounds unexpected! Feel free to share more if you'd like.",
        "neutral": "I'm here to listen. No pressure, just take your time."
    }

    sentiment_responses = {
        "positive": "That's great to hear! Keep those good vibes flowing.",
        "negative": "I'm here for you. Let's talk it out when you're ready.",
        "neutral": "I'm here to support you however I can."
    }

    return f"{context} {emotion_responses.get(emotion, '')} {sentiment_responses.get(sentiment, '')} {dialogue_response}".strip()

# Chat loop with error handling
def chat_loop():
    print("Hello! I'm here to support your mental well-being. Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Bot: I'm here whenever you're ready to talk.")
            continue
        if user_input.lower() == 'quit':
            print("Take care! Remember, you're stronger than you think.")
            break
        response = get_response(user_input)
        print("Bot:", response)

if __name__ == '__main__':
    chat_loop()
