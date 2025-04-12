import streamlit as st
import json
import os
from groq import Groq
from typing import Generator

# Page Config
st.set_page_config(page_title="💬 MindEase - Mental Health Chatbot", page_icon="🧠", layout="centered")

# Feedback File
FEEDBACK_FILE = "chat_feedback.json"

# Ensure Feedback File Exists
def load_feedback_data():
    if not os.path.exists(FEEDBACK_FILE):
        data = {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "improvement_suggestions": [],
            "enhance_empathy": False
        }
        save_feedback_data(data)
        return data
    with open(FEEDBACK_FILE, "r") as file:
        return json.load(file)

def save_feedback_data(data):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(data, file, indent=4)

feedback_data = load_feedback_data()

# Initialize Client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": None,
        "moods": [],
        "topics": [],
        "tone_preference": "neutral"
    }

# --- Sidebar ---
BEST_MODEL = "llama-3.3-70b-versatile"
model_specs = {
    BEST_MODEL: {"name": "LLaMA3.3-70b", "temp": 0.4, "max_tokens": 1024}
}
st.sidebar.title("Settings")
st.sidebar.markdown("### 🧠 Best Model Selected Automatically")
st.sidebar.markdown(f"**Model:** {model_specs[BEST_MODEL]['name']}")
st.sidebar.markdown(f"**Temperature:** `{model_specs[BEST_MODEL]['temp']}`")
st.sidebar.markdown(f"**Max Tokens:** `{model_specs[BEST_MODEL]['max_tokens']}`")
st.sidebar.button("🗑 Clear Chat", on_click=lambda: st.session_state.clear())

# --- Title ---
st.markdown("<h1 style='text-align: center;'>💬 MindEase - Your Mental Health Chatbot</h1>", unsafe_allow_html=True)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar="🧑‍⚕️" if msg["role"] == "user" else "🤖"):
        st.markdown(f"**{'You' if msg['role'] == 'user' else 'MindEase'}:** {msg['content']}")

# Generator
def chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Extract profile data
def extract_profile_data(user_input):
    input_lower = user_input.lower()
    if "my name is" in input_lower:
        name = input_lower.split("my name is")[-1].strip().split(" ")[0]
        st.session_state.user_profile["name"] = name.capitalize()

    mood_keywords = ["happy", "sad", "anxious", "angry", "tired", "stressed", "lonely"]
    for mood in mood_keywords:
        if mood in input_lower:
            st.session_state.user_profile["moods"].append(mood)
            break

# Precise psychiatrist-style system prompt
def build_system_prompt():
    name = st.session_state.user_profile["name"]
    mood = st.session_state.user_profile["moods"][-1] if st.session_state.user_profile["moods"] else None
    tone = feedback_data["enhance_empathy"]

    base = (
        "You are MindEase, a licensed virtual psychiatrist from NirveonX Health. "
        "Your goal is to provide clear, topic-specific, evidence-based responses rooted in psychological science. "
        "Avoid vague or overly generalized advice. Engage the user with thoughtful, focused follow-up questions. "
        "Use a professional and compassionate tone. Apply concepts like CBT, mindfulness, and trauma-informed care if relevant. "
    )

    if name:
        base += f"You know the patient's name is {name}. Address them by name respectfully to build rapport. "
    if mood:
        base += f"The user feels {mood}. Adjust tone to acknowledge and respond to this emotional state. "
    if tone:
        base += "Feedback suggests enhancing emotional support. Use extra empathy and validation. "

    return [{"role": "system", "content": base}]

# Handle user input
if prompt := st.chat_input("How are you feeling today?"):
    extract_profile_data(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(f"**You:** {prompt}")

    full_response = ""
    try:
        chat_completion = client.chat.completions.create(
            model=BEST_MODEL,
            messages=build_system_prompt() + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=model_specs[BEST_MODEL]["max_tokens"],
            temperature=model_specs[BEST_MODEL]["temp"],
            stream=True
        )
        with st.chat_message("assistant", avatar="🤖"):
            full_response = st.write_stream(chat_responses(chat_completion))

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except:
        st.error("Something went wrong. Please try again later.", icon="🚨")

    # Emergency support check
    emergency_keywords = [
        "suicide", "end my life", "kill myself", "no reason to live", "giving up",
        "can't go on", "worthless", "hopeless", "i want to die", "i feel empty", "I want to hurt myself"
    ]
    if any(word in prompt.lower() for word in emergency_keywords):
        st.error("⚠️ You're not alone. Support is available.", icon="🚨")
        st.warning(
            "**Please talk to someone immediately.**\n\n"
            "📞 **KIRAN Helpline (India)**: `1800-599-0019` (24x7)\n"
            "🏥 Visit your nearest hospital or mental health clinic.\n"
            "We care about you. 💙",
            icon="🧠"
        )

# --- Feedback Section ---
st.sidebar.subheader("💬 Share your feedback")
feedback = st.sidebar.radio("Rate this chat experience:",
                            ["😡 Very Bad", "😞 Bad", "😐 Neutral", "😊 Good", "😍 Very Good"],
                            index=None)

if feedback:
    category = {
        "😡 Very Bad": "negative",
        "😞 Bad": "negative",
        "😐 Neutral": "neutral",
        "😊 Good": "positive",
        "😍 Very Good": "positive"
    }[feedback]
    feedback_data[category] += 1

    if category in ["neutral", "negative"]:
        st.sidebar.warning("We value your thoughts. Help us improve.")
        tip = st.sidebar.text_input("What could be better?", key="feedback_tip")
        if tip:
            feedback_data["improvement_suggestions"].append(tip)
            feedback_data["enhance_empathy"] = True
    else:
        feedback_data["enhance_empathy"] = False

    save_feedback_data(feedback_data)

# --- Feedback Summary ---
st.sidebar.subheader("📊 Feedback Summary")
st.sidebar.write(f"👍 Positive: {feedback_data['positive']}")
st.sidebar.write(f"😐 Neutral: {feedback_data['neutral']}")
st.sidebar.write(f"👎 Negative: {feedback_data['negative']}")

if feedback_data["improvement_suggestions"]:
    st.sidebar.subheader("📝 Suggestions")
    for s in feedback_data["improvement_suggestions"][-3:]:
        st.sidebar.write(f"- {s}")
