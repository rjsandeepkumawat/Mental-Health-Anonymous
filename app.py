import streamlit as st
import json
from typing import Generator
from groq import Groq

# Set Page Config
st.set_page_config(page_title="💬 MindEase Chatbot", page_icon="🧠", layout="centered")

# Initialize Groq Client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Define Feedback File
FEEDBACK_FILE = "chat_feedback.json"

# Load & Save Feedback Data
def load_feedback_data():
    try:
        with open(FEEDBACK_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"positive": 0, "negative": 0, "improvement_suggestions": []}

def save_feedback_data(data):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(data, file, indent=4)

feedback_data = load_feedback_data()

# Session State for Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Model Selection
models = {
    "gemma2-9b-it": "Gemma2-9b-it",
    "llama-3.3-70b-versatile": "LLaMA3.3-70b",
    "llama-3.1-8b-instant": "LLaMA3.1-8b",
    "llama3-70b-8192": "LLaMA3-70b",
    "llama3-8b-8192": "LLaMA3-8b",
}
st.sidebar.title("Settings")
model_option = st.sidebar.selectbox("Choose a model:", list(models.keys()), format_func=lambda x: models[x])

# Clear Chat Button
def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button("🗑 Clear Chat", on_click=clear_chat)

# Chat UI Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💬 MindEase Chatbot</h1>", unsafe_allow_html=True)

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar="🧑‍⚕️" if msg["role"] == "user" else "🤖"):
        st.markdown(f"**{'You' if msg['role'] == 'user' else 'Bot'}:** {msg['content']}")

# Generate Chat Response
def chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Handle User Input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(f"**You:** {prompt}")

    # Fixed bot name/identity response
    name_queries = [
        "what is your name", "who are you", "your name", "tell me your name",
        "who am I talking to", "identify yourself"
    ]
    if any(q in prompt.lower() for q in name_queries):
        bot_intro = "I'm a Language Model from NirveonX Health, here to support and talk to you. 😊"
        st.session_state.messages.append({"role": "assistant", "content": bot_intro})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**Bot:** {bot_intro}")
        st.stop()

    # Few-shot learning examples
    few_shot_examples = [
        {"role": "system", "content": "You are MindEase, a supportive and friendly mental health chatbot by NirveonX Health."},
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I'm a Language Model from NirveonX Health, here to support and talk to you. 😊"},
        {"role": "user", "content": "I feel sad and lonely."},
        {"role": "assistant", "content": "I'm really sorry you're feeling this way. You're not alone. I'm here for you. 💙"},
        {"role": "user", "content": "Can you help me feel better?"},
        {"role": "assistant", "content": "Absolutely. Let’s talk it through. Sometimes just expressing how you feel can help. 💬"},
    ]

    full_response = ""
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=few_shot_examples + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=512,
            stream=True
        )
        with st.chat_message("assistant", avatar="🤖"):
            response_gen = chat_responses(chat_completion)
            full_response = st.write_stream(response_gen)
    except:
        st.error("Something went wrong.", icon="🚨")

    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})

    # Emergency Detection
    emergency_keywords = [
        "suicide", "end my life", "kill myself", "no reason to live",
        "giving up", "can't go on", "hopeless", "worthless", "i want to die",
        "life is meaningless", "tired of everything", "i feel empty", "I want to hurt myself",
    ]

    if any(word in prompt.lower() for word in emergency_keywords):
        st.error("⚠️ It seems you're going through a really tough time.", icon="🚨")
        st.warning(
            "**You're not alone.** Please talk to someone you trust or reach out to a mental health professional.\n\n"
            "📞 **Indian National Mental Health Helpline (KIRAN)**: `1800-599-0019` (Toll-Free, 24x7)\n\n"
            "🏥 Visit your **nearest hospital** or mental health clinic if needed.\n\n"
            "Your well-being matters. We're here for you. 💙",
            icon="🧠"
        )

# Sidebar Feedback
st.sidebar.subheader("💬 Chat Feedback")

feedback = st.sidebar.radio(
    "Rate your experience:",
    ["😡 Very Bad", "😞 Bad", "😐 Neutral", "😊 Good", "😍 Excellent"],
    index=None
)

if feedback:
    score = {"😡 Very Bad": 1, "😞 Bad": 2, "😐 Neutral": 3, "😊 Good": 4, "😍 Excellent": 5}[feedback]
    feedback_data["positive" if score > 3 else "negative"] += 1

    if score <= 2:
        st.sidebar.warning("What can we improve?")
        tip = st.sidebar.text_input("Your suggestion:")
        if tip:
            feedback_data["improvement_suggestions"].append(tip)

    save_feedback_data(feedback_data)

# Feedback Summary
st.sidebar.subheader("📊 Feedback Summary")
st.sidebar.write(f"👍 Positive: {feedback_data['positive']} | 👎 Negative: {feedback_data['negative']}")
if feedback_data["improvement_suggestions"]:
    st.sidebar.subheader("🔍 Recent Suggestions")
    for s in feedback_data["improvement_suggestions"][-3:]:
        st.sidebar.write(f"- {s}")
