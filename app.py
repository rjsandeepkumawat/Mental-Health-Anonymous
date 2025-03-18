import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

# Load chatbot model and tokenizer (using DialoGPT to avoid sentencepiece issues)
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

chatbot_model, chatbot_tokenizer = load_model()

# Load additional pipelines (removing gated model AIMH/mental-roberta-large)
@st.cache_resource
def load_pipelines():
    text_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    asr_whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large")
    return text_classifier, asr_whisper

text_classifier, asr_whisper = load_pipelines()

# Load conversation patterns from JSON
@st.cache_resource
def load_conversation_patterns():
    with open('sample.json', 'r') as f:
        sample_data = json.load(f)
    with open('samples.json', 'r') as f:
        samples_data = json.load(f)
    return sample_data, samples_data

sample_data, samples_data = load_conversation_patterns()

# Generate response

def check_trigger_response(user_input):
    user_input_lower = user_input.lower()
    for convo in sample_data["conversations"]:
        for keyword in convo["trigger_keywords"]:
            if keyword in user_input_lower:
                return convo["response"], convo["follow_up_questions"], convo.get("treatment", "")
    return None, None, None


def generate_response(user_input, chat_history_ids, temperature, max_new_tokens, top_p):
    matched_response, follow_ups, treatment = check_trigger_response(user_input)
    if matched_response:
        response = matched_response
        if follow_ups:
            response += "\n\n" + "\n".join(follow_ups)
        if treatment:
            response += f"\n\nSuggested treatment: {treatment}"
        return response, chat_history_ids

    new_user_input_ids = chatbot_tokenizer.encode(user_input + chatbot_tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    max_length = bot_input_ids.shape[-1] + max_new_tokens

    chat_history_ids = chatbot_model.generate(
        bot_input_ids,
        max_length=max_length,
        pad_token_id=chatbot_tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2
    )
    response = chatbot_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    if not response.strip():
        response = "Sorry, I couldn't process that. Could you rephrase?"
    return response.strip(), chat_history_ids

# Sidebar settings
st.sidebar.title("🧠 MindEase Chatbot")
st.sidebar.markdown("Your Safe Space to Talk & Heal. 💙")
st.sidebar.subheader("Models and Parameters")
model_choice = st.sidebar.selectbox("Choose a model", ["Model1"])
temperature = st.sidebar.slider("Temperature", 0.01, 1.00, 0.7)
max_new_tokens = st.sidebar.slider("Max New Tokens", 50, 500, 150)
top_p = st.sidebar.slider("Top-p (Nucleus Sampling)", 0.1, 1.0, 0.9)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = None
if "chat_log" not in st.session_state:
    st.session_state["chat_log"] = []

# Chat UI
st.title("🧠 MindEase Chatbot")
st.write("How may I assist you today?")
user_input = st.text_input("Type a message...", key="input")

if user_input:
    with st.spinner("Thinking..."):
        chatbot_response, st.session_state["chat_history"] = generate_response(user_input, st.session_state["chat_history"], temperature, max_new_tokens, top_p)
        st.session_state["chat_log"].append(("You", user_input))
        st.session_state["chat_log"].append(("Bot", chatbot_response))

# Display chat log in bubble format
for sender, message in st.session_state["chat_log"]:
    if sender == "You":
        st.markdown(f"""
            <div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin-bottom:5px; text-align:right; color:black;'>
                <strong>{sender}:</strong> {message}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color:#F1F0F0; padding:10px; border-radius:10px; margin-bottom:5px; text-align:left; color:red;'>
                <strong>{sender}:</strong> {message}
            </div>""", unsafe_allow_html=True)

# Auto scroll
st.markdown("""
<script>
    var inputBox = window.parent.document.querySelector('input[data-testid=\"stTextInput\"]');
    inputBox.focus();
    window.scrollTo(0, document.body.scrollHeight);
</script>
""", unsafe_allow_html=True)

# Clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state["chat_history"] = None
    st.session_state["chat_log"] = []
    st.experimental_rerun()
