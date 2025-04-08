import streamlit as st
from datasets import load_dataset
from groq import Groq, APIError
import json
import datetime
import hashlib
import time
from typing import Dict, List, Tuple

# ========== CONFIGURATION ==========
MODEL_NAME = "llama3-70b-8192"  # Groq's current fastest high-quality model
TEMPERATURE = 0.7  # Balance between creativity and reliability
MAX_TOKENS = 350   # Longer responses when needed

# ========== INITIALIZATION ==========
@st.cache_resource
def init_groq_client():
    return Groq(api_key="")

client = init_groq_client()

# ========== CHATBOT CLASS ==========
class MindEaseChatbot:
    def __init__(self):
        self.chat_history: List[Tuple[str, str]] = []
        self.session_id = None
        self.crisis_mode = False
        self.last_sentiment = {"label": "NEUTRAL", "score": 0.5}
        self._load_resources()
    
    def _load_resources(self):
        """Load mental health resources and dataset"""
        try:
            self.dataset = load_dataset("Amod/mental_health_counseling_conversations")
            st.session_state.dataset_loaded = True
        except Exception as e:
            st.warning(f"Couldn't load full dataset: {str(e)}")
    
    def _generate_session_id(self, user_input: str) -> str:
        """Create unique session ID for tracking"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        input_hash = hashlib.md5(user_input.encode()).hexdigest()[:6]
        return f"ME_{timestamp}_{input_hash}"
    
    def analyze_sentiment(self, text: str) -> Dict[str, str | float]:
        """Advanced sentiment analysis with emotional nuance detection"""
        if not text.strip():
            return {"label": "NEUTRAL", "score": 0.5}
        
        prompt = """Analyze this message for a mental health context. Return JSON with:
        - 'sentiment' (positive/neutral/negative)
        - 'confidence' (0-1)
        - 'emotions' (list of detected emotions)
        - 'urgency' (1-5 scale)"""
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=150
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return {
                'label': analysis.get('sentiment', 'neutral').upper(),
                'score': float(analysis.get('confidence', 0.5)),
                'emotions': analysis.get('emotions', []),
                'urgency': int(analysis.get('urgency', 1))
            }
            
        except Exception as e:
            st.error(f"Sentiment analysis error: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5, "emotions": [], "urgency": 1}
    
    def detect_crisis(self, text: str) -> bool:
        """Multi-layered crisis detection system"""
        text_lower = text.lower()
        
        # 1. Keyword detection with weights
        crisis_phrases = {
            'suicide': 0.9, 'kill myself': 1.0, 'end it all': 0.95,
            'self-harm': 0.85, 'want to die': 0.9, 'no reason to live': 0.95,
            'can\'t go on': 0.8, 'suicidal': 0.95
        }
        
        # 2. Sentiment analysis boost
        sentiment = self.analyze_sentiment(text)
        sentiment_boost = 0.4 if sentiment['label'] == 'NEGATIVE' and sentiment['urgency'] >= 4 else 0
        
        # 3. Combined scoring
        crisis_score = sum(
            weight for phrase, weight in crisis_phrases.items() 
            if phrase in text_lower
        ) + sentiment_boost
        
        if crisis_score > 0.85:
            self.crisis_mode = True
            return True
        return False
    
    def generate_response(self, user_input: str) -> str:
        """Generate compassionate, context-aware response"""
        if not user_input.strip():
            return "I'm here to listen. Could you share how you're feeling?"
        
        # Initialize session if first message
        if not self.session_id:
            self.session_id = self._generate_session_id(user_input)
        
        # Crisis detection takes priority
        if self.detect_crisis(user_input):
            return self._get_crisis_response()
        
        # Enhanced sentiment analysis
        self.last_sentiment = self.analyze_sentiment(user_input)
        
        try:
            # Build conversation context
            messages = self._build_message_history(user_input)
            
            # Get AI response
            response = client.chat.completions.create(
                messages=messages,
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stop=["\n\n"]
            )
            
            bot_response = response.choices[0].message.content
            
            # Maintain conversation history
            self._update_chat_history(user_input, bot_response)
            
            return bot_response
            
        except APIError as e:
            st.error(f"API Error: {str(e)}")
            return "I'm having technical difficulties. Please try again shortly."
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return "I'm having trouble understanding. Could you rephrase that?"
    
    def _build_message_history(self, user_input: str) -> List[Dict]:
        """Construct the conversation context"""
        messages = [{
            "role": "system",
            "content": """You are MindEase, an advanced mental health assistant. Provide:
            - Empathetic, clinically-informed responses
            - Evidence-based psychological support
            - Crisis intervention when needed
            - Responses limited to 2-3 sentences
            - Always validate feelings first"""
        }]
        
        # Add last 3 exchanges for context
        for user_msg, bot_msg in self.chat_history[-3:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        
        # Add current message with emotional context
        if self.last_sentiment['label'] == 'NEGATIVE':
            emotional_context = {
                "role": "user",
                "content": f"[User appears distressed - urgency {self.last_sentiment['urgency']}/5] {user_input}"
            }
        else:
            emotional_context = {"role": "user", "content": user_input}
            
        messages.append(emotional_context)
        return messages
    
    def _update_chat_history(self, user_input: str, bot_response: str):
        """Maintain conversation history with size limit"""
        self.chat_history.append((user_input, bot_response))
        if len(self.chat_history) > 12:  # Keep last 12 exchanges
            self.chat_history = self.chat_history[-12:]
    
    def _get_crisis_response(self) -> str:
        """Immediate crisis intervention protocol"""
        return f"""🚨 **MindEase Crisis Alert** (Session: {self.session_id})

**Immediate Support Available:**
1. 🌎 Global Hotline: [Befrienders Worldwide](https://www.befrienders.org/)
2. 🇺🇸 US: 988 Suicide & Crisis Lifeline
3. 🇬🇧 UK: 116 123 (Samaritans)
4. 💬 Crisis Text Line: Text HOME to 741741

**You're Not Alone:**
- Your feelings are valid and important
- This pain can feel overwhelming, but help is available
- Would you like me to connect you with a trained counselor?"""

# ========== UI COMPONENTS ==========
def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="MindEase 2.0 | AI Mental Health Support",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS with calming color scheme
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
    }
    .stChatInput input {
        border: 1px solid #e2e8f0 !important;
    }
    .user-message {
        background: #ffffff;
        border-left: 4px solid #4f46e5;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .bot-message {
        background: #f0f9ff;
        border-left: 4px solid #0891b2;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .crisis-alert {
        background: #fff1f2;
        border-left: 4px solid #e11d48;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.85; }
        100% { opacity: 1; }
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def display_resources():
    """Mental health resources sidebar"""
    with st.sidebar:
        st.title("🧠 MindEase Resources")
        st.markdown("""
        **24/7 Crisis Support:**
        - 🌍 [Befrienders Worldwide](https://www.befrienders.org/)
        - 🇺🇸 988 Suicide & Crisis Lifeline
        - 🇬🇧 116 123 (Samaritans)
        
        **Professional Help:**
        - [OpenCounseling](https://www.opencounseling.com/)
        - [Psychology Today Therapists](https://www.psychologytoday.com/)
        """)
        
        if st.session_state.get('chatbot'):
            with st.expander("Session Details"):
                st.caption(f"Session ID: {st.session_state.chatbot.session_id or 'Not started'}")
                if st.session_state.chatbot.crisis_mode:
                    st.error("Crisis mode activated")
                st.metric("Conversation Length", f"{len(st.session_state.chatbot.chat_history)} messages")

def display_conversation(chatbot):
    """Render the chat interface"""
    for i, (user_msg, bot_msg) in enumerate(chatbot.chat_history):
        # User message
        with st.chat_message("user"):
            st.markdown(f"<div class='user-message'>{user_msg}</div>", unsafe_allow_html=True)
        
        # Bot message with crisis styling if needed
        with st.chat_message("assistant"):
            if chatbot.crisis_mode and i == len(chatbot.chat_history)-1:
                st.markdown(f"<div class='bot-message crisis-alert'>{bot_msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-message'>{bot_msg}</div>", unsafe_allow_html=True)
    
    # Show sentiment analysis if available
    if hasattr(chatbot, 'last_sentiment') and chatbot.chat_history:
        with st.expander("Emotional Analysis"):
            sent = chatbot.last_sentiment
            cols = st.columns(3)
            cols[0].metric("Sentiment", sent['label'])
            cols[1].metric("Confidence", f"{sent['score']*100:.1f}%")
            cols[2].metric("Urgency Level", sent.get('urgency', 1), delta="1-5 scale")
            
            if sent.get('emotions'):
                st.caption(f"Detected emotions: {', '.join(sent['emotions'])}")

# ========== MAIN APP ==========
def main():
    setup_page()
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = MindEaseChatbot()
    
    # Page header
    st.title("MindEase 2.0")
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
    <h3 style='color: #4f46e5;'>Your compassionate AI mental health companion</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display resources
    display_resources()
    
    # Chat interface
    user_input = st.chat_input("How are you feeling today?")
    
    if user_input:
        with st.spinner("MindEase is thinking..."):
            start_time = time.time()
            bot_response = st.session_state.chatbot.generate_response(user_input)
            response_time = time.time() - start_time
            
            # Log performance (can be removed in production)
            st.session_state.last_response_time = response_time
    
        # Display conversation
        display_conversation(st.session_state.chatbot)
        
        # Show performance metrics (debug only)
        if st.session_state.get('last_response_time'):
            st.caption(f"Response generated in {st.session_state.last_response_time:.2f}s")

if __name__ == "__main__":
    main()
