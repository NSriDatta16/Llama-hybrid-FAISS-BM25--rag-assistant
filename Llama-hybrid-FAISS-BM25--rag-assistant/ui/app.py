# ui/app.py
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8008")

st.set_page_config(page_title="SmartGPT", page_icon="🧠", layout="wide")
st.title("🧠 SmartGPT")

with st.sidebar:
    st.markdown("### Settings")
    provider = st.selectbox("Provider", ["Groq", "OpenAI"], index=0)
    alpha = st.slider("Hybrid α (dense weight)", 0.0, 1.0, 1.0, 0.05)
    top_k = st.slider("Top-K", 1, 10, 5, 1)

    groq_model = st.selectbox(
    "Groq model",
    [
        "llama-3.3-70b-versatile",
        "llama-3.3-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    index=0
    )

    st.caption(f"Backend: [{API_BASE}]({API_BASE})")

# chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# render history
for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

q = st.chat_input("Ask anything about the knowledge base…")
if q:
    st.session_state.chat.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        ph = st.empty()
        ph.markdown("Thinking…")

        # choose backend
        endpoint = "/ask_groq" if provider == "Groq" else "/ask"

        payload = {
            "query": q,
            "top_k": int(top_k),
            "alpha": float(alpha),
        }
        if provider == "Groq":
            payload["model"] = groq_model  # backend will use this

        try:
            r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()

            answer = (data.get("answer") or "(no answer)").strip()

            # No citations rendered
            md = answer

            ph.markdown(md)
            st.session_state.chat.append(("assistant", md))

        except Exception as e:
            ph.markdown(f"❌ **Error:** {e}")
