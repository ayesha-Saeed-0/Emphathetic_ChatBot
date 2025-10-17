# app.py
import streamlit as st
import torch
from model_utils import load_model, encode_text, decode_ids, greedy_decode

st.set_page_config(page_title="Empathetic Chatbot", layout="centered")

st.title("💬 Empathetic Transformer Chatbot")
st.markdown("Trained from scratch on the Empathetic Dialogues dataset.")

# --- Paths (update these) ---
MODEL_PATH = "transformer_best.pt"
VOCAB_PATH = "vocab.json"

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("Loading model... please wait ⏳")
model, stoi, itos = load_model(MODEL_PATH, VOCAB_PATH, device)
st.success("Model loaded successfully!")

# --- Chat state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Input boxes ---
emotion = st.text_input("Emotion (optional)", value="sentimental")
situation = st.text_area("Situation", value="I remember going to the fireworks with my best friend.")
user_input = st.text_area("Your message", value="I miss those days.")

if st.button("Get Empathetic Reply"):
    input_text = f"Emotion: {emotion} | Situation: {situation} | Customer: {user_input} Agent:"
    input_ids = torch.tensor([encode_text(input_text.lower(), stoi)], dtype=torch.long).to(device)
    output_ids = greedy_decode(model, input_ids, stoi, itos, device)
    response = decode_ids(output_ids[0].cpu().numpy(), itos)

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Agent", response))

# --- Display chat ---
for role, text in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {text}")
    else:
        st.markdown(f"**🤖 {role}:** {text}")
