Empathetic Transformer Chatbot  
Trained From Scratch on the Empathetic Dialogues Dataset  

Deployed App:👉 [Live Demo on Streamlit](https://emphatheticchatbot-jexmjwrijyppjlksdrkphf.streamlit.app/)  
Model Weights: Hosted on [Hugging Face Hub](https://huggingface.co/asherrer/empathetic-transformer/tree/main)  

---

 Overview
This project implements a **Transformer Encoder–Decoder** chatbot trained **from scratch** (no pretrained weights) to generate empathetic responses based on emotion and situation.  
It follows the **Empathetic Dialogues** dataset format from Facebook AI.

---

Architecture
- Embedding dimension: 256  
- Heads: 4  
- Encoder/Decoder layers: 2 each  
- Feed-forward dim: 512  
- Dropout: 0.1  
- Training: 2000 samples × 20 epochs (Adam, LR 3e-4)  
- Metrics: BLEU, ROUGE-L, chrF, Perplexity  

---

Usage

Run Locally
```bash
git clone https://github.com/ayesha-Saeed-0/Emphathetic_ChatBot.git
pip install -r requirements.txt
streamlit run app.py
