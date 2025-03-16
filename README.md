# NLP_Project_AI_Rewriting_Assistant

## Project Overview
Our project aims to build an AI-powered assistant that rewrites emails in different tones (formal, casual, persuasive, apologetic, assertive) while preserving the original meaning. We are leveraging transformer-based models (T5, GPT-4) to enable automatic tone adaptation, enhancing communication effectiveness.

## 1. Data Collection & Preprocessing
🔹 Datasets Chosen:

Enron Email Dataset (Business Emails - Formal, Casual, Persuasive)
Hillary Clinton Email Dataset (Diplomatic & Formal Emails)
Customer Service Email Dataset (Apologetic & Persuasive Emails)
🔹 Preprocessing Steps Completed:
✔ Removed email metadata (headers, timestamps, signatures)
✔ Normalized text (lowercasing, punctuation removal, contraction expansion)
✔ Tokenized the content for processing
✔ Merged all datasets into a unified preprocessed dataset

## 2. Model Development
🔹 Baseline Model:
✔ A Seq2Seq model with attention mechanism as a starting point

🔹 Transformer-Based Models Selected:
✔ T5 (Text-to-Text Transfer Transformer) Fine-Tuning 
✔ GPT-4 Fine-Tuning for Email Tone Adaptation

## 3. Model Training & Fine-Tuning
🔹 Training Process:
✔ Dataset split into training (70%), validation (20%), and testing (10%)
✔ Fine-tuning T5 on Hugging Face for different email tones
✔ Initial results show promising text coherence and tone adaptation

🔹 Reinforcement Learning Integration:
✔ Implemented feedback collection mechanism (thumbs-up/down on generated emails)
✔ Next step: Use feedback for model improvement

## 4. Evaluation Metrics Defined
✔ BLEU Score - Measures similarity to human-written emails
✔ ROUGE Score - Evaluates text overlap with reference emails
✔ Human Evaluation - Assesses fluency, coherence, and tone accuracy



