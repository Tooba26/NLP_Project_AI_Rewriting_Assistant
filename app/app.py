import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIRS = {
    "Informal to Formal (Default Model)": "ToobiBabe/email_tone_rewrite",
    "Formal to Informal": "ToobiBabe/formal_informal_rewriter",
    "Informal to Apologetic": "ToobiBabe/apologetic_email_rewriter",
    "Formal to Apologetic": "ToobiBabe/formal_apologetic_rewriter",
    "Apologetic to Formal": "ToobiBabe/apologetic_formal_rewriter",
    "Apologetic to Informal": "ToobiBabe/apologetic_informal_rewrite"
}



# Cache models/tokenizers
@st.cache_resource
def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

# App UI
st.title("Email Tone Rewriter")
st.write("Transform your email's tone using AI-powered rewriting.")

# Dropdown to pick tone transformation
tone_option = st.selectbox("Select Rewriting Style", list(MODEL_DIRS.keys()))

# User input
recipient_name = st.text_input("Recipient Name (e.g., Nomi):")
sender_name = st.text_input("Your Name (e.g., Alex):")
email_body = st.text_area("Enter Email Body:", height=200)

# Button to rewrite
if st.button("Rewrite Email"):
    if not email_body.strip():
        st.warning("Please enter the email body.")
    elif not recipient_name.strip() or not sender_name.strip():
        st.warning("Please enter both recipient and sender names.")
    else:
        model_path = MODEL_DIRS[tone_option]
        model, tokenizer = load_model(model_path)

        # Input prefix from tone label
        input_prefix = tone_option.split(" to ")[0].lower() + ": "
        input_text = input_prefix + email_body.strip()

        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        rewritten_body = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Format email
        header = f"Dear {recipient_name},\n\n"
        footer = f"\n\nBest regards,\n{sender_name}"
        final_email = header + rewritten_body + footer

        st.subheader("Rewritten Email:")
        st.text_area("Output", final_email, height=300)

