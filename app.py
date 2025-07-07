import streamlit as st
from model.summarizer import generate_summary

st.set_page_config(page_title="NLP Summarizer", layout="centered")
st.title("🧠 Text Summarizer")
st.markdown("Paste a long article below, and get a concise summary using a transformer model.")

article_text = st.text_area("📝 Paste your article here", height=300, placeholder="Paste or type your article...")

if st.button("📄 Generate Summary"):
    if not article_text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Summarizing..."):
            summary = generate_summary(article_text)
        st.success("✅ Summary:")
        st.write(summary)
#clean_env\Scripts\activate for terminal