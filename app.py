import streamlit as st
from model.summarizer import generate_summary
import pandas as pd

st.set_page_config(page_title="NLP Summarizer", layout="centered")
st.title("🧠 Text Summarizer")
st.markdown("Paste a long article below or upload a CSV file to get summaries using a transformer model.")

# Manual input
article_text = st.text_area("📝 Paste your article here", height=300, placeholder="Paste or type your article...")

if st.button("📄 Generate Summary"):
    if not article_text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Summarizing..."):
            summary = generate_summary(article_text)
        st.success("✅ Summary:")
        st.write(summary)

# File upload
uploaded_file = st.file_uploader("📂 Upload CSV file (CNN/DailyMail format)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Preview of uploaded data:")
    st.dataframe(df.head())

    column_to_summarize = st.selectbox("🧠 Select the column containing articles:", df.columns)
    num_rows = st.slider("🔢 Number of articles to summarize", 1, len(df), 10)
    df = df.head(num_rows)

    if st.button("📝 Generate Summaries for Selected Rows"):
        st.warning("⏳ This may take a few minutes...")
        summaries = []
        for i, text in enumerate(df[column_to_summarize]):
            st.write(f"📍 Processing article {i+1} of {len(df)}...")
            summary = generate_summary(str(text))
            summaries.append(summary)

        df['summary'] = summaries
        st.success("✅ Summarization completed!")
        st.dataframe(df[['summary']].head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV with Summaries", data=csv, file_name="summarized_articles.csv", mime='text/csv')


#clean_env\Scripts\activate for terminal