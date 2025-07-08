import streamlit as st
import os
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt_tab')
nltk.download('punkt')

# ✅ Access secure token
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    use_auth_token=os.environ["HF_TOKEN"]
)

corrector = pipeline(
    "text2text-generation",
    model="prithivida/grammar_error_correcter_v1",
    use_auth_token=os.environ["HF_TOKEN"]
)


def chunk_text(text: str, max_words: int = 350) -> list:
    """
    Splits text into chunks with approximately max_words words per chunk.
    Returns a list of text chunks.
    """
    sentences = sent_tokenize(text)  # Split the article into sentences
    chunks = []                      # Stores final text chunks
    current_chunk = ""              # Temp string to hold one chunk
    word_count = 0                  # Count words in current chunk

    for sentence in sentences:
        words_in_sentence = len(word_tokenize(sentence))

        # If adding this sentence keeps chunk under max_words → add it
        if word_count + words_in_sentence <= max_words:
            current_chunk += " " + sentence
            word_count += words_in_sentence
        else:
            # Chunk full → push to list and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            word_count = words_in_sentence

    # Add the final chunk (if any)
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks  # Return list of chunked text strings

def generate_summary(article_text: str, max_length: int = 256) -> str:
    """
    Main function: generates a high-quality summary from a long article.
    Steps:
    1. Split text into manageable chunks.
    2. Summarize each chunk individually.
    3. Combine those partial summaries.
    4. Optionally re-summarize combined summary for coherence.
    5. Apply grammar correction (optional).
    """

    # Step 1: Split article into chunks
    chunks = chunk_text(article_text)

    # Step 2: Summarize each chunk individually
    partial_summaries = []
    for chunk in chunks:
        try:
            input_words = len(chunk.split())
            
            # Skip chunks that are too small
            if input_words < 30:
                print(f"⚠️ Skipping short chunk with {input_words} words")
                continue

            # Calculate safe max_length
            dynamic_max_len = min(max_length, max(50, input_words - 10))

            summary = summarizer(
                chunk,
                max_length=dynamic_max_len,
                min_length=30,
                do_sample=False,
                early_stopping=True,
                return_full_text=True
            )

            partial_summaries.append(summary[0]['summary_text'])

        except Exception as e:
            print(f"❌ Chunk error: {e}")
            partial_summaries.append("[Error summarizing chunk]")

    # Step 3 (Optional): Join and return combined summary
    final_summary = " ".join(partial_summaries)
    return final_summary


# OPTIONAL test block
if __name__ == "__main__":      
    sample_text = """Paste or load a sample article here..."""
    print("Generated Summary:\n", generate_summary(sample_text))


