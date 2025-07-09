from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')

# Load summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text: str, max_words: int = 350) -> list:
    """
    Splits text into chunks with approximately max_words words per chunk,
    ensuring chunks end at sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_len = 0

    for sentence in sentences:
        sentence_len = len(word_tokenize(sentence))
        if current_len + sentence_len <= max_words:
            current_chunk += " " + sentence
            current_len += sentence_len
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_len = sentence_len

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def generate_summary(article_text: str) -> str:
    """
    Summarizes a long article text by:
    1. Splitting into chunks
    2. Summarizing each chunk
    3. Combining all summaries
    """
    chunks = chunk_text(article_text)
    partial_summaries = []

    for chunk in chunks:
        try:
            if len(chunk.split()) < 30:
                continue  # skip too-short chunks

            summary = summarizer(chunk)[0]['summary_text']
            partial_summaries.append(summary)
        except Exception as e:
            print(f"Chunk error: {e}")
            partial_summaries.append("[Error summarizing chunk]")

    return " ".join(partial_summaries)

# For CLI testing
if __name__ == "__main__":
    sample_text = """Paste your test article here."""
    print(generate_summary(sample_text))
