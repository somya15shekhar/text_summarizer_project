#You now have a clean, reusable summarizer module that Works for any article string Summarizes and 
# optionally corrects grammar Can be called from anywhere: notebook, Streamlit, et


from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')


# Transformers pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")

def chunk_text(text: str, max_words: int =600) -> list:
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


def generate_summary(article_text: str) -> str:

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
            # Summarize this chunk with max/min length settings
            summary = summarizer(
                chunk,
                max_length = 250,
                min_length = 60,
                do_sample =False)
            
            partial_summaries.append(summary[0]['summary_text'])

        except Exception as e:
            partial_summaries.append("[Error summarizing chunk]")
            print(f"chunk error: {e}")

    # Step 3: Join all partial summaries together
    combined_summary = " ".join(partial_summaries)


#OPTIONAL:
if __name__ == "__main__":      # Test block	Only for testing outside Streamlit
    sample_text = """Paste or load a sample article here..."""
    print("Generated Summary:\n", generate_summary(sample_text))

