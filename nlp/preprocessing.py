import string
import re

# We use spaCy because it's like a 'linguistic expert' for Python.
# It understands that "running" and "ran" are the same word ("run").
# We disable the parser and NER components since we only need lemmatization —
# this makes processing ~5x faster.
try:
    import spacy
    try:
        # 'sm' stands for 'small' - it's fast and enough for our needs.
        # disable=['parser','ner'] skips expensive components we don't need.
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        nlp = None
except (ImportError, Exception):
    nlp = None


def _clean_raw(text: str) -> str:
    """Steps 1-3: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Remove all non-ASCII characters (including emojis) and non-printable chars
    # We do a rigorous filtering to remove NUL bytes and other non-printables
    text = "".join(char for char in text if 32 <= ord(char) <= 126)
    # Collapse any resulting weird whitespace or hidden chars
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    """
    Cleans a single text string so our AI can understand it more easily.
    For large corpora prefer batch_preprocess_texts() which is much faster.
    """
    if not isinstance(text, str):
        return ""

    text = _clean_raw(text)

    # Lemmatization (using spaCy):
    # This turns words like "agreements" into "agreement".
    if nlp:
        try:
            doc = nlp(text)
            # We also ignore 'stopwords' (common words like 'the', 'is' that add no value).
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
            return " ".join(tokens)
        except Exception:
            pass

    # Fallback: If spaCy isn't working, we do a simple split.
    tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens)


def batch_preprocess_texts(texts, batch_size=512):
    """
    Efficiently preprocess a list/Series of texts using spaCy's nlp.pipe(),
    which is significantly faster than calling preprocess_text() row-by-row.
    Falls back to the single-text path if spaCy is unavailable.
    """
    cleaned = [_clean_raw(t) if isinstance(t, str) else "" for t in texts]

    if nlp:
        results = []
        for doc in nlp.pipe(cleaned, batch_size=batch_size):
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
            results.append(" ".join(tokens))
        return results

    # Fallback without spaCy
    return [" ".join(t for t in text.split() if len(t) > 2) for text in cleaned]

if __name__ == "__main__":
    # Test our 'cleaning machine'
    sample = "The parties are currently agreeing to the terms!"
    print(f"Before: {sample}")
    print(f"After : {preprocess_text(sample)}")
