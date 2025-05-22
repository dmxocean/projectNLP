from langdetect import detect, LangDetectException

def detect_sentence_language(text: str, default: str = "es") -> str:
    """
    Detect language for a single sentence using langdetect

    Args:
        text (str): Sentence text
        default (str): Default language to return if detection fails

    Returns:
        str: Detected language code ('es' or 'ca')
    """
    lang = default
    if text and isinstance(text, str) and len(text) > 10:  # Check for minimal length
        try:
            detected_code = detect(text)
            if detected_code == "es":
                lang = "es"
            elif detected_code == "ca":
                lang = "ca"
        except LangDetectException:
            pass  # Keep default language
    return lang