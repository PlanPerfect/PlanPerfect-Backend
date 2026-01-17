import re

def SanitizeTextForPDF(text):
    """
    Sanitize text to remove characters that cause rendering issues in PDFs.
    Replaces problematic Unicode characters with ASCII-safe equivalents.
    """
    if not isinstance(text, str):
        return str(text)

    replacements = {
        '■': '-',    # Black square
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove remaining non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
