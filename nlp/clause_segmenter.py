import re

def segment_clauses(text):
    """
    Contracts are long! This function splits a long document into smaller 'clauses'.
    Think of it like cutting a long sandwich into bite-sized pieces.
    """
    if not text:
        return []

    # Legal contracts usually use specific symbols to start new sections:
    # 1. Numbers followed by a period (1. , 2. )
    # 2. Words like "Article" or "Section"
    # 3. Double newlines (paragraphs)
    
    # We use 'Regex' (Regular Expressions) which is like a super-powered CTRL+F search.
    # This pattern looks for common legal markers.
    # Updated to handle mid-line Article/Section more robustly and catch them at line start.
    # Added escape for period check to ensure split before numbers.
    pattern = r'\n\s*\n|(?<=\n|\.)\s*(?=\d+\.\s|Article\s+[IVXLCDM\d]+|Section\s+\d+|[A-Z]\.\s)|^Article\s+[IVXLCDM\d]+|^Section\s+\d+|^[A-Z]\.\s'
    
    # Split the text based on where those markers appear.
    segments = re.split(pattern, text, flags=re.IGNORECASE)
    
    # Final cleanup: Remove whitespace and ignore tiny segments.
    clauses = []
    for seg in segments:
        clean_seg = seg.strip()
        if clean_seg and len(clean_seg) > 10: # Only keep meaningful text
            clauses.append(clean_seg)
            
    return clauses

if __name__ == "__main__":
    test_text = "1. First Clause. \n\n 2. Second Clause. \n Article III: Third Clause."
    print(f"Found {len(segment_clauses(test_text))} clauses.")
