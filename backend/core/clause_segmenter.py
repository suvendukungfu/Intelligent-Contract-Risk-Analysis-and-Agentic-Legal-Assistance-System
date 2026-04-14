"""
Clause segmenter for dividing contract text into individual clauses.
Implements Requirements 2.1, 2.2, 2.3, 2.4, 2.5.
"""

import re
import logging
from typing import List
from uuid import uuid4

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Clause segmentation will use regex patterns only.")

from api.models import ParsedDocument, Clause

logger = logging.getLogger(__name__)


class ClauseSegmenter:
    """
    Segmenter for dividing contract documents into individual clauses.
    
    Uses multiple strategies:
    1. Numbered/lettered clause patterns (e.g., "1.", "a)", "Article 1")
    2. Section headers and structural markers
    3. spaCy sentence tokenization as fallback
    
    Features:
    - Preserves original clause text
    - Assigns unique UUIDs to each clause
    - Handles edge cases (documents with <2 clauses)
    - Tracks character positions
    """
    
    def __init__(self):
        """Initialize the clause segmenter."""
        self.min_clause_length = 10
        self.nlp = None
        
        # Try to load spaCy model if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                self.nlp = None
        
        # Compile regex patterns for clause boundaries
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for identifying clause boundaries."""
        # Pattern 1: Numbered clauses (1., 2., 3., etc.)
        self.numbered_pattern = re.compile(
            r'^\s*(\d+)\.\s+',
            re.MULTILINE
        )
        
        # Pattern 2: Lettered clauses (a., b., c., or (a), (b), (c))
        self.lettered_pattern = re.compile(
            r'^\s*(\(?[a-z]\)?)[\.\)]\s+',
            re.MULTILINE | re.IGNORECASE
        )
        
        # Pattern 3: Section/Article headers
        self.section_pattern = re.compile(
            r'^\s*(Section|Article|Clause|Paragraph|Part)\s+(\d+|[IVX]+)',
            re.MULTILINE | re.IGNORECASE
        )
        
        # Pattern 4: Roman numerals
        self.roman_pattern = re.compile(
            r'^\s*([IVX]+)\.\s+',
            re.MULTILINE
        )
        
        # Pattern 5: Double newline (paragraph breaks)
        self.paragraph_pattern = re.compile(r'\n\n+')
    
    def segment(self, document: ParsedDocument) -> List[Clause]:
        """
        Segment document into clauses.
        
        Args:
            document: ParsedDocument with text content
            
        Returns:
            List of Clause objects with unique IDs
        """
        text = document.text
        
        if not text or len(text.strip()) < self.min_clause_length:
            logger.warning(
                f"Document {document.id} has insufficient text for segmentation"
            )
            # Return entire document as single clause
            return self._create_single_clause(document)
        
        # Try different segmentation strategies in order of preference
        clauses = None
        
        # Strategy 1: Numbered clauses
        clauses = self._segment_by_numbered_pattern(text, document.id)
        if clauses and len(clauses) >= 2:
            logger.info(
                f"Segmented document {document.id} into {len(clauses)} clauses "
                "using numbered pattern"
            )
            return clauses
        
        # Strategy 2: Section/Article headers
        clauses = self._segment_by_section_pattern(text, document.id)
        if clauses and len(clauses) >= 2:
            logger.info(
                f"Segmented document {document.id} into {len(clauses)} clauses "
                "using section pattern"
            )
            return clauses
        
        # Strategy 3: Paragraph breaks
        clauses = self._segment_by_paragraphs(text, document.id)
        if clauses and len(clauses) >= 2:
            logger.info(
                f"Segmented document {document.id} into {len(clauses)} clauses "
                "using paragraph breaks"
            )
            return clauses
        
        # Strategy 4: spaCy sentence tokenization (fallback)
        if self.nlp:
            clauses = self._segment_by_sentences(text, document.id)
            if clauses and len(clauses) >= 2:
                logger.info(
                    f"Segmented document {document.id} into {len(clauses)} clauses "
                    "using spaCy sentences"
                )
                return clauses
        
        # Edge case: No clear boundaries found, treat as single clause
        logger.info(
            f"Document {document.id} has no clear clause boundaries. "
            "Treating as single clause."
        )
        return self._create_single_clause(document)
    
    def _segment_by_numbered_pattern(
        self, 
        text: str, 
        document_id: str
    ) -> List[Clause]:
        """Segment text using numbered clause patterns (1., 2., 3., etc.)."""
        matches = list(self.numbered_pattern.finditer(text))
        
        if len(matches) < 2:
            return []
        
        clauses = []
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            # Determine end position
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            clause_text = text[start_pos:end_pos].strip()
            
            # Skip if too short
            if len(clause_text) < self.min_clause_length:
                continue
            
            clause = Clause(
                id=str(uuid4()),
                document_id=document_id,
                text=clause_text,
                position=i,
                start_char=start_pos,
                end_char=end_pos
            )
            clauses.append(clause)
        
        return clauses
    
    def _segment_by_section_pattern(
        self, 
        text: str, 
        document_id: str
    ) -> List[Clause]:
        """Segment text using section/article headers."""
        matches = list(self.section_pattern.finditer(text))
        
        if len(matches) < 2:
            return []
        
        clauses = []
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            # Determine end position
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            clause_text = text[start_pos:end_pos].strip()
            
            # Skip if too short
            if len(clause_text) < self.min_clause_length:
                continue
            
            clause = Clause(
                id=str(uuid4()),
                document_id=document_id,
                text=clause_text,
                position=i,
                start_char=start_pos,
                end_char=end_pos
            )
            clauses.append(clause)
        
        return clauses
    
    def _segment_by_paragraphs(
        self, 
        text: str, 
        document_id: str
    ) -> List[Clause]:
        """Segment text using paragraph breaks (double newlines)."""
        # Split by double newlines
        paragraphs = self.paragraph_pattern.split(text)
        
        if len(paragraphs) < 2:
            return []
        
        clauses = []
        current_pos = 0
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            
            # Skip empty or too-short paragraphs
            if not para or len(para) < self.min_clause_length:
                current_pos += len(para) + 2  # +2 for newlines
                continue
            
            # Find actual position in original text
            start_pos = text.find(para, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(para)
            
            clause = Clause(
                id=str(uuid4()),
                document_id=document_id,
                text=para,
                position=len(clauses),
                start_char=start_pos,
                end_char=end_pos
            )
            clauses.append(clause)
            
            current_pos = end_pos + 2  # +2 for newlines
        
        return clauses
    
    def _segment_by_sentences(
        self, 
        text: str, 
        document_id: str
    ) -> List[Clause]:
        """Segment text using spaCy sentence tokenization."""
        if not self.nlp:
            return []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        clauses = []
        
        for i, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            
            # Skip if too short
            if len(sent_text) < self.min_clause_length:
                continue
            
            clause = Clause(
                id=str(uuid4()),
                document_id=document_id,
                text=sent_text,
                position=i,
                start_char=sent.start_char,
                end_char=sent.end_char
            )
            clauses.append(clause)
        
        return clauses
    
    def _create_single_clause(self, document: ParsedDocument) -> List[Clause]:
        """
        Create a single clause from entire document.
        
        Used when:
        - Document has <2 clauses
        - No clear clause boundaries found
        """
        clause = Clause(
            id=str(uuid4()),
            document_id=document.id,
            text=document.text.strip(),
            position=0,
            start_char=0,
            end_char=len(document.text)
        )
        
        return [clause]
    
    def validate_clauses(self, clauses: List[Clause]) -> bool:
        """
        Validate that clauses meet requirements.
        
        Checks:
        - All clause IDs are unique
        - All clauses have non-empty text
        - Positions are sequential
        
        Args:
            clauses: List of Clause objects
            
        Returns:
            True if valid, False otherwise
        """
        if not clauses:
            return False
        
        # Check unique IDs
        clause_ids = [c.id for c in clauses]
        if len(clause_ids) != len(set(clause_ids)):
            logger.error("Duplicate clause IDs found")
            return False
        
        # Check non-empty text
        for clause in clauses:
            if not clause.text or len(clause.text.strip()) < self.min_clause_length:
                logger.error(f"Clause {clause.id} has insufficient text")
                return False
        
        # Check sequential positions
        positions = [c.position for c in clauses]
        expected_positions = list(range(len(clauses)))
        if positions != expected_positions:
            logger.warning("Clause positions are not sequential")
        
        return True
