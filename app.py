"""
Root-level entry point for Hugging Face Spaces deployment.
Hugging Face expects an `app.py` in the root folder.
"""
import sys
import os

# Ensure the root directory is in the PYTHONPATH
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import and run the main Streamlit application
from app.streamlit_app import main

if __name__ == "__main__":
    main()
