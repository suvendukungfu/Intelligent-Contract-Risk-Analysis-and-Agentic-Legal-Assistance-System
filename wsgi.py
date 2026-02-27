import os
import sys

# Path Alignment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.index import app

if __name__ == "__main__":
    app.run()
