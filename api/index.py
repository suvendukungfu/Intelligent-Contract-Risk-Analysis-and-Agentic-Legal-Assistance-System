from flask import Flask, request, jsonify, render_template
import os
import sys

# Path Alignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pdfplumber
    import pandas as pd
    from nlp.clause_segmenter import segment_clauses
    from models.inference import risk_engine
except ImportError as e:
    print(f"Error importing ML modules: {e}")

app = Flask(__name__, template_folder='../templates', static_folder='../static')

def get_text_from_file(file, filename):
    if filename.lower().endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
         return file.read().decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        raw_text = get_text_from_file(file, file.filename)
        clauses = segment_clauses(raw_text)
        
        results = []
        for idx, c in enumerate(clauses):
            label, conf, reasons = risk_engine.analyze_clause(c)
            
            # Scikit-learn int32/float64 are not JSON serializable by default
            try:
                conf = float(conf)
            except:
                pass
                
            results.append({
                "id": idx,
                "Clause": c,
                "Risk Level": label,
                "Confidence": conf,
                "Keywords": reasons
            })
            
        return jsonify({'results': results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
