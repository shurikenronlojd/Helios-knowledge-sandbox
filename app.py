from flask import Flask, render_template_string, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load the AI model
print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def get_feedback(similarity_score):
    """Generate helpful feedback based on similarity score"""
    if similarity_score > 0.85:
        return {
            "status": "too_similar",
            "message": "⚠️ This summary is too close to the original source.",
            "advice": "Try to capture the core idea without using the author's sentence structure. Think about what the author is trying to say, then explain it as if you were telling a friend."
        }
    elif similarity_score > 0.70:
        return {
            "status": "borderline",
            "message": "⚡ Getting closer, but still quite similar to the original.",
            "advice": "Focus on the main idea and express it in your natural writing style."
        }
    else:
        return {
            "status": "good",
            "message": "✅ Great summary! You've captured the essence in your own words.",
            "advice": "This shows good understanding. You can add this source to your knowledge library."
        }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Helios - Knowledge Sandbox</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-section h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 150px;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            display: block;
            margin: 30px auto;
            width: 100%;
            max-width: 300px;
        }
        
        .button:hover {
            transform: translateY(-2px);
        }
        
        .button:active {
            transform: translateY(0);
        }
        
        .results {
            display: none;
            margin-top: 30px;
            padding: 30px;
            border-radius: 15px;
            animation: slideIn 0.4s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .results.good {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        
        .results.borderline {
            background: #fff3cd;
            border: 2px solid #ffc107;
        }
        
        .results.too_similar {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }
        
        .results h3 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .similarity-score {
            font-size: 3em;
            font-weight: bold;
            margin: 20px 0;
        }
        
        .advice {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.7);
            border-radius: 10px;
            font-style: italic;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2em;
            }
            .main-card {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>☀️ Helios</h1>
            <p>Knowledge Sandbox - Learn to Summarize Like a Scholar</p>
        </div>
        
        <div class="main-card">
            <div class="input-section">
                <h2>📚 Original Source Text</h2>
                <textarea id="sourceText" placeholder="Paste the original text from your source here..."></textarea>
            </div>
            
            <div class="input-section">
                <h2>✍️ Your Summary</h2>
                <textarea id="summaryText" placeholder="Write your summary in your own words..."></textarea>
            </div>
            
            <button class="button" onclick="checkSimilarity()">
                🔍 Check My Summary
            </button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing with AI...</p>
            </div>
            
            <div class="results" id="results">
                <h3 id="message"></h3>
                <div class="similarity-score" id="score"></div>
                <div class="advice" id="advice"></div>
            </div>
        </div>
    </div>
    
    <script>
        async function checkSimilarity() {
            const sourceText = document.getElementById('sourceText').value.trim();
            const summaryText = document.getElementById('summaryText').value.trim();
            
            if (!sourceText || !summaryText) {
                alert('Please fill in both text areas!');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        source: sourceText,
                        summary: summaryText
                    })
                });
                
                const data = await response.json();
                
                document.getElementById('loading').style.display = 'none';
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.className = 'results ' + data.status;
                resultsDiv.style.display = 'block';
                
                document.getElementById('message').textContent = data.message;
                document.getElementById('score').textContent = 
                    Math.round(data.similarity * 100) + '% Similar';
                document.getElementById('advice').textContent = data.advice;
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Error analyzing text. Please try again.');
                console.error(error);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    source_text = data.get('source', '')
    summary_text = data.get('summary', '')
    
    similarity = calculate_similarity(source_text, summary_text)
    
    feedback = get_feedback(similarity)
    feedback['similarity'] = float(similarity)
    
    return jsonify(feedback)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
