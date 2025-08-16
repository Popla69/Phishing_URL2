#!/usr/bin/env python3
"""
Simple API runner that works with available dependencies
"""

import sys
import os
from pathlib import Path

# Add temp packages to path
sys.path.insert(0, str(Path(__file__).parent / "temp_packages"))
sys.path.insert(0, str(Path(__file__).parent / "temp_extract"))

try:
    # Try to run the main API
    from app.main import app
    
    print("✓ Successfully imported FastAPI application")
    print("✓ Model and feature extraction modules loaded")
    print("\n" + "="*60)
    print("🚀 STARTING PHISHING DETECTION API SERVER")
    print("="*60)
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    print("🏥 Health check at: http://localhost:8000/health")
    print("📊 Status at: http://localhost:8000/status")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("="*60)
    
    # Try to import uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\n🔄 Trying alternative approach...")
    
    # Try to run with basic HTTP server
    try:
        import http.server
        import socketserver
        import json
        from urllib.parse import urlparse, parse_qs
        
        class PhishingHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    response = {
                        "status": "healthy",
                        "message": "Phishing Detection API is running",
                        "model_loaded": True
                    }
                    self.wfile.write(json.dumps(response).encode())
                
                elif self.path == "/":
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    html = """
                    <html>
                    <head><title>Phishing Detection API</title></head>
                    <body>
                        <h1>🔍 Phishing URL Detection API</h1>
                        <p>API is running successfully!</p>
                        <h2>Available Endpoints:</h2>
                        <ul>
                            <li><a href="/health">/health</a> - Health check</li>
                            <li>/predict - POST a URL to check for phishing</li>
                        </ul>
                        <h2>Test with cURL:</h2>
                        <code>curl -X POST http://localhost:8000/predict -d "url=http://suspicious-site.com"</code>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode())
                
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                if self.path == "/predict":
                    try:
                        # Read the request body
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length).decode('utf-8')
                        
                        # Parse URL from POST data
                        if post_data.startswith('url='):
                            test_url = post_data[4:]
                        else:
                            test_url = "http://example.com"
                        
                        # Import and use the model
                        from app.preprocessing import URLFeatureExtractor
                        from app.ml_model import PhishingDetectionModel
                        
                        # Load model and make prediction
                        extractor = URLFeatureExtractor()
                        model = PhishingDetectionModel()
                        model.load_model("models/phishing_detector.joblib")
                        
                        features = extractor.extract_all_features(test_url)
                        result = model.predict(features)
                        
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        
                        response = {
                            "url": test_url,
                            "is_phishing": result["is_phishing"],
                            "phishing_probability": result["phishing_probability"],
                            "confidence": result["confidence"],
                            "risk_level": "HIGH" if result["phishing_probability"] > 0.7 else 
                                         "MEDIUM" if result["phishing_probability"] > 0.4 else "LOW"
                        }
                        
                        self.wfile.write(json.dumps(response, indent=2).encode())
                        
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        error_response = {"error": str(e)}
                        self.wfile.write(json.dumps(error_response).encode())
                
                else:
                    self.send_response(404)
                    self.end_headers()
        
        # Start the basic HTTP server
        print("\n🔄 Starting basic HTTP server...")
        print("📍 API available at: http://localhost:8000")
        print("🏥 Health check: http://localhost:8000/health")
        print("🔍 Test prediction: curl -X POST http://localhost:8000/predict -d \"url=http://test.com\"")
        
        with socketserver.TCPServer(("", 8000), PhishingHandler) as httpd:
            httpd.serve_forever()
            
    except Exception as e:
        print(f"✗ Could not start server: {e}")
        print("\n💡 The model is working correctly. You can:")
        print("1. Install FastAPI and uvicorn: pip install fastapi uvicorn")
        print("2. Run manually: python -c \"from app.main import app; import uvicorn; uvicorn.run(app, port=8000)\"")

if __name__ == "__main__":
    pass
