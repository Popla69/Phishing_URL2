#!/usr/bin/env python3
"""
Complete Web Server with HTML Frontend for Phishing Detection
Starts both API backend and serves web frontend
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs, unquote
import html

# Add temp packages to path
sys.path.insert(0, str(Path(__file__).parent / "temp_packages"))
sys.path.insert(0, str(Path(__file__).parent / "temp_extract"))

class PhishingWebServer(http.server.BaseHTTPRequestHandler):
    # Class variables to store the ML components
    extractor = None
    model = None
    
    @classmethod
    def setup_ml_components(cls):
        """Initialize ML components once for the class"""
        if cls.extractor is None:
            try:
                print("🔧 Loading ML components...")
                from app.preprocessing import URLFeatureExtractor
                from app.ml_model import PhishingDetectionModel
                
                cls.extractor = URLFeatureExtractor()
                cls.model = PhishingDetectionModel()
                cls.model.load_model("models/phishing_detector.joblib")
                print(f"✅ ML components loaded: {cls.model.model_type}")
                
            except Exception as e:
                print(f"❌ Error loading ML components: {e}")
                raise
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/" or self.path == "/index.html":
            self.serve_frontend()
        elif self.path == "/health":
            self.serve_health_check()
        elif self.path == "/api/health":
            self.serve_api_health()
        elif self.path.startswith("/static/"):
            self.serve_static_files()
        else:
            self.send_error(404, "Page not found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/api/check-url":
            self.handle_single_url_check()
        elif self.path == "/api/check-batch":
            self.handle_batch_url_check()
        else:
            self.send_error(404, "API endpoint not found")
    
    def serve_frontend(self):
        """Serve the main HTML frontend"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Phishing URL Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .tab-container {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        
        .tab {
            background: #f8f9fa;
            border: none;
            padding: 15px 30px;
            margin: 0 5px;
            cursor: pointer;
            border-radius: 10px;
            font-size: 1.1em;
            transition: all 0.3s ease;
        }
        
        .tab.active {
            background: #667eea;
            color: white;
        }
        
        .tab:hover {
            transform: translateY(-2px);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #333;
        }
        
        .url-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1.1em;
            transition: border-color 0.3s ease;
        }
        
        .url-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .batch-input {
            width: 100%;
            min-height: 200px;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1em;
            resize: vertical;
            font-family: monospace;
        }
        
        .batch-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-right: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .result-item {
            background: white;
            padding: 20px;
            margin: 10px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .result-safe {
            border-left: 5px solid #28a745;
        }
        
        .result-phishing {
            border-left: 5px solid #dc3545;
        }
        
        .result-url {
            font-weight: 600;
            margin-bottom: 10px;
            word-break: break-all;
        }
        
        .result-verdict {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        
        .result-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .detail-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stats {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        
        .stats h3 {
            margin-bottom: 15px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        
        .examples {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .examples h3 {
            margin-bottom: 15px;
            color: #1976d2;
        }
        
        .example-urls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .example-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .example-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .file-upload {
            border: 2px dashed #667eea;
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-upload:hover {
            background: #f8f9fa;
        }
        
        .file-upload.dragover {
            border-color: #764ba2;
            background: #f0f8ff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Phishing URL Detector</h1>
            <p>Advanced ML-powered phishing detection system</p>
            <p>Accuracy: 92.6% | Speed: 12.7ms per URL</p>
        </div>
        
        <div class="main-content">
            <div class="tab-container">
                <button class="tab active" onclick="switchTab('single')">🔍 Single URL</button>
                <button class="tab" onclick="switchTab('batch')">📋 Batch Check</button>
                <button class="tab" onclick="switchTab('file')">📁 File Upload</button>
            </div>
            
            <!-- Single URL Tab -->
            <div id="single-tab" class="tab-content active">
                <div class="input-group">
                    <label for="single-url">🌐 Enter URL to check:</label>
                    <input type="text" id="single-url" class="url-input" 
                           placeholder="e.g., https://suspicious-site.com or google.com"
                           onkeypress="if(event.key==='Enter') checkSingleURL()">
                </div>
                
                <button class="btn" onclick="checkSingleURL()">🔍 Check URL</button>
                <button class="btn" onclick="clearResults()">🗑️ Clear Results</button>
                
                <div class="examples">
                    <h3>💡 Try these examples:</h3>
                    <div class="example-urls">
                        <div class="example-item" onclick="setExampleURL('https://www.google.com')">
                            <strong>✅ Safe URL:</strong><br>
                            https://www.google.com
                        </div>
                        <div class="example-item" onclick="setExampleURL('http://paypal-security-alert.fake.com/login')">
                            <strong>🚨 Phishing URL:</strong><br>
                            http://paypal-security-alert.fake.com/login
                        </div>
                        <div class="example-item" onclick="setExampleURL('https://github.com')">
                            <strong>✅ Safe URL:</strong><br>
                            https://github.com
                        </div>
                        <div class="example-item" onclick="setExampleURL('http://amazon-account-suspended.scam.org/verify')">
                            <strong>🚨 Phishing URL:</strong><br>
                            http://amazon-account-suspended.scam.org/verify
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Batch URLs Tab -->
            <div id="batch-tab" class="tab-content">
                <div class="input-group">
                    <label for="batch-urls">📋 Enter multiple URLs (one per line):</label>
                    <textarea id="batch-urls" class="batch-input" 
                              placeholder="https://www.google.com
http://suspicious-paypal.fake.com/login
https://github.com
http://amazon-phishing.scam.org/verify
https://stackoverflow.com"></textarea>
                </div>
                
                <button class="btn" onclick="checkBatchURLs()">📊 Check All URLs</button>
                <button class="btn" onclick="loadExampleBatch()">📝 Load Examples</button>
                <button class="btn" onclick="clearResults()">🗑️ Clear Results</button>
            </div>
            
            <!-- File Upload Tab -->
            <div id="file-tab" class="tab-content">
                <div class="input-group">
                    <label>📁 Upload a text file with URLs:</label>
                    <div class="file-upload" id="file-upload" onclick="document.getElementById('file-input').click()">
                        <div>📎 Click to select file or drag and drop</div>
                        <div>Supported formats: .txt, .csv</div>
                        <div>One URL per line</div>
                    </div>
                    <input type="file" id="file-input" accept=".txt,.csv" style="display:none" onchange="handleFileSelect(this)">
                </div>
                
                <button class="btn" onclick="processUploadedFile()">🔍 Process File</button>
                <button class="btn" onclick="clearResults()">🗑️ Clear Results</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>🔍 Analyzing URLs... Please wait</div>
            </div>
            
            <div class="results" id="results"></div>
        </div>
    </div>

    <script>
        let uploadedFileContent = '';
        
        function switchTab(tab) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tab + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            // Clear results when switching tabs
            clearResults();
        }
        
        function setExampleURL(url) {
            document.getElementById('single-url').value = url;
        }
        
        function loadExampleBatch() {
            const exampleUrls = `https://www.google.com
http://paypal-security-alert.fake.com/login
https://github.com
http://amazon-account-suspended.scam.org/verify
https://www.microsoft.com
http://bit.ly/suspicious-link
https://stackoverflow.com
http://facebook-security-check.phishing.net/signin`;
            document.getElementById('batch-urls').value = exampleUrls;
        }
        
        function clearResults() {
            document.getElementById('results').innerHTML = '';
            document.getElementById('results').classList.remove('show');
        }
        
        function showLoading() {
            document.getElementById('loading').classList.add('show');
            document.querySelectorAll('.btn').forEach(btn => btn.disabled = true);
        }
        
        function hideLoading() {
            document.getElementById('loading').classList.remove('show');
            document.querySelectorAll('.btn').forEach(btn => btn.disabled = false);
        }
        
        async function checkSingleURL() {
            const url = document.getElementById('single-url').value.trim();
            if (!url) {
                alert('Please enter a URL to check');
                return;
            }
            
            showLoading();
            
            try {
                const response = await fetch('/api/check-url', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url: url})
                });
                
                const data = await response.json();
                displaySingleResult(data);
                
            } catch (error) {
                console.error('Error:', error);
                displayError('Error checking URL: ' + error.message);
            } finally {
                hideLoading();
            }
        }
        
        async function checkBatchURLs() {
            const urlText = document.getElementById('batch-urls').value.trim();
            if (!urlText) {
                alert('Please enter URLs to check');
                return;
            }
            
            const urls = urlText.split('\\n').map(url => url.trim()).filter(url => url);
            if (urls.length === 0) {
                alert('No valid URLs found');
                return;
            }
            
            showLoading();
            
            try {
                const response = await fetch('/api/check-batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({urls: urls})
                });
                
                const data = await response.json();
                displayBatchResults(data);
                
            } catch (error) {
                console.error('Error:', error);
                displayError('Error checking URLs: ' + error.message);
            } finally {
                hideLoading();
            }
        }
        
        function handleFileSelect(input) {
            const file = input.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedFileContent = e.target.result;
                document.querySelector('.file-upload').innerHTML = 
                    `<div>✅ File loaded: ${file.name}</div>
                     <div>${uploadedFileContent.split('\\n').length} lines</div>`;
            };
            reader.readAsText(file);
        }
        
        function processUploadedFile() {
            if (!uploadedFileContent) {
                alert('Please select a file first');
                return;
            }
            
            const urls = uploadedFileContent.split('\\n').map(url => url.trim()).filter(url => url);
            if (urls.length === 0) {
                alert('No valid URLs found in file');
                return;
            }
            
            // Set URLs in batch textarea and process
            document.getElementById('batch-urls').value = urls.join('\\n');
            checkBatchURLs();
        }
        
        function displaySingleResult(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            
            if (data.error) {
                displayError(data.error);
                return;
            }
            
            const resultHTML = createResultHTML(data);
            resultsDiv.innerHTML = resultHTML;
            resultsDiv.classList.add('show');
        }
        
        function displayBatchResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            
            if (data.error) {
                displayError(data.error);
                return;
            }
            
            let html = '<h3>📊 Batch Analysis Results</h3>';
            
            // Summary stats
            const safeCount = data.results.filter(r => !r.is_phishing).length;
            const phishingCount = data.results.filter(r => r.is_phishing).length;
            
            html += `<div class="stats">
                <h3>Summary</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-number">${data.total_processed}</span>
                        <div>Total URLs</div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${safeCount}</span>
                        <div>✅ Safe</div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${phishingCount}</span>
                        <div>🚨 Phishing</div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${data.processing_time_ms.toFixed(0)}ms</span>
                        <div>⚡ Total Time</div>
                    </div>
                </div>
            </div>`;
            
            // Individual results
            data.results.forEach(result => {
                html += createResultHTML(result);
            });
            
            resultsDiv.innerHTML = html;
            resultsDiv.classList.add('show');
        }
        
        function createResultHTML(data) {
            const isPhishing = data.is_phishing;
            const resultClass = isPhishing ? 'result-phishing' : 'result-safe';
            const verdict = isPhishing ? '🚨 PHISHING DETECTED' : '✅ APPEARS SAFE';
            const riskLevel = getRiskLevel(data.phishing_probability);
            
            return `
                <div class="result-item ${resultClass}">
                    <div class="result-url">🌐 ${data.url}</div>
                    <div class="result-verdict">${verdict}</div>
                    <div class="result-details">
                        <div class="detail-item">
                            <strong>Risk Level:</strong><br>
                            ${riskLevel}
                        </div>
                        <div class="detail-item">
                            <strong>Phishing Probability:</strong><br>
                            ${(data.phishing_probability * 100).toFixed(1)}%
                        </div>
                        <div class="detail-item">
                            <strong>Confidence:</strong><br>
                            ${(data.confidence * 100).toFixed(1)}%
                        </div>
                        <div class="detail-item">
                            <strong>Processing Time:</strong><br>
                            ${data.processing_time_ms ? data.processing_time_ms.toFixed(1) + 'ms' : 'N/A'}
                        </div>
                    </div>
                </div>
            `;
        }
        
        function getRiskLevel(probability) {
            if (probability >= 0.8) return '🔥 VERY HIGH';
            if (probability >= 0.6) return '🚨 HIGH';
            if (probability >= 0.4) return '⚠️ MEDIUM';
            return '✅ LOW';
        }
        
        function displayError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="result-item" style="border-left: 5px solid #ffc107;">
                    <div class="result-verdict">❌ Error</div>
                    <div>${message}</div>
                </div>
            `;
            resultsDiv.classList.add('show');
        }
        
        // File drag and drop
        const fileUpload = document.getElementById('file-upload');
        
        fileUpload.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUpload.classList.add('dragover');
        });
        
        fileUpload.addEventListener('dragleave', () => {
            fileUpload.classList.remove('dragover');
        });
        
        fileUpload.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUpload.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('file-input').files = files;
                handleFileSelect(document.getElementById('file-input'));
            }
        });
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def serve_health_check(self):
        """Serve basic health check"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        
        health_html = f"""
        <html>
        <head><title>Health Check</title></head>
        <body>
            <h1>🛡️ Phishing Detector Health Check</h1>
            <p>✅ Server is running</p>
            <p>✅ Model loaded: {self.model.model_type if self.model else 'Not loaded'}</p>
            <p>⏰ Time: {datetime.now()}</p>
            <p><a href="/">← Back to Main Interface</a></p>
        </body>
        </html>
        """
        self.wfile.write(health_html.encode())
    
    def serve_api_health(self):
        """Serve API health check as JSON"""
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": self.model is not None,
            "model_type": self.model.model_type if self.model else None
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(health_data).encode())
    
    def handle_single_url_check(self):
        """Handle single URL check API"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            try:
                data = json.loads(post_data.decode('utf-8'))
                url = data.get('url', '').strip()
            except json.JSONDecodeError:
                # Fallback to form data
                post_str = post_data.decode('utf-8')
                if post_str.startswith('url='):
                    url = unquote(post_str[4:])
                else:
                    url = post_str
            
            if not url:
                self.send_error_response("URL is required")
                return
            
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            # Analyze URL
            start_time = time.time()
            features = self.extractor.extract_all_features(url)
            result = self.model.predict(features)
            processing_time = (time.time() - start_time) * 1000
            
            # Prepare response
            response_data = {
                "url": url,
                "is_phishing": result["is_phishing"],
                "phishing_probability": result["phishing_probability"],
                "legitimate_probability": result["legitimate_probability"],
                "confidence": result["confidence"],
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_json_response(response_data)
            
        except Exception as e:
            self.send_error_response(f"Error processing URL: {str(e)}")
    
    def handle_batch_url_check(self):
        """Handle batch URL check API"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            urls = data.get('urls', [])
            if not urls:
                self.send_error_response("URLs array is required")
                return
            
            # Process URLs
            start_time = time.time()
            results = []
            
            for url in urls:
                try:
                    # Add protocol if missing
                    if not url.startswith(('http://', 'https://')):
                        url = 'http://' + url
                    
                    # Analyze URL
                    url_start_time = time.time()
                    features = self.extractor.extract_all_features(url)
                    result = self.model.predict(features)
                    url_processing_time = (time.time() - url_start_time) * 1000
                    
                    results.append({
                        "url": url,
                        "is_phishing": result["is_phishing"],
                        "phishing_probability": result["phishing_probability"],
                        "legitimate_probability": result["legitimate_probability"],
                        "confidence": result["confidence"],
                        "processing_time_ms": url_processing_time
                    })
                    
                except Exception as e:
                    results.append({
                        "url": url,
                        "error": str(e),
                        "is_phishing": None
                    })
            
            total_time = (time.time() - start_time) * 1000
            
            # Prepare response
            response_data = {
                "results": results,
                "total_processed": len(results),
                "processing_time_ms": total_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_json_response(response_data)
            
        except Exception as e:
            self.send_error_response(f"Error processing batch: {str(e)}")
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def send_error_response(self, message):
        """Send error response"""
        error_data = {
            "error": message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(error_data).encode())
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")

def start_web_server(port=8000):
    """Start the web server"""
    print("🚀 PHISHING DETECTION WEB SERVER")
    print("="*50)
    
    # Initialize ML components
    try:
        PhishingWebServer.setup_ml_components()
    except Exception as e:
        print(f"❌ Failed to load ML components: {e}")
        return
    
    # Start server
    try:
        with socketserver.TCPServer(("", port), PhishingWebServer) as httpd:
            print(f"✅ Server started successfully!")
            print()
            print("📍 ENDPOINTS:")
            print(f"   🌐 Web Interface: http://localhost:{port}")
            print(f"   🏥 Health Check:  http://localhost:{port}/health")
            print(f"   🔗 API Health:    http://localhost:{port}/api/health")
            print()
            print("💡 FEATURES:")
            print("   ✅ Single URL checking")
            print("   ✅ Batch URL processing")
            print("   ✅ File upload support")
            print("   ✅ Real-time results")
            print("   ✅ Beautiful web interface")
            print()
            print("🛑 Press Ctrl+C to stop the server")
            print("="*50)
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    start_web_server()
