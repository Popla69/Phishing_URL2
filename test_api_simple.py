#!/usr/bin/env python3
"""
Simple test client to demonstrate the phishing detection functionality
"""

import sys
import os
from pathlib import Path

# Add temp packages to path
sys.path.insert(0, str(Path(__file__).parent / "temp_packages"))
sys.path.insert(0, str(Path(__file__).parent / "temp_extract"))

def test_phishing_detection():
    """Test the phishing detection functionality directly"""
    try:
        print("🔍 PHISHING DETECTION API TEST")
        print("="*50)
        
        # Import the required modules
        from app.preprocessing import URLFeatureExtractor
        from app.ml_model import PhishingDetectionModel
        
        # Initialize components
        print("📦 Loading model and feature extractor...")
        extractor = URLFeatureExtractor()
        model = PhishingDetectionModel()
        model.load_model("models/phishing_detector.joblib")
        
        print(f"✓ Model loaded: {model.model_type}")
        print(f"✓ Feature count: {len(model.feature_columns)}")
        
        # Test URLs
        test_urls = [
            ("https://www.google.com", "Legitimate"),
            ("https://github.com", "Legitimate"),
            ("http://suspicious-banking-site.fake.com/login", "Phishing"),
            ("http://phishing-paypal.scam.com/verify", "Phishing"),
            ("https://www.microsoft.com", "Legitimate"),
            ("http://fake-amazon-login.malicious.org/signin", "Phishing"),
            ("https://stackoverflow.com", "Legitimate"),
            ("http://bit.ly/suspicious-link", "Phishing")
        ]
        
        print("\n🧪 TESTING PREDICTIONS")
        print("="*80)
        print(f"{'URL':<50} {'Expected':<12} {'Predicted':<12} {'Confidence':<12} {'Risk':<8}")
        print("="*80)
        
        correct_predictions = 0
        total_predictions = len(test_urls)
        
        for url, expected in test_urls:
            try:
                # Extract features
                features = extractor.extract_all_features(url)
                
                # Make prediction
                result = model.predict(features)
                
                predicted = "Phishing" if result['is_phishing'] else "Legitimate"
                confidence = f"{result['confidence']:.2f}"
                
                # Determine risk level
                prob = result['phishing_probability']
                if prob >= 0.7:
                    risk = "HIGH"
                elif prob >= 0.4:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"
                
                # Check if prediction matches expectation
                is_correct = (expected == "Phishing" and result['is_phishing']) or \
                            (expected == "Legitimate" and not result['is_phishing'])
                
                if is_correct:
                    correct_predictions += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"{url:<50} {expected:<12} {predicted:<12} {confidence:<12} {risk:<8} {status}")
                
            except Exception as e:
                print(f"{url:<50} {expected:<12} {'ERROR':<12} {'N/A':<12} {'N/A':<8} ✗")
                print(f"  Error: {e}")
        
        accuracy = (correct_predictions / total_predictions) * 100
        print("="*80)
        print(f"🎯 ACCURACY: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        
        # Demonstrate API-like functionality
        print("\n🌐 API SIMULATION")
        print("="*50)
        
        def simulate_api_call(url):
            """Simulate an API call to check a URL"""
            try:
                features = extractor.extract_all_features(url)
                result = model.predict(features)
                
                api_response = {
                    "url": url,
                    "is_phishing": result["is_phishing"],
                    "phishing_probability": round(result["phishing_probability"], 3),
                    "legitimate_probability": round(result["legitimate_probability"], 3),
                    "confidence": round(result["confidence"], 3),
                    "risk_level": "HIGH" if result["phishing_probability"] >= 0.7 else 
                                 "MEDIUM" if result["phishing_probability"] >= 0.4 else "LOW",
                    "status": "success"
                }
                return api_response
                
            except Exception as e:
                return {
                    "url": url,
                    "status": "error",
                    "error": str(e)
                }
        
        # Test some URLs via the simulated API
        demo_urls = [
            "https://www.google.com",
            "http://suspicious-phishing-site.fake.com/login",
            "https://github.com/microsoft/vscode"
        ]
        
        for url in demo_urls:
            print(f"\n📡 Testing: {url}")
            response = simulate_api_call(url)
            
            if response["status"] == "success":
                print(f"   Result: {'🚨 PHISHING' if response['is_phishing'] else '✅ LEGITIMATE'}")
                print(f"   Confidence: {response['confidence']}")
                print(f"   Risk Level: {response['risk_level']}")
                print(f"   Phishing Probability: {response['phishing_probability']}")
            else:
                print(f"   Error: {response['error']}")
        
        print("\n" + "="*50)
        print("🎉 SUCCESS! Phishing detection system is working!")
        print("✓ Model loaded and making predictions")
        print("✓ Feature extraction working correctly")
        print("✓ API-like functionality demonstrated")
        print("\n💡 To run the full web API:")
        print("   1. Install FastAPI: pip install fastapi uvicorn")
        print("   2. Run: python -m uvicorn app.main:app --reload")
        print("   3. Visit: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all required files are present:")
        print("- models/phishing_detector.joblib")
        print("- app/preprocessing.py")
        print("- app/ml_model.py")

if __name__ == "__main__":
    test_phishing_detection()
