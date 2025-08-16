#!/usr/bin/env python3
"""
Simple command-line URL checker
Usage: python check_url.py [URL]
"""

import sys
import os
from pathlib import Path

# Add temp packages to path
sys.path.insert(0, str(Path(__file__).parent / "temp_packages"))
sys.path.insert(0, str(Path(__file__).parent / "temp_extract"))

def check_url(url):
    """Check a single URL for phishing"""
    try:
        # Import required modules
        from app.preprocessing import URLFeatureExtractor
        from app.ml_model import PhishingDetectionModel
        
        # Initialize components
        extractor = URLFeatureExtractor()
        model = PhishingDetectionModel()
        model.load_model("models/phishing_detector.joblib")
        
        # Add http:// if no scheme provided
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Extract features and predict
        features = extractor.extract_all_features(url)
        result = model.predict(features)
        
        # Display results
        print("🔍 PHISHING DETECTION RESULT")
        print("="*50)
        print(f"URL: {url}")
        print()
        
        if result['is_phishing']:
            print("🚨 RESULT: PHISHING DETECTED!")
            print("   ⚠️  This URL appears to be MALICIOUS")
        else:
            print("✅ RESULT: APPEARS LEGITIMATE")
            print("   ✅ This URL seems to be safe")
        
        print()
        print(f"📊 Details:")
        print(f"   Phishing Probability: {result['phishing_probability']:.1%}")
        print(f"   Confidence: {result['confidence']:.1%}")
        
        # Risk level
        prob = result['phishing_probability']
        if prob >= 0.8:
            risk = "🔥 VERY HIGH"
        elif prob >= 0.6:
            risk = "🚨 HIGH"
        elif prob >= 0.4:
            risk = "⚠️  MEDIUM"
        else:
            risk = "✅ LOW"
        
        print(f"   Risk Level: {risk}")
        print("="*50)
        
        return result['is_phishing']
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("🛡️  SIMPLE URL PHISHING CHECKER")
        print("="*40)
        print("Usage:")
        print(f"  python {sys.argv[0]} [URL]")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} google.com")
        print(f"  python {sys.argv[0]} http://suspicious-site.com")
        print(f"  python {sys.argv[0]} https://github.com")
        print()
        print("Or enter URL interactively:")
        url = input("🌐 Enter URL to check: ").strip()
        if url:
            check_url(url)
    else:
        url = sys.argv[1]
        check_url(url)

if __name__ == "__main__":
    main()
