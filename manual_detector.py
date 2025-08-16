#!/usr/bin/env python3
"""
Manual Interactive Phishing URL Detector
Run this to manually test URLs one by one or in batches
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add temp packages to path
sys.path.insert(0, str(Path(__file__).parent / "temp_packages"))
sys.path.insert(0, str(Path(__file__).parent / "temp_extract"))

class ManualPhishingDetector:
    def __init__(self):
        self.extractor = None
        self.model = None
        self.setup_detector()
    
    def setup_detector(self):
        """Initialize the phishing detection components"""
        try:
            print("🔧 Setting up Phishing Detector...")
            
            # Import required modules
            from app.preprocessing import URLFeatureExtractor
            from app.ml_model import PhishingDetectionModel
            
            # Initialize components
            self.extractor = URLFeatureExtractor()
            self.model = PhishingDetectionModel()
            self.model.load_model("models/phishing_detector.joblib")
            
            print(f"✅ Setup complete!")
            print(f"   Model: {self.model.model_type}")
            print(f"   Features: {len(self.model.feature_columns)}")
            print()
            
        except Exception as e:
            print(f"❌ Error setting up detector: {e}")
            print("Make sure you have:")
            print("- models/phishing_detector.joblib")
            print("- app/preprocessing.py")
            print("- app/ml_model.py")
            sys.exit(1)
    
    def analyze_url(self, url):
        """Analyze a single URL for phishing"""
        try:
            # Add http:// if no scheme provided
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            # Extract features
            features = self.extractor.extract_all_features(url)
            
            # Make prediction
            result = self.model.predict(features)
            
            # Determine risk level
            prob = result['phishing_probability']
            if prob >= 0.8:
                risk_level = "🔥 VERY HIGH"
                risk_color = "RED"
            elif prob >= 0.6:
                risk_level = "🚨 HIGH"
                risk_color = "ORANGE"
            elif prob >= 0.4:
                risk_level = "⚠️  MEDIUM"
                risk_color = "YELLOW"
            else:
                risk_level = "✅ LOW"
                risk_color = "GREEN"
            
            return {
                'url': url,
                'is_phishing': result['is_phishing'],
                'phishing_probability': result['phishing_probability'],
                'legitimate_probability': result['legitimate_probability'],
                'confidence': result['confidence'],
                'risk_level': risk_level,
                'risk_color': risk_color,
                'features': features
            }
            
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'is_phishing': None
            }
    
    def display_result(self, analysis):
        """Display analysis results in a nice format"""
        if 'error' in analysis:
            print(f"❌ Error analyzing URL: {analysis['error']}")
            return
        
        print("="*70)
        print(f"🔍 PHISHING ANALYSIS RESULTS")
        print("="*70)
        print(f"🌐 URL: {analysis['url']}")
        print()
        
        # Main verdict
        if analysis['is_phishing']:
            print(f"🚨 VERDICT: PHISHING DETECTED!")
            print(f"   This URL appears to be MALICIOUS")
        else:
            print(f"✅ VERDICT: APPEARS LEGITIMATE")
            print(f"   This URL seems to be safe")
        
        print()
        print(f"📊 RISK ASSESSMENT:")
        print(f"   Risk Level: {analysis['risk_level']}")
        print(f"   Phishing Probability: {analysis['phishing_probability']:.1%}")
        print(f"   Legitimate Probability: {analysis['legitimate_probability']:.1%}")
        print(f"   Confidence: {analysis['confidence']:.1%}")
        
        # Recommendations
        print()
        print(f"💡 RECOMMENDATIONS:")
        if analysis['phishing_probability'] >= 0.8:
            print("   🛑 BLOCK - High risk of phishing attack")
            print("   🚫 Do not enter personal information")
            print("   ⚠️  Report to security team")
        elif analysis['phishing_probability'] >= 0.6:
            print("   ⚠️  CAUTION - Potential phishing risk")
            print("   🔍 Verify domain authenticity")
            print("   🛡️  Use extra caution with personal data")
        elif analysis['phishing_probability'] >= 0.4:
            print("   👀 MONITOR - Some suspicious characteristics")
            print("   🔎 Double-check before entering sensitive data")
        else:
            print("   ✅ PROCEED - Low risk detected")
            print("   🛡️  Standard security precautions apply")
        
        print("="*70)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("🔍 INTERACTIVE PHISHING DETECTOR")
        print("="*50)
        print("Enter URLs to analyze (type 'quit' to exit)")
        print("Examples:")
        print("  - google.com")
        print("  - http://suspicious-paypal-login.fake.com")
        print("  - https://github.com")
        print()
        
        while True:
            try:
                url = input("🌐 Enter URL to check: ").strip()
                
                if url.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not url:
                    continue
                
                print(f"\n🔍 Analyzing: {url}")
                analysis = self.analyze_url(url)
                self.display_result(analysis)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def batch_mode(self, urls):
        """Analyze multiple URLs at once"""
        print(f"🔍 BATCH ANALYSIS - {len(urls)} URLs")
        print("="*70)
        
        results = []
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Analyzing: {url}")
            analysis = self.analyze_url(url)
            results.append(analysis)
            
            # Quick summary
            if 'error' not in analysis:
                status = "🚨 PHISHING" if analysis['is_phishing'] else "✅ SAFE"
                prob = analysis['phishing_probability']
                print(f"         Result: {status} ({prob:.1%} risk)")
            else:
                print(f"         Result: ❌ ERROR")
            print()
        
        # Summary
        safe_count = sum(1 for r in results if r.get('is_phishing') == False)
        phishing_count = sum(1 for r in results if r.get('is_phishing') == True)
        error_count = sum(1 for r in results if 'error' in r)
        
        print("📊 BATCH SUMMARY:")
        print(f"   ✅ Safe URLs: {safe_count}")
        print(f"   🚨 Phishing URLs: {phishing_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"   📊 Total: {len(urls)}")
        
        return results
    
    def demo_mode(self):
        """Run with pre-defined demo URLs"""
        demo_urls = [
            "https://www.google.com",
            "https://github.com",
            "http://paypal-security-alert.fake.com/login",
            "http://amazon-account-suspended.scam.org/verify",
            "https://www.microsoft.com",
            "http://bit.ly/suspicious-link",
            "https://stackoverflow.com",
            "http://facebook-security-check.phishing.net/signin"
        ]
        
        print("🎮 DEMO MODE - Testing with sample URLs")
        print("="*50)
        
        return self.batch_mode(demo_urls)

def main():
    """Main function with menu system"""
    print("🛡️  MANUAL PHISHING DETECTOR")
    print("="*50)
    print("Choose how you want to run the detector:")
    print()
    print("1. 🎮 Demo Mode - Test with sample URLs")
    print("2. 💬 Interactive Mode - Enter URLs manually")
    print("3. 📋 Batch Mode - Test multiple URLs at once")
    print("4. 🔍 Single URL - Test one URL and exit")
    print("5. ❌ Exit")
    print()
    
    detector = ManualPhishingDetector()
    
    while True:
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == '1':
                detector.demo_mode()
                break
                
            elif choice == '2':
                detector.interactive_mode()
                break
                
            elif choice == '3':
                print("\n📋 BATCH MODE")
                print("Enter URLs separated by commas or line breaks:")
                print("Example: google.com, suspicious-site.com, github.com")
                
                urls_input = input("\nEnter URLs: ").strip()
                if urls_input:
                    # Split by comma or newline
                    urls = [url.strip() for url in urls_input.replace('\n', ',').split(',') if url.strip()]
                    if urls:
                        detector.batch_mode(urls)
                break
                
            elif choice == '4':
                url = input("\n🌐 Enter URL to check: ").strip()
                if url:
                    print(f"\n🔍 Analyzing: {url}")
                    analysis = detector.analyze_url(url)
                    detector.display_result(analysis)
                break
                
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
