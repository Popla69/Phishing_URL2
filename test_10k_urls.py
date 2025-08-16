#!/usr/bin/env python3
"""
Comprehensive test script for 10,000 URLs
Tests both phishing and legitimate URLs for extensive validation
"""

import sys
import os
import random
import time
import json
from pathlib import Path
from datetime import datetime
import csv

# Add temp packages to path
sys.path.insert(0, str(Path(__file__).parent / "temp_packages"))
sys.path.insert(0, str(Path(__file__).parent / "temp_extract"))

def generate_legitimate_urls(count=5000):
    """Generate legitimate URLs from various categories"""
    
    # Major tech companies
    tech_companies = [
        "google.com", "microsoft.com", "apple.com", "amazon.com", "meta.com",
        "netflix.com", "adobe.com", "salesforce.com", "oracle.com", "ibm.com",
        "intel.com", "nvidia.com", "amd.com", "cisco.com", "vmware.com"
    ]
    
    # Popular websites
    popular_sites = [
        "youtube.com", "wikipedia.org", "reddit.com", "stackoverflow.com",
        "github.com", "linkedin.com", "twitter.com", "instagram.com",
        "facebook.com", "tiktok.com", "pinterest.com", "discord.com",
        "slack.com", "zoom.us", "dropbox.com"
    ]
    
    # News and media
    news_sites = [
        "cnn.com", "bbc.com", "reuters.com", "bloomberg.com", "wsj.com",
        "nytimes.com", "washingtonpost.com", "theguardian.com", "forbes.com",
        "techcrunch.com", "wired.com", "ars-technica.com", "engadget.com"
    ]
    
    # Educational institutions
    edu_sites = [
        "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu", "caltech.edu",
        "cmu.edu", "princeton.edu", "yale.edu", "columbia.edu", "uchicago.edu",
        "cornell.edu", "upenn.edu", "duke.edu", "northwestern.edu", "brown.edu"
    ]
    
    # E-commerce and services
    ecommerce_sites = [
        "ebay.com", "etsy.com", "shopify.com", "walmart.com", "target.com",
        "bestbuy.com", "costco.com", "homedepot.com", "lowes.com", "wayfair.com",
        "overstock.com", "newegg.com", "alibaba.com", "aliexpress.com"
    ]
    
    # Government and organizations
    gov_sites = [
        "usa.gov", "irs.gov", "fda.gov", "cdc.gov", "nasa.gov",
        "nih.gov", "fbi.gov", "dhs.gov", "state.gov", "treasury.gov",
        "un.org", "who.int", "worldbank.org", "imf.org", "redcross.org"
    ]
    
    # Combine all legitimate domains
    all_legitimate = tech_companies + popular_sites + news_sites + edu_sites + ecommerce_sites + gov_sites
    
    legitimate_urls = []
    
    # Generate variations with different schemes and paths
    schemes = ["https://", "http://"]
    prefixes = ["", "www.", "m.", "mobile.", "app.", "api."]
    paths = [
        "", "/", "/home", "/about", "/contact", "/products", "/services",
        "/news", "/blog", "/support", "/help", "/login", "/register",
        "/search", "/explore", "/discover", "/trending", "/popular",
        "/account", "/profile", "/settings", "/dashboard", "/admin"
    ]
    
    for _ in range(count):
        domain = random.choice(all_legitimate)
        scheme = random.choice(schemes)
        prefix = random.choice(prefixes)
        path = random.choice(paths)
        
        # Add some query parameters occasionally
        query = ""
        if random.random() < 0.3:  # 30% chance of query parameters
            params = ["q=search", "id=123", "page=1", "sort=date", "filter=all", "lang=en"]
            query = "?" + random.choice(params)
        
        url = f"{scheme}{prefix}{domain}{path}{query}"
        legitimate_urls.append((url, 0))  # 0 = legitimate
    
    return legitimate_urls

def generate_phishing_urls(count=5000):
    """Generate realistic phishing URLs based on common patterns"""
    
    # Common phishing targets
    target_brands = [
        "paypal", "amazon", "microsoft", "google", "apple", "facebook",
        "netflix", "ebay", "linkedin", "instagram", "twitter", "dropbox",
        "adobe", "yahoo", "hotmail", "gmail", "outlook", "icloud"
    ]
    
    # Banking and financial targets
    financial_targets = [
        "bankofamerica", "chase", "wellsfargo", "citibank", "usbank",
        "americanexpress", "visa", "mastercard", "discover", "capitalone"
    ]
    
    # Suspicious domains and TLDs
    suspicious_tlds = [
        ".tk", ".ml", ".ga", ".cf", ".pw", ".click", ".download", ".bid",
        ".win", ".party", ".trade", ".date", ".stream", ".review", ".faith"
    ]
    
    # Phishing domain patterns
    domain_patterns = [
        "{brand}-{action}.{suspicious_domain}",
        "{brand}{action}.{suspicious_domain}", 
        "{brand}-security.{suspicious_domain}",
        "{brand}-verification.{suspicious_domain}",
        "{brand}-update.{suspicious_domain}",
        "{brand}-support.{suspicious_domain}",
        "secure-{brand}.{suspicious_domain}",
        "verify-{brand}.{suspicious_domain}",
        "{brand}-alert.{suspicious_domain}",
        "{brand}-notice.{suspicious_domain}"
    ]
    
    actions = [
        "login", "signin", "verify", "update", "security", "account",
        "support", "help", "alert", "notice", "suspended", "limited",
        "urgent", "immediate", "action", "required", "confirmation"
    ]
    
    suspicious_domains = [
        "securitycenter.com", "verification-center.net", "account-update.org",
        "security-alert.info", "urgent-action.biz", "account-suspended.co",
        "verify-identity.net", "security-notice.org", "account-verification.info",
        "login-security.com", "account-alert.net", "verification-required.org"
    ]
    
    phishing_paths = [
        "/login", "/signin", "/verify", "/update", "/security", "/account",
        "/confirm", "/activate", "/suspend", "/limited", "/blocked", "/locked",
        "/verification", "/authentication", "/security-check", "/account-review",
        "/urgent-action", "/immediate-response", "/click-here", "/verify-now"
    ]
    
    phishing_urls = []
    
    for _ in range(count):
        # Choose pattern type
        pattern_type = random.choice(["domain_spoofing", "subdomain_abuse", "suspicious_tld"])
        
        if pattern_type == "domain_spoofing":
            brand = random.choice(target_brands + financial_targets)
            action = random.choice(actions)
            pattern = random.choice(domain_patterns)
            suspicious_domain = random.choice(suspicious_domains)
            
            domain = pattern.format(
                brand=brand,
                action=action,
                suspicious_domain=suspicious_domain
            )
            
        elif pattern_type == "subdomain_abuse":
            brand = random.choice(target_brands + financial_targets)
            action = random.choice(actions)
            suspicious_domain = random.choice(suspicious_domains)
            domain = f"{brand}-{action}.{suspicious_domain}"
            
        else:  # suspicious_tld
            brand = random.choice(target_brands + financial_targets)
            tld = random.choice(suspicious_tlds)
            domain = f"{brand}-security{tld}"
        
        scheme = random.choice(["http://", "https://"])  # Phishing often uses HTTP
        path = random.choice(phishing_paths)
        
        # Add suspicious query parameters
        query = ""
        if random.random() < 0.6:  # 60% chance for phishing URLs
            suspicious_params = [
                "verify=true", "urgent=1", "action=required", "suspend=false",
                "alert=security", "confirm=account", "update=required", "login=verify"
            ]
            query = "?" + random.choice(suspicious_params)
        
        url = f"{scheme}{domain}{path}{query}"
        phishing_urls.append((url, 1))  # 1 = phishing
    
    return phishing_urls

def test_massive_urls():
    """Test the phishing detection system with 10,000 URLs"""
    try:
        print("🔍 MASSIVE PHISHING DETECTION TEST - 10,000 URLs")
        print("="*70)
        
        # Import required modules
        from app.preprocessing import URLFeatureExtractor
        from app.ml_model import PhishingDetectionModel
        
        # Initialize components
        print("📦 Loading model and feature extractor...")
        extractor = URLFeatureExtractor()
        model = PhishingDetectionModel()
        model.load_model("models/phishing_detector.joblib")
        
        print(f"✓ Model loaded: {model.model_type}")
        print(f"✓ Feature count: {len(model.feature_columns)}")
        
        # Generate test URLs
        print("\n🏗️  GENERATING TEST DATASET")
        print("="*70)
        print("📊 Generating 5,000 legitimate URLs...")
        legitimate_urls = generate_legitimate_urls(5000)
        print("🚨 Generating 5,000 phishing URLs...")
        phishing_urls = generate_phishing_urls(5000)
        
        # Combine and shuffle
        all_urls = legitimate_urls + phishing_urls
        random.shuffle(all_urls)
        
        print(f"✓ Generated {len(all_urls)} total URLs")
        print(f"   - {len(legitimate_urls)} legitimate URLs")
        print(f"   - {len(phishing_urls)} phishing URLs")
        
        # Test in batches
        print("\n🧪 STARTING MASSIVE TESTING")
        print("="*70)
        
        batch_size = 100
        total_batches = len(all_urls) // batch_size
        
        results = {
            "total_tested": 0,
            "correct_predictions": 0,
            "false_positives": 0,  # Legitimate classified as phishing
            "false_negatives": 0,  # Phishing classified as legitimate
            "true_positives": 0,   # Phishing correctly identified
            "true_negatives": 0,   # Legitimate correctly identified
            "processing_times": [],
            "predictions": []
        }
        
        start_time = time.time()
        
        print(f"Processing {len(all_urls)} URLs in {total_batches} batches of {batch_size}...")
        
        for batch_num in range(total_batches + 1):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(all_urls))
            
            if batch_start >= len(all_urls):
                break
                
            batch_urls = all_urls[batch_start:batch_end]
            batch_start_time = time.time()
            
            for url, actual_label in batch_urls:
                try:
                    # Extract features and predict
                    url_start_time = time.time()
                    features = extractor.extract_all_features(url)
                    result = model.predict(features)
                    url_processing_time = time.time() - url_start_time
                    
                    predicted_label = 1 if result['is_phishing'] else 0
                    
                    # Update statistics
                    results["total_tested"] += 1
                    results["processing_times"].append(url_processing_time)
                    
                    if predicted_label == actual_label:
                        results["correct_predictions"] += 1
                        if actual_label == 1:
                            results["true_positives"] += 1
                        else:
                            results["true_negatives"] += 1
                    else:
                        if actual_label == 0 and predicted_label == 1:
                            results["false_positives"] += 1
                        elif actual_label == 1 and predicted_label == 0:
                            results["false_negatives"] += 1
                    
                    # Store prediction details
                    results["predictions"].append({
                        "url": url,
                        "actual": actual_label,
                        "predicted": predicted_label,
                        "probability": result["phishing_probability"],
                        "confidence": result["confidence"],
                        "processing_time_ms": url_processing_time * 1000
                    })
                    
                except Exception as e:
                    print(f"  ⚠️  Error processing URL: {url[:50]}... - {e}")
                    continue
            
            batch_time = time.time() - batch_start_time
            
            # Progress update
            if batch_num % 10 == 0 or batch_num == total_batches:
                accuracy = (results["correct_predictions"] / results["total_tested"]) * 100 if results["total_tested"] > 0 else 0
                avg_time = sum(results["processing_times"]) / len(results["processing_times"]) * 1000 if results["processing_times"] else 0
                
                print(f"  Batch {batch_num+1:3d}/{total_batches} | "
                      f"Tested: {results['total_tested']:5d} | "
                      f"Accuracy: {accuracy:5.1f}% | "
                      f"Avg Time: {avg_time:4.1f}ms | "
                      f"Batch Time: {batch_time:4.1f}s")
        
        total_time = time.time() - start_time
        
        # Calculate final statistics
        accuracy = (results["correct_predictions"] / results["total_tested"]) * 100
        precision = results["true_positives"] / (results["true_positives"] + results["false_positives"]) * 100 if (results["true_positives"] + results["false_positives"]) > 0 else 0
        recall = results["true_positives"] / (results["true_positives"] + results["false_negatives"]) * 100 if (results["true_positives"] + results["false_negatives"]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_processing_time = sum(results["processing_times"]) / len(results["processing_times"]) * 1000
        min_processing_time = min(results["processing_times"]) * 1000
        max_processing_time = max(results["processing_times"]) * 1000
        
        # Display comprehensive results
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("="*70)
        print(f"🔢 Total URLs Tested: {results['total_tested']:,}")
        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        print(f"🏃 Throughput: {results['total_tested']/total_time:.1f} URLs/second")
        
        print("\n📈 ACCURACY METRICS:")
        print("-" * 50)
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        print(f"  Precision:        {precision:.2f}%")
        print(f"  Recall:           {recall:.2f}%")
        print(f"  F1-Score:         {f1_score:.2f}%")
        
        print("\n🎯 CONFUSION MATRIX:")
        print("-" * 50)
        print(f"  True Positives:   {results['true_positives']:,} (Phishing correctly identified)")
        print(f"  True Negatives:   {results['true_negatives']:,} (Legitimate correctly identified)")
        print(f"  False Positives:  {results['false_positives']:,} (Legitimate wrongly flagged)")
        print(f"  False Negatives:  {results['false_negatives']:,} (Phishing missed)")
        
        print("\n⚡ PERFORMANCE METRICS:")
        print("-" * 50)
        print(f"  Average Time:     {avg_processing_time:.2f}ms per URL")
        print(f"  Minimum Time:     {min_processing_time:.2f}ms")
        print(f"  Maximum Time:     {max_processing_time:.2f}ms")
        print(f"  URLs per Second:  {1000/avg_processing_time:.1f}")
        
        # Risk level distribution
        risk_levels = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for pred in results["predictions"]:
            prob = pred["probability"]
            if prob >= 0.7:
                risk_levels["HIGH"] += 1
            elif prob >= 0.4:
                risk_levels["MEDIUM"] += 1
            else:
                risk_levels["LOW"] += 1
        
        print("\n🚨 RISK LEVEL DISTRIBUTION:")
        print("-" * 50)
        print(f"  HIGH Risk:        {risk_levels['HIGH']:,} ({risk_levels['HIGH']/results['total_tested']*100:.1f}%)")
        print(f"  MEDIUM Risk:      {risk_levels['MEDIUM']:,} ({risk_levels['MEDIUM']/results['total_tested']*100:.1f}%)")
        print(f"  LOW Risk:         {risk_levels['LOW']:,} ({risk_levels['LOW']/results['total_tested']*100:.1f}%)")
        
        # Save detailed results
        results_file = f"test_results_10k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Performance rating
        print("\n🏆 SYSTEM PERFORMANCE RATING:")
        print("="*50)
        if accuracy >= 95:
            rating = "🥇 EXCELLENT"
        elif accuracy >= 90:
            rating = "🥈 VERY GOOD"
        elif accuracy >= 85:
            rating = "🥉 GOOD"
        elif accuracy >= 80:
            rating = "👍 ACCEPTABLE"
        else:
            rating = "⚠️  NEEDS IMPROVEMENT"
        
        print(f"Rating: {rating} ({accuracy:.1f}% accuracy)")
        
        print("\n" + "="*70)
        print("🎉 MASSIVE TEST COMPLETED SUCCESSFULLY!")
        print("✅ System tested with 10,000 diverse URLs")
        print("✅ Performance metrics calculated")
        print("✅ Detailed results saved")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_massive_urls()
