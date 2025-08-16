#!/usr/bin/env python3
"""
Create sample training datasets for phishing detection
"""

import pandas as pd
import random
from datetime import datetime

def create_sample_dataset(size=100):
    """Create a sample dataset for training"""
    
    # Legitimate URLs (label = 0)
    legitimate_urls = [
        "https://www.google.com",
        "https://www.microsoft.com", 
        "https://github.com",
        "https://stackoverflow.com",
        "https://www.wikipedia.org",
        "https://www.amazon.com",
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.linkedin.com",
        "https://www.youtube.com",
        "https://www.reddit.com",
        "https://www.netflix.com",
        "https://www.adobe.com",
        "https://www.apple.com",
        "https://www.ibm.com",
        "https://www.oracle.com",
        "https://www.salesforce.com",
        "https://www.zoom.us",
        "https://www.dropbox.com",
        "https://www.slack.com",
        "https://mail.google.com",
        "https://docs.google.com",
        "https://drive.google.com",
        "https://office.com",
        "https://outlook.com"
    ]
    
    # Phishing URLs (label = 1)
    phishing_urls = [
        "http://paypal-security-alert.fake.com/login",
        "http://amazon-account-suspended.scam.org/verify",
        "http://microsoft-security-update.phish.net/signin",
        "http://google-account-limited.evil.com/recovery",
        "http://facebook-security-check.malicious.net/signin",
        "http://apple-id-locked.fake.org/unlock",
        "http://netflix-payment-failed.scam.com/billing",
        "http://linkedin-profile-restricted.phish.org/verify",
        "http://twitter-account-suspended.fake.net/appeal",
        "http://github-security-alert.evil.org/verify",
        "http://dropbox-storage-full.scam.net/upgrade",
        "http://slack-workspace-expired.fake.com/renew",
        "http://zoom-meeting-expired.phish.org/rejoin",
        "http://adobe-license-expired.malicious.com/renew",
        "http://office365-mailbox-full.scam.net/cleanup",
        "http://bank-of-america-alert.fake.org/signin",
        "http://chase-security-notice.phish.com/verify",
        "http://wells-fargo-fraud-alert.evil.net/confirm",
        "http://citibank-account-locked.scam.org/unlock",
        "http://usbank-unusual-activity.fake.com/review",
        "http://americanexpress-card-blocked.phish.net/activate",
        "http://visa-transaction-declined.malicious.org/verify",
        "http://mastercard-security-code.scam.com/confirm",
        "http://discover-rewards-expired.fake.net/claim",
        "http://capital-one-data-breach.evil.org/secure"
    ]
    
    # Create balanced dataset
    half_size = size // 2
    
    # Sample URLs
    selected_legitimate = random.choices(legitimate_urls, k=half_size)
    selected_phishing = random.choices(phishing_urls, k=half_size)
    
    # Create DataFrame
    data = []
    
    # Add legitimate URLs
    for url in selected_legitimate:
        data.append({"url": url, "label": 0})
    
    # Add phishing URLs  
    for url in selected_phishing:
        data.append({"url": url, "label": 1})
    
    # Shuffle the data
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    return df

def create_datasets():
    """Create multiple sample datasets"""
    
    print("🔧 Creating sample training datasets...")
    
    # Create different sized datasets
    datasets = [
        ("small_training_set.csv", 50, "Small training set for quick testing"),
        ("medium_training_set.csv", 200, "Medium training set for development"),
        ("large_training_set.csv", 1000, "Large training set for better accuracy")
    ]
    
    for filename, size, description in datasets:
        print(f"📊 Creating {filename} ({size} URLs)...")
        
        df = create_sample_dataset(size)
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False)
        
        # Show statistics
        legitimate_count = (df['label'] == 0).sum()
        phishing_count = (df['label'] == 1).sum()
        
        print(f"   ✅ {description}")
        print(f"   📈 {legitimate_count} legitimate, {phishing_count} phishing URLs")
        print(f"   💾 Saved to {filepath}")
        print()
    
    print("✅ All sample datasets created!")
    print()
    print("🚀 Ready to train! Try these commands:")
    print("   python train_model.py --dataset data/small_training_set.csv")
    print("   python train_model.py --dataset data/medium_training_set.csv --compare-models")
    print("   python train_model.py --dataset data/large_training_set.csv --cross-validation")

if __name__ == "__main__":
    create_datasets()
