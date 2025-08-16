# 🧠 Complete Model Training Guide for Windows

## 🚀 Quick Start Training

### **📊 STEP 1: Prepare Your Dataset**

#### **Option A: Use Sample Data (For Testing)**
```cmd
# Use the existing sample dataset
python train_model.py --dataset data/sample_dataset.csv --sample-size 100
```

#### **Option B: Create Your Own Dataset**
Create a CSV file with this format:
```csv
url,label
https://www.google.com,0
https://github.com,0
http://paypal-security-alert.fake.com/login,1
http://amazon-account-suspended.scam.org/verify,1
https://stackoverflow.com,0
```

#### **Option C: Download Public Datasets**
```powershell
# Example: Download PhishTank data (you'll need to register)
# Save as data/new_phishing_dataset.csv
```

---

### **🔧 STEP 2: Basic Training Commands**

#### **🎯 Simple Training (Recommended First Try):**
```cmd
python train_model.py --dataset data/your_dataset.csv
```

#### **🚀 Advanced Training with Options:**
```cmd
python train_model.py ^
    --dataset data/your_dataset.csv ^
    --model-type random_forest ^
    --compare-models ^
    --cross-validation
```

#### **⚡ Quick Test Training (Small Dataset):**
```cmd
python train_model.py ^
    --dataset data/your_dataset.csv ^
    --sample-size 1000 ^
    --model-type random_forest
```

---

### **🎛️ STEP 3: Training Options Explained**

#### **📊 Dataset Options:**
```cmd
--dataset path/to/file.csv          # Your dataset file
--url-column "website"              # If your URL column isn't named "url"
--label-column "is_phishing"        # If your label column isn't named "label"
--sample-size 5000                  # Use only part of dataset for testing
```

#### **🧠 Model Type Options:**
```cmd
--model-type random_forest          # Good balance (recommended)
--model-type xgboost               # Best performance (if available)
--model-type gradient_boost        # Good alternative
--model-type logistic              # Fast training
--model-type svm                   # Good for small datasets
```

#### **🔬 Advanced Options:**
```cmd
--compare-models                   # Test multiple models
--hyperparameter-tuning           # Optimize model parameters
--cross-validation                # Validate model performance
--output-dir models/new_model     # Save model to specific location
```

---

### **📈 STEP 4: Training Examples by Use Case**

#### **🎯 Scenario 1: First Time Training**
```cmd
# Start simple with sample data
python train_model.py ^
    --dataset data/sample_dataset.csv ^
    --model-type random_forest ^
    --sample-size 50
```

#### **🏭 Scenario 2: Production Training**
```cmd
# Full training with your dataset
python train_model.py ^
    --dataset data/my_phishing_data.csv ^
    --model-type random_forest ^
    --cross-validation
```

#### **🔬 Scenario 3: Model Comparison**
```cmd
# Compare different algorithms
python train_model.py ^
    --dataset data/my_phishing_data.csv ^
    --compare-models ^
    --sample-size 10000
```

#### **⚡ Scenario 4: Hyperparameter Optimization**
```cmd
# Find best parameters (slow but optimal)
python train_model.py ^
    --dataset data/my_phishing_data.csv ^
    --model-type random_forest ^
    --hyperparameter-tuning
```

#### **📊 Scenario 5: Custom Column Names**
```cmd
# If your CSV has different column names
python train_model.py ^
    --dataset data/custom_data.csv ^
    --url-column "website_url" ^
    --label-column "is_malicious"
```

---

### **📁 STEP 5: Dataset Preparation Examples**

#### **📝 Create Sample Dataset Script:**
```cmd
# Save this as create_sample_data.py
```

```python
import pandas as pd

# Create sample data
urls = [
    ("https://www.google.com", 0),
    ("https://github.com", 0),
    ("https://stackoverflow.com", 0),
    ("http://paypal-verify.fake.com/login", 1),
    ("http://amazon-suspend.scam.org/verify", 1),
    ("http://microsoft-security.phish.net/update", 1),
    ("https://www.microsoft.com", 0),
    ("https://www.wikipedia.org", 0),
    ("http://facebook-alert.evil.com/signin", 1),
    ("http://google-suspended.fake.net/verify", 1)
]

df = pd.DataFrame(urls, columns=['url', 'label'])
df.to_csv('data/my_sample_dataset.csv', index=False)
print("✅ Sample dataset created!")
```

#### **🔄 Run the Sample Data Creator:**
```cmd
python create_sample_data.py
```

---

### **⚙️ STEP 6: Understanding Training Output**

#### **📊 What You'll See During Training:**
```
🔧 Starting feature extraction...
📊 Processed 1000/5000 URLs
✅ Feature extraction complete. Shape: (5000, 40)
🧠 Training random_forest model...
📈 Test Accuracy: 85.6%
📊 F1 Score: 72.3%
🎯 Model training complete!
💾 Model saved to models/phishing_detector.joblib
```

#### **📋 Training Results Include:**
- **Accuracy Metrics**: How well the model performs
- **Feature Importance**: Which URL patterns matter most
- **Confusion Matrix**: Breakdown of correct/incorrect predictions
- **Model Comparison**: If you used --compare-models

---

### **🔍 STEP 7: Advanced Training Scenarios**

#### **🔄 Retraining Existing Model:**
```cmd
# Backup current model first
copy models\phishing_detector.joblib models\backup_phishing_detector.joblib

# Train new model
python train_model.py ^
    --dataset data\new_dataset.csv ^
    --model-type random_forest
```

#### **📊 Large Dataset Training:**
```cmd
# For datasets > 100k URLs
python train_model.py ^
    --dataset data\large_dataset.csv ^
    --model-type random_forest ^
    --sample-size 50000
```

#### **🎯 Model Performance Tuning:**
```cmd
# Optimize for best performance
python train_model.py ^
    --dataset data\training_data.csv ^
    --model-type random_forest ^
    --hyperparameter-tuning ^
    --cross-validation
```

---

### **📈 STEP 8: Monitoring Training Progress**

#### **📋 Check Training Logs:**
```cmd
# View training progress
type training.log
```

#### **📊 Model Performance Files:**
- **models/model_comparison.csv**: Compare different algorithms
- **models/training_report_[model].json**: Detailed training results
- **training.log**: Training progress and errors

---

### **🛠️ STEP 9: Troubleshooting**

#### **❌ Common Issues:**

**Problem**: "Dataset file not found"
```cmd
# Solution: Check file path
dir data\
python train_model.py --dataset data\correct_filename.csv
```

**Problem**: "Column not found"
```cmd
# Solution: Check column names in your CSV
python -c "import pandas as pd; print(pd.read_csv('data/your_file.csv').columns.tolist())"
```

**Problem**: "Out of memory"
```cmd
# Solution: Use sample size
python train_model.py --dataset data\large_file.csv --sample-size 10000
```

**Problem**: "Training too slow"
```cmd
# Solution: Use random forest instead of SVM
python train_model.py --dataset data\file.csv --model-type random_forest
```

---

### **🎯 STEP 10: Production Deployment**

#### **🔄 Replace Current Model:**
```cmd
# Backup current model
copy models\phishing_detector.joblib models\backup\phishing_detector_backup.joblib

# Copy new model
copy models\new_model.joblib models\phishing_detector.joblib

# Test the new model
python test_model.py
```

#### **🧪 A/B Testing:**
```cmd
# Keep both models and compare
python train_model.py --dataset data\new_data.csv --output-dir models\experimental\
```

---

## 🎉 Quick Training Checklist

1. ✅ **Prepare Dataset**: Create CSV with url,label columns
2. ✅ **Start Simple**: `python train_model.py --dataset data/file.csv`
3. ✅ **Check Results**: Look at accuracy in training output
4. ✅ **Test Model**: `python test_model.py` 
5. ✅ **Deploy**: Replace old model file
6. ✅ **Validate**: Test with web interface

## 📞 Need Help?

- **📋 Check training logs**: `type training.log`
- **🔍 Test current model**: `python test_model.py`
- **🌐 Verify web interface**: `python web_server.py`

**Happy training! 🧠🚀**
