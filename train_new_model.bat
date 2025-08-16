@echo off
cls
echo.
echo 🧠 PHISHING DETECTOR MODEL TRAINING
echo ====================================
echo.

echo 🔧 Available training options:
echo.
echo 1. 🎯 Quick Training (Small Dataset)
echo 2. 🏭 Production Training (Medium Dataset) 
echo 3. 🔬 Advanced Training (Large Dataset + Optimization)
echo 4. 📊 Model Comparison (Test Multiple Algorithms)
echo 5. 📁 Custom Dataset Training
echo 6. ❌ Exit
echo.

set /p choice="Select training option (1-6): "

if "%choice%"=="1" (
    echo.
    echo 🎯 Starting Quick Training...
    python train_model.py --dataset data/small_training_set.csv --model-type random_forest
    goto :end
)

if "%choice%"=="2" (
    echo.
    echo 🏭 Starting Production Training...
    python train_model.py --dataset data/medium_training_set.csv --model-type random_forest --cross-validation
    goto :end
)

if "%choice%"=="3" (
    echo.
    echo 🔬 Starting Advanced Training...
    python train_model.py --dataset data/large_training_set.csv --model-type random_forest --hyperparameter-tuning --cross-validation
    goto :end
)

if "%choice%"=="4" (
    echo.
    echo 📊 Starting Model Comparison...
    python train_model.py --dataset data/medium_training_set.csv --compare-models
    goto :end
)

if "%choice%"=="5" (
    echo.
    set /p custom_dataset="Enter path to your dataset (e.g., data/my_dataset.csv): "
    if not "%custom_dataset%"=="" (
        echo 📁 Training with custom dataset: %custom_dataset%
        python train_model.py --dataset "%custom_dataset%" --model-type random_forest --cross-validation
    ) else (
        echo ❌ No dataset path provided.
    )
    goto :end
)

if "%choice%"=="6" (
    echo.
    echo 👋 Exiting...
    goto :exit
)

echo ❌ Invalid choice. Please select 1-6.

:end
echo.
echo ✅ Training completed!
echo.
echo 🧪 Testing the new model...
python test_model.py
echo.
echo 🌐 You can now test the model with:
echo    python web_server.py
echo    or
echo    python check_url.py suspicious-site.com
echo.

:exit
pause
