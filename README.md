# Restaurant Revenue Prediction

A machine learning regression model to predict restaurant revenue using ensemble methods.

## Dataset

- **Training data**: 137 restaurants with historical revenue
- **Test data**: 100,000 restaurants requiring predictions
- **Features**: 
  - Open Date
  - City, City Group (Big Cities/Other)
  - Type (FC/IL)
  - P1-P37: Various restaurant attributes

## Model Architecture

### Ensemble Approach
The solution uses a weighted ensemble of 4 regression models:

1. **Random Forest Regressor** (25% weight)
   - 200 estimators
   - Max depth: 15
   - Optimized for variance reduction

2. **Gradient Boosting Regressor** (25% weight)
   - 200 estimators
   - Learning rate: 0.05
   - Max depth: 5
   - Sequential error correction

3. **Extra Trees Regressor** (25% weight)
   - 200 estimators
   - Max depth: 15
   - Additional randomization for robustness

4. **Ridge Regression** (25% weight)
   - Alpha: 100.0
   - Linear baseline with L2 regularization

### Feature Engineering

- **Temporal features**: Extract year, month, day from Open Date
- **Days since opening**: Calculate restaurant age
- **Categorical encoding**: Label encoding for City
- **Binary mapping**: City Group and Type converted to 0/1
- **Missing value imputation**: Median imputation for all features

## Performance

Cross-validation results (5-fold):
- **Random Forest**: RMSE = 2,484,421 (±513,909)
- **Gradient Boosting**: RMSE = 2,922,799 (±203,782)
- **Extra Trees**: RMSE = 2,445,857 (±563,521)
- **Ridge**: RMSE = 2,591,402 (±569,649)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train_model.py
```

This will:
1. Load and preprocess the data
2. Train all 4 models with cross-validation
3. Generate ensemble predictions
4. Create `submission.csv` with predictions for all test restaurants

## Output

- **submission.csv**: Contains Id and Prediction columns for 100,000 test restaurants
- Prediction statistics:
  - Mean: ~5,000,000
  - Range: 1,096,165 - 13,184,607

## Evaluation Metric

**Root Mean Squared Error (RMSE)**

RMSE = √(Σ(ŷᵢ - yᵢ)² / n)

Where:
- ŷᵢ = predicted revenue
- yᵢ = actual revenue
- n = number of samples

RMSE penalizes large errors more heavily than Mean Absolute Error, making it suitable for this revenue prediction task.
