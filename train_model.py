import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def preprocess_features(df, is_train=True):
    df = df.copy()
    
    df['Open Date'] = pd.to_datetime(df['Open Date'], format='%m/%d/%Y')
    df['Open_Year'] = df['Open Date'].dt.year
    df['Open_Month'] = df['Open Date'].dt.month
    df['Open_Day'] = df['Open Date'].dt.day
    df['Days_Since_Open'] = (pd.Timestamp('2015-01-01') - df['Open Date']).dt.days
    
    df['City_Group'] = df['City Group'].map({'Big Cities': 1, 'Other': 0})
    df['Type'] = df['Type'].map({'FC': 1, 'IL': 0})
    
    le_city = LabelEncoder()
    df['City_Encoded'] = le_city.fit_transform(df['City'])
    
    feature_cols = ['Open_Year', 'Open_Month', 'Open_Day', 'Days_Since_Open',
                    'City_Group', 'Type', 'City_Encoded']
    
    for col in df.columns:
        if col.startswith('P') and col[1:].isdigit():
            feature_cols.append(col)
    
    X = df[feature_cols]
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.median())
    
    if is_train:
        y = df['revenue']
        return X, y
    else:
        return X

def train_ensemble_model(X_train, y_train):
    models = {
        'rf': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, 
                                     min_samples_leaf=2, random_state=42, n_jobs=-1),
        'gbm': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                         min_samples_split=5, random_state=42),
        'et': ExtraTreesRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1),
        'ridge': Ridge(alpha=100.0, random_state=42)
    }
    
    print("Training models with cross-validation...")
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name.upper()}...")
        model.fit(X_train, y_train)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, 
                                    scoring='neg_mean_squared_error', n_jobs=-1)
        rmse_scores = np.sqrt(-cv_scores)
        
        print(f"{name.upper()} - CV RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
        
        trained_models[name] = model
    
    return trained_models

def make_predictions(models, X_test):
    predictions = {}
    
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    
    ensemble_pred = (predictions['rf'] * 0.25 + 
                    predictions['gbm'] * 0.25 + 
                    predictions['et'] * 0.25 +
                    predictions['ridge'] * 0.25)
    
    return ensemble_pred

def main():
    print("Loading data...")
    train_df, test_df = load_data()
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    print("\nPreprocessing features...")
    X_train, y_train = preprocess_features(train_df, is_train=True)
    X_test = preprocess_features(test_df, is_train=False)
    
    print(f"Feature shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    models = train_ensemble_model(X_train, y_train)
    
    print("\nMaking predictions on test set...")
    predictions = make_predictions(models, X_test)
    
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'Prediction': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created: submission.csv")
    print(f"Prediction stats - Mean: {predictions.mean():.2f}, Std: {predictions.std():.2f}")
    print(f"Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")

if __name__ == "__main__":
    main()
