import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
from pathlib import Path

def train_classifier():
    FEATURES_FILE = Path(r"data/processed/classification/bach/embeddings/uni2/bach_uni2h_features.parquet")
    MODEL_FILE = Path(r"artifacts/models/downstream/classifier.joblib")
    SCALER_FILE = Path(r"artifacts/models/downstream/scaler.joblib")
    
    print("Loading features...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df)} tiles.")
    
    # We want to perform an image-level split, not tile-level split
    # Get unique images and their corresponding class
    image_df = df[['original_image', 'class']].drop_duplicates()
    
    print(f"Total unique images: {len(image_df)}")
    
    # Stratified split on the unique images
    train_imgs, test_imgs = train_test_split(
        image_df['original_image'], 
        test_size=0.2, 
        stratify=image_df['class'],
        random_state=42
    )
    
    train_df = df[df['original_image'].isin(train_imgs)]
    test_df = df[df['original_image'].isin(test_imgs)]
    
    print(f"Train tiles: {len(train_df)} | Test tiles: {len(test_df)}")
    
    # Extract features
    X_train = np.vstack(train_df['embedding'].values)
    y_train = train_df['class'].values
    
    X_test = np.vstack(test_df['embedding'].values)
    y_test = test_df['class'].values
    
    # Map classes to integers
    classes = ["Normal", "Benign", "InSitu", "Invasive"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_train_idx = np.array([class_to_idx[c] for c in y_train])
    y_test_idx = np.array([class_to_idx[c] for c in y_test])
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Logistic Regression classifier...")
    # Logistic Regression is a strong baseline for deep embeddings
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train_scaled, y_train_idx)
    
    print("Evaluating classifier...")
    y_pred_idx = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test_idx, y_pred_idx)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_idx, y_pred_idx, target_names=classes))
    
    try:
        auc = roc_auc_score(y_test_idx, y_pred_proba, multi_class='ovr')
        print(f"ROC AUC (OVR): {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
        
    print(f"\nSaving model and scaler to {MODEL_FILE} and {SCALER_FILE}...")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Done!")

if __name__ == "__main__":
    train_classifier()
