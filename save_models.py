"""
Script to extract and save trained models from the Jupyter notebook
Run this after training models in the notebook
"""

import os
import pickle
import sys
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import librosa
from collections import Counter

# Import the functions from your notebook
def extract_features(file, n_mfcc=40, target_sr=48000, min_duration=1.0):
    """Same feature extraction function from notebook"""
    try:
        audio, sr = librosa.load(file, sr=target_sr)
        if len(audio) < min_duration * target_sr:
            print(f"⚠️ Skipping short file: {file} (duration < {min_duration}s)")
            return None

        # RMS energy check for cleaning
        rms = librosa.feature.rms(y=audio)
        if np.mean(rms) < 0.005 or np.max(rms) > 0.95:
            print(f"⚠️ Skipping file with invalid RMS: {file} (mean RMS: {np.mean(rms):.4f})")
            return None

        # MFCC + deltas
        mfcc = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc)
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=target_sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=target_sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=target_sr)
        zcr = librosa.feature.zero_crossing_rate(audio)

        # Chroma and contrast
        chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=target_sr)

        # Additional features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=target_sr)
        rms_feat = librosa.feature.rms(y=audio)
        f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        # Stack all features: mean + std
        feat = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta1, axis=1), np.std(delta1, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1),
            np.mean(spec_centroid), np.std(spec_centroid),
            np.mean(spec_bw), np.std(spec_bw),
            np.mean(rolloff), np.std(rolloff),
            np.mean(zcr), np.std(zcr),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(contrast, axis=1), np.std(contrast, axis=1),
            np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),
            np.mean(rms_feat), np.std(rms_feat),
            np.mean(f0), np.std(f0)
        ])

        if not np.isfinite(feat).all():
            print(f"⚠️ Skipping file with invalid features: {file}")
            return None
        return feat
    except Exception as e:
        print(f"⚠️ Error processing {file}: {e}")
        return None

def build_ravdess_dataset(ravdess_dir):
    """Same dataset building function from notebook"""
    features, labels = [], []
    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }
    skipped = 0
    total_files = 0
    
    for root, _, files in os.walk(ravdess_dir):
        for f in files:
            if f.endswith(".wav"):
                total_files += 1
                parts = f.split("-")
                if len(parts) < 3:
                    skipped += 1
                    continue
                emotion = emotion_map.get(parts[2])
                if not emotion:
                    skipped += 1
                    continue
                file_path = os.path.join(root, f)
                feat = extract_features(file_path)
                if feat is not None:
                    features.append(feat)
                    labels.append(emotion)
                else:
                    skipped += 1
    
    print(f"🧹 Cleaned dataset: Processed {total_files} files, skipped {skipped}, kept {len(features)}")
    return np.array(features), np.array(labels)

def train_and_save_models(dataset_dir="../audio_speech_actors_01-24"):
    """Train models and save them for the Flask app"""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("🔎 Extracting features...")
    X, y = build_ravdess_dataset(dataset_dir)
    
    if len(X) == 0:
        raise ValueError("No valid data after cleaning. Check dataset or cleaning thresholds.")
    
    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print("Class distribution before balancing:", Counter(y))

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Balance dataset
    ros = RandomOverSampler(random_state=42)
    X, y_enc = ros.fit_resample(X, y_enc)
    print("Class distribution after balancing:", Counter(y_enc))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Train models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        "Logistic Regression": LogisticRegression(C=1, solver="liblinear", max_iter=1000, random_state=42),
        "SVM": SVC(C=1, kernel="rbf", gamma="scale", probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }

    best_model = None
    best_acc = 0
    best_name = ""
    metrics = {}

    for name, model in models.items():
        print(f"\n⚡ Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        metrics[name] = {"accuracy": float(acc)}
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
        
        print(f"{name} Accuracy: {acc:.4f}")

    print(f"\n🏆 Best Model: {best_name} (Accuracy {best_acc:.4f})")

    # Save the best model and preprocessing objects
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Also save all individual models for manual selection
    model_key_map = {
        "Random Forest": "rf",
        "Logistic Regression": "logreg",
        "SVM": "svm",
        "LightGBM": "lgbm"
    }
    for name, model in models.items():
        key = model_key_map[name]
        with open(f'models/{key}.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Save metadata with metrics and best model reference
    meta = {
        "best_name": best_name,
        "best_key": model_key_map[best_name],
        "metrics": {model_key_map[n]: v for n, v in metrics.items()},
        "label_classes": le.classes_.tolist()
    }
    with open('models/models_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ Models saved successfully!")
    print(f"📁 Saved files:")
    print(f"   - models/best_model.pkl ({best_name})")
    print(f"   - models/scaler.pkl")
    print(f"   - models/label_encoder.pkl")
    print(f"   - models/rf.pkl, models/svm.pkl, models/lgbm.pkl, models/logreg.pkl")
    print(f"   - models/models_meta.json")

if __name__ == "__main__":
    # Check if dataset directory exists
    dataset_dir = "../audio_speech_actors_01-24"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        print("Please ensure the RAVDESS dataset is in the parent directory.")
        sys.exit(1)
    
    try:
        train_and_save_models(dataset_dir)
    except Exception as e:
        print(f"❌ Error training models: {e}")
        sys.exit(1)
