from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import librosa
import numpy as np
import soundfile as sf
import tempfile
import threading
import time
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sounddevice as sd
from collections import deque
import io
import base64

app = Flask(__name__)
CORS(app)

# Global variables for recording
is_recording = False
audio_data = deque()
recording_thread = None
sample_rate = 48000

# Load pre-trained models and scalers
models = {}
label_encoder = None
scaler = None

def load_models():
    """Load pre-trained models and preprocessing objects"""
    global models, label_encoder, scaler
    
    try:
        # Load models
        with open('models/best_model.pkl', 'rb') as f:
            models['best'] = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Try to load individual models for manual selection if available
        optional_models = {
            'rf': 'models/rf.pkl',
            'svm': 'models/svm.pkl',
            'lgbm': 'models/lgbm.pkl',
            'logreg': 'models/logreg.pkl'
        }
        for key, path in optional_models.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        models[key] = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Could not load optional model {key}: {e}")

        # Respect models_meta.json if present (align 'best' alias)
        meta_path = 'models/models_meta.json'
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
                best_key = meta.get('best_key')
                if best_key and best_key in models:
                    models['best'] = models[best_key]
            except Exception as e:
                print(f"⚠️ Could not read models_meta.json: {e}")

        print("✅ Models loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def extract_features(audio_data, sr=48000, n_mfcc=40):
    """Extract features from audio data with tolerant preprocessing for short/quiet clips"""
    try:
        # Ensure numpy array
        audio = np.asarray(audio_data, dtype=np.float32)

        # Pad very short clips to 0.5s to avoid feature failures
        min_seconds = 0.5
        min_len = int(sr * min_seconds)
        if len(audio) < min_len:
            pad_width = min_len - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')

        # Normalize amplitude to avoid clipping and stabilize RMS-based checks
        max_abs = np.max(np.abs(audio)) if audio.size else 0.0
        if max_abs > 0:
            audio = (audio / max_abs) * 0.95

        # RMS energy check (relaxed for quiet inputs)
        rms = librosa.feature.rms(y=audio)
        if np.mean(rms) < 0.0005:
            return None

        # MFCC + deltas
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)

        # Chroma and contrast
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        # Additional features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
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
            return None
        return feat
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_emotion(audio_data, model_key: str = 'best'):
    """Predict emotion from audio data"""
    try:
        features = extract_features(audio_data)
        if features is None:
            return None, None
            
        # Reshape for prediction
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Predict
        model = models.get(model_key) or models['best']
        prediction = model.predict(features_scaled)[0]
        emotion = label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)[0]
            emotions = label_encoder.classes_
            probabilities = dict(zip(emotions, probs.tolist()))
            
        return emotion, probabilities
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None, None

def record_audio():
    """Record audio in a separate thread"""
    global is_recording, audio_data
    
    def audio_callback(indata, frames, time, status):
        if is_recording:
            audio_data.extend(indata[:, 0])  # Mono channel
    
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        while is_recording:
            time.sleep(0.1)

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start audio recording"""
    global is_recording, audio_data, recording_thread
    
    if is_recording:
        return jsonify({'status': 'already_recording'})
    
    # Clear previous audio data
    audio_data.clear()
    is_recording = True
    
    # Start recording thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    
    return jsonify({'status': 'recording_started'})

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop audio recording and return emotion prediction"""
    global is_recording, audio_data, recording_thread
    
    if not is_recording:
        return jsonify({'status': 'not_recording'})
    
    # Stop recording
    is_recording = False
    if recording_thread:
        recording_thread.join()
    
    # Convert audio data to numpy array (tolerate short/quiet recordings)
    if len(audio_data) == 0:
        return jsonify({'status': 'no_audio', 'error': 'No audio data recorded'})

    # Enforce minimum duration requirement 5 seconds
    recorded_duration = len(audio_data) / sample_rate
    if recorded_duration < 5.0:
        return jsonify({'status': 'too_short', 'error': 'Recording is shorter than 5 seconds. Please record at least 5 seconds.'})

    audio_array = np.array(audio_data, dtype=np.float32)
    # If extremely short, pad to 0.5s for feature stability
    min_len = int(sample_rate * 0.5)
    if len(audio_array) < min_len:
        audio_array = np.pad(audio_array, (0, min_len - len(audio_array)), mode='constant')
    
    # Determine requested model (optional)
    requested_model = 'best'
    payload = request.get_json(silent=True) or {}
    model_param = payload.get('model') if isinstance(payload, dict) else None
    if isinstance(model_param, str):
        key = model_param.lower()
        if key in models or key in ('auto', 'best'):
            requested_model = 'best' if key in ('auto', 'best') else key

    # Predict emotion
    emotion, probabilities = predict_emotion(audio_array, model_key=requested_model)
    
    if emotion is None:
        return jsonify({'status': 'prediction_failed', 'error': 'Could not process audio'})
    
    # Save audio file for debugging (optional)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_array, sample_rate)
        audio_file = tmp_file.name
    
    response = {
        'status': 'success',
        'emotion': emotion,
        'probabilities': probabilities,
        'audio_duration': len(audio_array) / sample_rate,
        'audio_file': audio_file,
        'model_used': requested_model
    }
    
    return jsonify(response)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current recording status"""
    return jsonify({
        'is_recording': is_recording,
        'models_loaded': len(models) > 0
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': len(models) > 0})

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load models
    if load_models():
        print("🚀 Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load models. Please train models first.")
