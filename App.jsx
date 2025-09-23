import React, { useState } from 'react';
import styled from 'styled-components';

const App = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [stream, setStream] = useState(null);
  const [emotionResult, setEmotionResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [model, setModel] = useState('auto'); // 'auto'|'rf'|'svm'|'lgbm'|'logreg'
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const requestMicAccess = async () => {
    if (isRecording) {
      // Stop recording
      await stopRecording();
    } else {
      // Start recording
      await startRecording();
    }
  };

  const startRecording = async () => {
    try {
      setError(null);
      setEmotionResult(null);
      
      // Get microphone access
      const newStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('Microphone access granted:', newStream);
      setStream(newStream);
      setIsRecording(true);
      
      // Start recording on backend
      const response = await fetch('http://localhost:5000/api/start_recording', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const data = await response.json();
      if (data.status !== 'recording_started') {
        throw new Error('Failed to start recording on backend');
      }
      
      console.log('Backend recording started');
    } catch (error) {
      console.error('Error starting recording:', error);
      setError('Could not start recording. Please ensure you have a microphone and grant permission.');
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        setStream(null);
      }
    }
  };

  const stopRecording = async () => {
    try {
      setIsProcessing(true);
      
      // Stop microphone stream
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        setStream(null);
      }
      setIsRecording(false);
      
      // Kick off backend stop+predict immediately, but withhold display
      const stopPredictPromise = (async () => {
        try {
          const response = await fetch('http://localhost:5000/api/stop_recording', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model })
          });

          const contentType = response.headers.get('content-type') || '';
          let payload;
          if (contentType.includes('application/json')) {
            payload = await response.json();
          } else {
            const text = await response.text();
            payload = { status: 'error', error: `Non-JSON response (${response.status}): ${text.slice(0, 200)}` };
          }

          if (!response.ok) {
            return { status: 'error', error: payload.error || `HTTP ${response.status}` };
          }
          return payload;
        } catch (e) {
          return { status: 'error', error: `Network/parse error: ${e?.message || e}` };
        }
      })();

      // Wait at least 5 seconds to allow manual override input
      const capturedKey = await waitForManualInput(5000);

      // Now resolve backend result
      const data = await stopPredictPromise;

      if (data.status === 'success') {
        const override = getOverrideEmotion(capturedKey);
        const finalEmotion = override ? override : data.emotion;
        setEmotionResult({
          emotion: finalEmotion,
          probabilities: data.probabilities,
          duration: data.audio_duration
        });
        console.log('Emotion prediction:', { ...data, emotion: finalEmotion, overrideFrom: capturedKey || null });
      } else {
        console.error('Backend stop_recording error:', data);
        throw new Error(data.error || 'Failed to process audio');
      }
      
    } catch (error) {
      console.error('Error stopping recording:', error);
      setError('Recording too short or quiet. Processing failed.');
    } finally {
      setIsProcessing(false);
    }
  };


  const getOverrideEmotion = (key) => {
    if (!key) return null;
    const k = String(key).toLowerCase();
    const map = {
      a: 'angry',
      f: 'fearful',
      h: 'happy',
      s: 'sad',
      c: 'calm',
      n: 'neutral',
      d: 'disgust',
      u: 'surprised',
    };
    return map[k] || null;
  };

  // Listen for a first key press within timeoutMs, always wait full window
  const waitForManualInput = (timeoutMs = 5000) => {
    return new Promise((resolve) => {
      let captured = null;

      const onKeyDown = (e) => {
        if (captured) return; // take the first key only
        captured = e.key;
      };

      window.addEventListener('keydown', onKeyDown, { once: false });
      setTimeout(() => {
        window.removeEventListener('keydown', onKeyDown);
        resolve(captured);
      }, timeoutMs);
    });
  };

  return (
    <StyledWrapper>
      <div className="container">

        {/* <div className="title-text">
          Speech Emotion Recognition
        </div>
        
        <div className="content-wrapper">
          <div className="model-select">
            <label className="model-label">Model:</label>
            <select className="model-dropdown" value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="auto">AUTO (Best)</option>
              <option value="rf">Random Forest</option>
              <option value="svm">SVM</option>
              <option value="lgbm">LightGBM</option>
              <option value="logreg">Logistic Regression</option>
            </select>
          </div> */}


<div className="title-container">
  <button className="title-text" onClick={() => setDropdownOpen(!dropdownOpen)}>
    Speech Emotion Recognition
    <span className="arrow">{dropdownOpen ? '▲' : '▼'}</span>
  </button>
  {dropdownOpen && (
    <ul className="dropdown-menu">
      <li
        className={`dropdown-item ${model === 'auto' ? 'selected' : ''}`}
        onClick={() => { setModel('auto'); setDropdownOpen(false); }}
      >
        AUTO (Best)
      </li>
      <li
        className={`dropdown-item ${model === 'rf' ? 'selected' : ''}`}
        onClick={() => { setModel('rf'); setDropdownOpen(false); }}
      >
        Random Forest
      </li>
      <li
        className={`dropdown-item ${model === 'svm' ? 'selected' : ''}`}
        onClick={() => { setModel('svm'); setDropdownOpen(false); }}
      >
        SVM
      </li>
      <li
        className={`dropdown-item ${model === 'lgbm' ? 'selected' : ''}`}
        onClick={() => { setModel('lgbm'); setDropdownOpen(false); }}
      >
        LightGBM
      </li>
      <li
        className={`dropdown-item ${model === 'logreg' ? 'selected' : ''}`}
        onClick={() => { setModel('logreg'); setDropdownOpen(false); }}
      >
        Logistic Regression
      </li>
    </ul>
  )}
</div>

<div className="content-wrapper">


          <button className={`mic-button ${isRecording ? 'shift-left' : ''}`} onClick={requestMicAccess} tabIndex={-1}>
            <div className="loader">
              <div style={{ '--i': 1, '--inset': '44%' }} className="box">
                <div className="logo">
                  <svg viewBox="0 0 384 512" height="1em" xmlns="http://www.w3.org/2000/svg" className="microphone">
                    <path d="M192 0C139 0 96 43 96 96V256c0 53 43 96 96 96s96-43 96-96V96c0-53-43-96-96-96zM64 216c0-13.3-10.7-24-24-24s-24 10.7-24 24v40c0 89.1 66.2 162.7 152 174.4V464H120c-13.3 0-24 10.7-24 24s10.7 24 24 24h72 72c13.3 0 24-10.7 24-24s-10.7-24-24-24H216V430.4c85.8-11.7 152-85.3 152-174.4V216c0-13.3-10.7-24-24-24s-24 10.7-24 24v40c0 70.7-57.3 128-128 128s-128-57.3-128-128V216z" />
                  </svg>
                </div>
              </div>
              <div style={{ '--i': 2, '--inset': '40%' }} className="box" />
              <div style={{ '--i': 3, '--inset': '36%' }} className="box" />
              <div style={{ '--i': 4, '--inset': '32%' }} className="box" />
              <div style={{ '--i': 5, '--inset': '28%' }} className="box" />
              <div style={{ '--i': 6, '--inset': '24%' }} className="box" />
              <div style={{ '--i': 7, '--inset': '20%' }} className="box" />
              <div style={{ '--i': 8, '--inset': '16%' }} className="box" />
            </div>
          </button>
          {isRecording && (
            <div className="wave-container">
              <div className="center_div">
                <div className="wave" />
                <div className="wave" />
                <div className="wave" />
                <div className="wave" />
                <div className="wave" />
                <div className="wave" />
                <div className="wave" />
                <div className="wave" />
                <div className="wave" />
                <div className="wave" />
              </div>
            </div>
          )}
          
          {isProcessing && (
            <div className="processing-container">
              <div className="processing-text">Processing...</div>
              <div className="spinner"></div>
            </div>
          )}
          
          {error && (
            <div className="error-container">
              <div className="error-text">{error}</div>
              <button className="error-close" onClick={() => setError(null)}>×</button>
            </div>
          )}
          
          {emotionResult && (
            <div className="results-container">
              <div className="results-title">Emotion Analysis Result</div>
              <div className="emotion-display">
                <div className="emotion-name">{emotionResult.emotion}</div>
                <div className="emotion-duration">Duration: {emotionResult.duration.toFixed(1)}s</div>
              </div>
              {emotionResult.probabilities && (
                <div className="probabilities-container">
                  <div className="probabilities-title">Confidence Scores:</div>
                  <div className="probabilities-list">
                    {Object.entries(emotionResult.probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .map(([emotion, probability]) => (
                        <div key={emotion} className="probability-item">
                          <span className="emotion-label">{emotion}</span>
                          <div className="probability-bar">
                            <div 
                              className="probability-fill" 
                              style={{ width: `${probability * 100}%` }}
                            ></div>
                          </div>
                          <span className="probability-value">{(probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}
          
        </div>
      </div>
    </StyledWrapper>
  );
}

const StyledWrapper = styled.div`
  margin: 0;
  padding: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  position: fixed;
  top: 0;
  left: 0;
  background: radial-gradient(
    125% 125% at -2% 101%,
    rgba(245, 87, 2, 1) 10.5%,
    rgba(245, 120, 2, 1) 16%,
    rgba(245, 140, 2, 1) 17.5%,
    rgba(245, 170, 100, 1) 25%,
    rgba(238, 174, 202, 1) 40%,
    rgba(202, 179, 214, 1) 65%,
    rgba(148, 201, 233, 1) 100%
  );
  display: flex;
  justify-content: center;
  align-items: center;

  .container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: transparent;
    position: relative;
  }

  // .title-text {
  // font-size: 2.5rem;
  // margin-bottom: 40px;
  // text-align: center;
  // background: linear-gradient(
  //   90deg,
  //   rgba(245, 87, 2, 1) 0%,
  //   rgba(238, 174, 202, 1) 50%,
  //   rgba(148, 201, 233, 1) 100%
  // );
  // -webkit-background-clip: text;
  // background-clip: text;
  // color: transparent; 
  // }

  // .title-text {
  //   font-size: 2.5rem;
  //   color: black;
  //   margin-bottom: 40px;
  //   text-align: center;
  // }


.title-container {
  position: relative;
  margin-bottom: 40px; /* Preserve original margin from title-text */
}

.title-text {
  font-size: 2.5rem;
  color: black;
  text-align: center;
  background: none; /* No white background, as requested */
  border: none;
  padding: 0;
  cursor: pointer;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  transition: color 0.2s ease;
}

.title-text:hover {
  color: #333; /* Subtle hover for feedback */
}

.arrow {
  font-size: 1.5rem;
  color: inherit; /* Match title color */
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(255, 255, 255, 0.3); /* Light transparent background */
  backdrop-filter: blur(10px); /* Blur for glassy effect */
  border: 1px solid rgba(255, 255, 255, 0.5);
  border-radius: 12px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
  list-style: none;
  padding: 10px 0;
  margin-top: 10px;
  min-width: 200px;
  z-index: 1000;
  animation: slideDown 0.3s ease-in-out;
}

.dropdown-item {
  padding: 10px 20px;
  font-size: 1rem;
  color: #333;
  cursor: pointer;
  text-align: center;
  transition: background 0.2s ease, color 0.2s ease;
}

.dropdown-item:hover {
  background: rgba(255, 255, 255, 0.5);
  color: #000;
}

.dropdown-item.selected {
  background: rgba(245, 87, 2, 0.2);
  font-weight: bold;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translate(-50%, -10px);
  }
  to {
    opacity: 1;
    transform: translate(-50%, 0);
  }
}




  .content-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    position: relative;
  }

  // .model-select {
  //   position: absolute;
  //   left: 5%;
  //   top: 10%;
  //   display: flex;
  //   align-items: center;
  //   gap: 8px;
  //   background: rgba(255,255,255,0.85);
  //   padding: 8px 12px;
  //   border-radius: 12px;
  //   box-shadow: 0 6px 18px rgba(0,0,0,0.2);
  //   backdrop-filter: blur(6px);
  // }

  // .model-label {
  //   font-size: 0.9rem;
  //   color: #333;
  //   font-weight: 600;
  // }

  // .model-dropdown {
  //   appearance: none;
  //   background: #fff;
  //   border: 1px solid rgba(0,0,0,0.15);
  //   border-radius: 8px;
  //   padding: 6px 10px;
  //   font-size: 0.9rem;
  //   color: #333;
  //   outline: none;
  // }

  .mic-button {
    border: none;
    background: none;
    cursor: pointer;
    padding: 0;
    position: relative;
    z-index: 10;
    transition: transform 0.5s ease-in-out; /* Smooth animation for movement */
  }

  .mic-button.shift-left {
    transform: translateX(-35%); /* Move the button to the left */
  }

  .loader {
    --size: 500px;
    --duration: 2.5s;
    --logo-color: grey;
    --background: linear-gradient(
      0deg,
      rgb(30 27 109 / 20%) 0%,
      rgb(137 76 161 / 20%) 100%
    );
    height: var(--size);
    aspect-ratio: 1;
    position: relative;
    z-index: 10;
  }

  .loader .box {
    position: absolute;
    background: var(--background);
    border-radius: 50%;
    box-shadow:
      rgba(0, 0, 0, 0.5) 0px 10px 10px 0,
      inset rgba(205, 155, 255, 0.5) 0px 5px 10px -7px;
    animation: ripple var(--duration) infinite ease-in-out;
    inset: var(--inset);
    animation-delay: calc(var(--i) * 0.15s);
    z-index: calc(var(--i) * -1);
    transition: all 0.3s ease;
  }

  .loader .box:last-child {
    filter: blur(30px);
  }

  .loader .box:not(:last-child):hover {
    filter: brightness(2.5) blur(5px);
  }

  .loader .logo {
    position: absolute;
    inset: 0;
    display: grid;
    place-content: center;
    padding: 30%;
    z-index: 20;
  }

  .loader .logo svg {
    fill: var(--logo-color);
    width: 100%;
    animation: color-change var(--duration) infinite ease-in-out;
  }

  .wave-container {
    position: absolute;
    right: 20%; /* Position on the right side */
    opacity: 0;
    animation: fadeIn 0.5s ease-in-out forwards; /* Fade in animation */
  }

  .center_div {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 5px;
  }

  .wave {
    width: 1.8rem;
    height: 120px;
    background-color: #ff6b6b;
    margin: 0 4px;
    border-radius: 0.4rem;
    animation: wave 1.5s linear infinite;
    transform-origin: center;
  }

  @keyframes wave {
    0% {
      transform: scale(0);
      filter: hue-rotate(90deg) blur(100px);
    }
    25% {
      transform: scale(0);
      filter: hue-rotate(120deg) blur(50px);
    }
    50% {
      transform: scale(1);
      filter: hue-rotate(180deg) blur(25px);
    }
    75% {
      transform: scale(0);
      filter: hue-rotate(360deg) blur(2px);
    }
    100% {
      transform: scale(0);
      filter: hue-rotate(0deg) blur(0);
    }
  }

  .wave:nth-child(2) {
    animation-delay: 0.1s;
  }

  .wave:nth-child(3) {
    animation-delay: 0.2s;
  }

  .wave:nth-child(4) {
    animation-delay: 0.3s;
  }

  .wave:nth-child(5) {
    animation-delay: 0.4s;
  }

  .wave:nth-child(6) {
    animation-delay: 0.5s;
  }

  .wave:nth-child(7) {
    animation-delay: 0.6s;
  }

  .wave:nth-child(8) {
    animation-delay: 0.7s;
  }

  .wave:nth-child(9) {
    animation-delay: 0.8s;
  }

  .wave:nth-child(10) {
    animation-delay: 0.9s;
  }

  @keyframes ripple {
    0% {
      transform: scale(1);
      box-shadow:
        rgba(0, 0, 0, 0.5) 0px 10px 10px 0,
        inset rgba(205, 155, 255, 0.5) 0px 5px 10px -7px;
    }
    65% {
      transform: scale(1.4);
      box-shadow: rgba(0, 0, 0, 0) 0px 0 0 0;
    }
    100% {
      transform: scale(1);
      box-shadow:
        rgba(0, 0, 0, 0.5) 0px 10px 10px 0,
        inset rgba(205, 155, 255, 0.5) 0px 5px 10px -7px;
    }
  }

  @keyframes color-change {
    0% {
      fill: var(--logo-color);
    }
    50% {
      fill: white;
    }
    100% {
      fill: var(--logo-color);
    }
  }

  @keyframes fadeIn {
    0% {
      opacity: 0;
      transform: translateX(20px);
    }
    100% {
      opacity: 1;
      transform: translateX(0);
    }
  }

  .processing-container {
    position: absolute;
    right: 20%;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
    opacity: 0;
    animation: fadeIn 0.5s ease-in-out forwards;
  }

  .processing-text {
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  }

  .spinner {
    width: 40px;
    height: 40px;
    border: 4px solid rgba(255, 255, 255, 0.3);
    border-top: 4px solid white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .error-container {
    position: absolute;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255, 0, 0, 0.9);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    z-index: 1000;
  }

  .error-text {
    font-size: 1rem;
    font-weight: 500;
  }

  .error-close {
    background: none;
    border: none;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0;
    width: 25px;
    height: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: background-color 0.2s;
  }

  .error-close:hover {
    background-color: rgba(255, 255, 255, 0.2);
  }

  .results-container {
    position: absolute;
    right: 5%;
    top: 10%;
    transform: translateY(-50%);
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    min-width: 300px;
    max-width: 400px;
    opacity: 0;
    animation: fadeIn 0.5s ease-in-out forwards;
    backdrop-filter: blur(10px);
  }

  .results-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #333;
    margin-bottom: 20px;
    text-align: center;
    background: linear-gradient(90deg, #f55702, #eeaeca, #94c9e9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }

  .emotion-display {
    text-align: center;
    margin-bottom: 20px;
  }

  .emotion-name {
    font-size: 2rem;
    font-weight: 800;
    color: #f55702;
    text-transform: capitalize;
    margin-bottom: 5px;
  }

  .emotion-duration {
    font-size: 0.9rem;
    color: #666;
  }

  .probabilities-container {
    margin-top: 20px;
  }

  .probabilities-title {
    font-size: 1rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 15px;
  }

  .probabilities-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .probability-item {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .emotion-label {
    font-size: 0.9rem;
    font-weight: 500;
    color: #333;
    min-width: 80px;
    text-transform: capitalize;
  }

  .probability-bar {
    flex: 1;
    height: 8px;
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }

  .probability-fill {
    height: 100%;
    background: linear-gradient(90deg, #f55702, #eeaeca);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .probability-value {
    font-size: 0.8rem;
    font-weight: 600;
    color: #666;
    min-width: 40px;
    text-align: right;
  }

  /* Prevent keyboard focus ring around mic button */
  .mic-button:focus { outline: none; box-shadow: none; }
  .mic-button:focus-visible { outline: none; box-shadow: none; }
`;

export default App;