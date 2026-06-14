# ASLAI — Practical Prototype

This folder contains the practical implementation of **ASLAI**, a sign language recognition and translation prototype developed for the **Sensorization and Environment** course at the **University of Minho**.

**Final Grade:** 18/20

## Overview

The practical work implements a real-time system capable of recognizing **American Sign Language (ASL)** gestures through a standard webcam and converting them into text and speech.

The prototype combines:

* real-time webcam capture;
* MediaPipe landmark extraction;
* static gesture recognition;
* dynamic gesture recognition;
* phrase composition;
* basic phrase correction;
* text-to-speech output;
* a web-based user interface;
* a FastAPI inference backend.

## Main Objective

The goal of the practical prototype is to demonstrate how Computer Vision and Machine Learning can be combined to create an accessible sign language translation system.

The system allows the user to perform gestures in front of the webcam and receive a prediction in real time. The recognized tokens can then be accumulated into a phrase, corrected and read aloud.

## System Architecture

The prototype is divided into four main parts:

```txt
TP/
├── frontend/
├── backend/
├── notebooks/
└── shared_models/
```

### Frontend

The frontend is a **Next.js** web application responsible for:

* accessing the webcam;
* capturing frames;
* sending frames to the backend;
* displaying the current prediction;
* showing the confidence score;
* allowing the user to switch between static and dynamic modes;
* composing the final phrase;
* triggering phrase correction;
* triggering text-to-speech.

### Backend

The backend is a **FastAPI** service responsible for:

* receiving images or frame sequences;
* running MediaPipe landmark extraction;
* loading trained model artifacts;
* performing static gesture inference;
* performing dynamic gesture inference;
* correcting recognized phrases;
* generating speech output.

### Notebooks

The notebooks contain the experimental and training work, including:

* landmark extraction;
* static gesture model training;
* dynamic gesture model training;
* LSTM experiments;
* Transformer experiments;
* model evaluation;
* real-time inference tests.

### Shared Models

The `shared_models/` folder contains the trained artifacts used by the backend, including:

* static MLP model;
* static scaler;
* static label encoder;
* dynamic LSTM model;
* dynamic Transformer models;
* MediaPipe hand landmarker task;
* MediaPipe pose landmarker task;
* dynamic label encoders.

## Recognition Modes

The prototype supports two recognition modes.

### 1. Static Gesture Recognition

Static recognition is used for ASL alphabet gestures.

The pipeline is:

```txt
Single webcam frame
  ↓
MediaPipe Hands
  ↓
21 hand landmarks
  ↓
3D landmark vector
  ↓
MLP classifier
  ↓
Predicted ASL letter
```

The static model recognizes ASL alphabet letters and control tokens such as `space` and `del`.

During notebook evaluation, the static MLP model achieved approximately:

* **97% validation accuracy**;
* **97% test accuracy**.

These results were obtained under dataset conditions and may vary in real-time webcam usage.

### 2. Dynamic Gesture Recognition

Dynamic recognition is used for ASL words involving movement.

The pipeline is:

```txt
Sequence of webcam frames
  ↓
MediaPipe landmark extraction
  ↓
Temporal sequence of features
  ↓
LSTM or Transformer model
  ↓
Predicted ASL word
```

The final backend prioritizes the Transformer-based dynamic model when available.

The dynamic model works with a vocabulary of ASL words and uses temporal information across multiple frames.

During experimentation, the dynamic models achieved results such as:

* **LSTM baseline:** around 65% validation accuracy;
* **Transformer model with 50 classes:** around 78% Top-1 accuracy and 90% Top-5 accuracy;
* **Transformer model with 100 classes:** around 76% Top-1 accuracy and 89% Top-5 accuracy.

## Features

The prototype includes:

* real-time webcam interface;
* static ASL alphabet recognition;
* dynamic ASL word recognition;
* confidence score display;
* phrase composition;
* manual token addition;
* delete and clear controls;
* phrase correction;
* text-to-speech output;
* dark interface theme;
* FastAPI backend with documented endpoints.

## Technologies Used

### Backend

* Python 3.11;
* FastAPI;
* Uvicorn;
* OpenCV;
* MediaPipe;
* NumPy;
* Scikit-learn;
* Joblib;
* PyTorch;
* Edge TTS;
* Pydantic Settings.

### Frontend

* Next.js;
* React;
* TypeScript;
* Tailwind CSS.

### Machine Learning

* MLP for static gesture recognition;
* LSTM for dynamic sequence recognition;
* Transformer Encoder for dynamic sequence recognition;
* MediaPipe landmarks as input features.

## Backend API

The backend exposes the following endpoints:

```txt
GET  /api/health
POST /api/predict
POST /api/predict_dynamic
POST /api/llm_correct
POST /api/speak
```

### Endpoint Description

| Endpoint               | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `/api/health`          | Checks backend status and loaded models              |
| `/api/predict`         | Predicts a static gesture from one frame             |
| `/api/predict_dynamic` | Predicts a dynamic gesture from a sequence of frames |
| `/api/llm_correct`     | Applies basic phrase correction                      |
| `/api/speak`           | Converts the phrase into speech                      |

## Local Setup

### 1. Backend

From the `TP/backend` folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

The backend will run at:

```txt
http://localhost:8000
```

The API documentation is available at:

```txt
http://localhost:8000/docs
```

### 2. Frontend

From the `TP/frontend` folder:

```bash
npm install
npm run dev
```

The frontend will run at:

```txt
http://localhost:3000
```

By default, the frontend communicates with:

```txt
http://localhost:8000/api
```

The API URL can be configured through:

```txt
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Environment Variables

The backend can be configured using the `.env` file.

Example values:

```env
ASLAI_APP_NAME=ASLAI API
ASLAI_APP_VERSION=0.1.0
ASLAI_API_PREFIX=/api
ASLAI_ENABLE_TTS=true
ASLAI_ENABLE_LLM_CORRECTION=false
ASLAI_MAX_DYNAMIC_FRAMES=60
ASLAI_SEQUENCE_LENGTH=30
ASLAI_MINIMUM_DYNAMIC_FRAMES=5
ASLAI_STATIC_CONFIDENCE_THRESHOLD=0.75
ASLAI_MODEL_ARTIFACTS_DIR=../shared_models
```

## How to Use

1. Start the backend.
2. Start the frontend.
3. Open the web application in the browser.
4. Allow webcam access.
5. Choose between static or dynamic mode.
6. Perform a gesture in front of the camera.
7. Check the predicted token and confidence score.
8. Add the prediction to the phrase.
9. Correct the phrase if needed.
10. Use text-to-speech to read the result aloud.

## Static Mode Usage

Static mode is designed for spelling words using ASL alphabet gestures.

The system continuously reads webcam frames and predicts the current gesture. The user can add the recognized letter to the phrase or use control gestures such as `space` and `del`.

## Dynamic Mode Usage

Dynamic mode is designed for recognizing complete ASL words.

The user starts the capture, performs the gesture, stops the capture and the backend classifies the sequence of frames.

This mode is more complex because it depends on movement, timing, camera position and body visibility.

## Model Artifacts

The backend expects the trained models to be available in:

```txt
TP/shared_models/
```

Important artifacts include:

```txt
mlp_asl_landmarks.joblib
scaler_asl_landmarks.joblib
label_encoder_asl_landmarks.joblib
asl_lstm_final.pt
asl_transformer_v2_100_2.pt
label_encoder_wlasl100.joblib
hand_landmarker.task
pose_landmarker_full.task
```

## Limitations

The current version has some limitations:

* the system is based on ASL, not Portuguese Sign Language;
* real-time predictions depend on lighting and camera quality;
* dynamic gesture recognition is sensitive to timing and movement variation;
* automatic segmentation of continuous sign language is not fully solved;
* the vocabulary is limited to the trained classes;
* phrase correction is basic and optional;
* the system is a prototype and not a certified translation tool.

## Future Improvements

Possible improvements include:

* adding support for Portuguese Sign Language;
* collecting an LGP dataset;
* improving dynamic gesture segmentation;
* increasing the vocabulary size;
* improving robustness in uncontrolled environments;
* adding user calibration;
* improving phrase correction with stronger language models;
* adding bidirectional translation;
* testing with real users;
* deploying the prototype online.

## Academic Context

Developed at:

**University of Minho**
**Master's Degree in Artificial Intelligence**
**Sensorization and Environment**
**Academic Year 2025/2026**

## Final Grade

**18/20**

## License

This project is intended for academic and educational purposes.
