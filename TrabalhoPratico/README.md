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
* phrase correction;
* text-to-speech output;
* a web-based user interface;
* a FastAPI inference backend.

## Main Objective

The main objective of this prototype is to demonstrate how **Computer Vision**, **Machine Learning** and real-time sensing can be combined to create an accessible sign language translation system.

The system allows the user to perform gestures in front of the webcam and receive predictions in real time. The recognized tokens can then be accumulated into a phrase, corrected and read aloud.

## System Architecture

The practical prototype is divided into the following main components:

```txt
TrabalhoPratico/
├── README.md
├── .gitignore
├── backend/
├── frontend/
├── notebooks/
├── shared_models/
└── docs/
```

### Frontend

The frontend is a **Next.js** web application responsible for:

* accessing the webcam;
* capturing frames from the video stream;
* sending frames to the backend;
* displaying the current prediction;
* showing the confidence score;
* switching between static and dynamic recognition modes;
* composing the final phrase;
* triggering phrase correction;
* triggering text-to-speech output.

### Backend

The backend is a **FastAPI** service responsible for:

* receiving images or frame sequences;
* running MediaPipe landmark extraction;
* loading trained model artifacts;
* performing static gesture inference;
* performing dynamic gesture inference;
* applying phrase correction;
* providing a speech output endpoint.

### Notebooks

The `notebooks/` folder contains the experimental and training work developed during the project.

It includes:

* static landmark extraction;
* static ASL alphabet model training;
* real-time static inference tests;
* dynamic video landmark extraction;
* LSTM experiments;
* Transformer experiments;
* model evaluation;
* training curves and confusion matrices.

### Shared Models

The `shared_models/` folder contains the trained artifacts used by the backend.

It includes:

* static MLP model;
* static scaler;
* static label encoder;
* dynamic LSTM model;
* dynamic Transformer models;
* MediaPipe hand landmarker task;
* MediaPipe pose landmarker task;
* dynamic label encoders.

### Docs

The `docs/` folder contains the official practical assignment and the final practical work presentation.

## Recognition Modes

The prototype supports two recognition modes.

## 1. Static Gesture Recognition

Static recognition is used for ASL alphabet gestures.

The system receives a single webcam frame, extracts hand landmarks and classifies the gesture as an ASL letter or control token.

The static recognition pipeline is:

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

The static model recognizes ASL alphabet letters and control tokens such as:

* `space`;
* `del`;
* `nothing`.

During notebook evaluation, the static MLP model achieved approximately:

* **97% validation accuracy**;
* **97% test accuracy**.

These results were obtained under dataset conditions and may vary in real-time webcam usage depending on lighting, camera quality, distance and hand positioning.

## 2. Dynamic Gesture Recognition

Dynamic recognition is used for ASL words involving movement.

Unlike static recognition, this mode analyses a sequence of frames instead of a single image.

The dynamic recognition pipeline is:

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

The final backend prioritizes the Transformer-based dynamic model when available and falls back to the LSTM model when necessary.

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
* automatic static token confirmation;
* manual token addition;
* phrase composition;
* delete and clear controls;
* phrase correction;
* text-to-speech output;
* dark/light theme toggle;
* FastAPI backend with documented endpoints.

## Phrase Correction

The system includes a phrase correction module.

By default, the backend applies local correction rules for common recognized sequences.

Optionally, the system can be configured to use a local LLM through **Ollama**, allowing ASL-like word sequences to be converted into more natural English sentences.

Example:

```txt
I WANT WATER
  ↓
I want some water, please.
```

## Text-to-Speech

The frontend uses the browser's built-in speech synthesis when available.

The backend also provides a `/api/speak` endpoint, which can generate speech using Edge TTS as a fallback.

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
| `/api/llm_correct`     | Applies phrase correction                            |
| `/api/speak`           | Converts text into speech                            |

## Local Setup

## 1. Backend

From the `TrabalhoPratico/backend` folder:

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

## 2. Frontend

From the `TrabalhoPratico/frontend` folder:

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

Example configuration:

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
ASLAI_HAND_LANDMARKER_TASK_PATH=../shared_models/hand_landmarker.task
ASLAI_POSE_LANDMARKER_TASK_PATH=../shared_models/pose_landmarker_lite.task
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

The system continuously captures webcam frames and predicts the current gesture.

When the same gesture remains stable for several frames and passes the confidence threshold, the frontend can automatically add the recognized token to the phrase.

The user can also manually add the current prediction.

## Dynamic Mode Usage

Dynamic mode is designed for recognizing complete ASL words.

The user starts the capture, performs the gesture, stops the capture and the backend classifies the sequence of frames.

This mode is more complex because it depends on:

* movement quality;
* timing;
* body visibility;
* camera position;
* lighting conditions;
* consistency between training and real-time usage.

## Model Artifacts

The backend expects the trained models to be available in:

```txt
TrabalhoPratico/shared_models/
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
pose_landmarker_lite.task
```

## Folder Structure

```txt
TrabalhoPratico/
├── README.md
├── .gitignore
├── backend/
│   ├── README.md
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── main.py
│       ├── api/
│       ├── core/
│       ├── services/
│       └── utils/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── package.json
│   └── tailwind.config.ts
│
├── notebooks/
│   ├── static/
│   └── dynamic/
│
├── shared_models/
│   ├── mlp_asl_landmarks.joblib
│   ├── scaler_asl_landmarks.joblib
│   ├── label_encoder_asl_landmarks.joblib
│   ├── asl_lstm_final.pt
│   ├── asl_transformer_v2_100_2.pt
│   ├── hand_landmarker.task
│   └── pose_landmarker_full.task
│
└── docs/
    ├── ApresentacaoTP.pdf
    └── Trabalho pratico 2526.pdf
```

## Limitations

The current version has some limitations:

* the system is based on ASL, not Portuguese Sign Language;
* real-time predictions depend on lighting and camera quality;
* dynamic gesture recognition is sensitive to timing and movement variation;
* automatic segmentation of continuous sign language is not fully solved;
* the vocabulary is limited to the trained classes;
* phrase correction is simple and may require stronger language models;
* the system is a prototype and not a certified translation tool.

## Future Improvements

Possible improvements include:

* adding support for Portuguese Sign Language;
* collecting an LGP dataset;
* improving dynamic gesture segmentation;
* increasing the dynamic vocabulary size;
* improving robustness in uncontrolled environments;
* adding user calibration;
* improving phrase correction with stronger local language models;
* improving the real-time dynamic recognition workflow;
* adding bidirectional translation from text/speech to sign language;
* testing the system with real users;
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
