# ASLAI — Sign Language Recognition and Translation with AI

ASLAI is an academic project developed for the **Sensorization and Environment** course, as part of the **Master's Degree in Artificial Intelligence** at the **University of Minho**.

The project explores the use of **Computer Vision**, **Machine Learning** and real-time sensing techniques to recognize sign language gestures and convert them into **text and speech**.

The repository contains both stages of the work:

* **TrabalhoInvestigacao — Research Work**: state-of-the-art analysis and scientific research paper;
* **TrabalhoPratico — Practical Work**: functional prototype for real-time sign language recognition.

## Final Grades

| Component            | Description         | Grade |
| -------------------- | ------------------- | ----: |
| TrabalhoInvestigacao | Research Work       | 19/20 |
| TrabalhoPratico      | Practical Prototype | 18/20 |

## Project Motivation

Sign language is a fundamental communication method for many deaf people. However, communication barriers still exist in everyday contexts such as education, healthcare, transport, public services and general customer support.

This project aims to explore how artificial intelligence can help reduce those barriers by creating an accessible, non-invasive and low-cost system capable of recognizing sign language gestures through a standard webcam.

## General Objective

The main objective of ASLAI is to design and implement a system capable of:

* capturing gestures through a camera;
* extracting visual features from the user's body and hands;
* recognizing static and dynamic gestures;
* converting recognized gestures into text;
* producing speech output from the generated phrase.

## Research and Practical Development

The project started with a research phase focused on the state of the art in sign language recognition and translation systems.

The research work studied:

* Portuguese Sign Language and its challenges;
* American Sign Language datasets;
* Computer Vision approaches;
* MediaPipe-based landmark extraction;
* static gesture recognition;
* dynamic gesture recognition;
* privacy, ethics and accessibility concerns;
* future trends in AI-based sign language translation.

The practical phase then implemented a working prototype based on the concepts studied during the research phase.

## Why ASL?

The original motivation of the project is related to **Portuguese Sign Language (LGP)**. However, large-scale public annotated datasets for LGP are still limited.

For this reason, the implementation focuses on **American Sign Language (ASL)**, which has more mature and publicly available datasets, such as:

* ASL Alphabet Dataset;
* WLASL;
* ASL-LEX.

The methodology can later be adapted to LGP if larger and more diverse datasets become available.

## Main Technologies

The project uses:

* Python;
* FastAPI;
* Next.js;
* React;
* TypeScript;
* Tailwind CSS;
* OpenCV;
* MediaPipe;
* Scikit-learn;
* PyTorch;
* Machine Learning models;
* Text-to-Speech.

## Repository Structure

```txt
.
├── TrabalhoInvestigacao/
│   ├── README.md
│   └── research-paper.pdf
│
├── TrabalhoPratico/
│   ├── README.md
│   ├── backend/
│   ├── frontend/
│   ├── notebooks/
│   └── shared_models/
│
└── README.md
```

## Project Components

### TrabalhoInvestigacao — Research Work

The research work presents a scientific analysis of sign language recognition and translation systems using Computer Vision and Machine Learning.

It discusses the technical feasibility of using webcam-based sensing, MediaPipe landmarks and machine learning models to recognize gestures and translate them into text and speech.

### TrabalhoPratico — Practical Work

The practical work implements a functional prototype with:

* webcam-based real-time recognition;
* static gesture recognition for ASL alphabet letters;
* dynamic gesture recognition for ASL words;
* phrase composition;
* basic phrase correction;
* text-to-speech output;
* web interface;
* FastAPI backend for inference.

## System Overview

The general pipeline follows these steps:

```txt
Webcam
  ↓
Frame Capture
  ↓
MediaPipe Landmark Extraction
  ↓
Machine Learning Model
  ↓
Gesture Prediction
  ↓
Text Composition
  ↓
Phrase Correction
  ↓
Speech Output
```

## Recognition Modes

The system supports two recognition modes:

### Static Recognition

Static recognition focuses on ASL alphabet gestures. The system receives a single frame, extracts hand landmarks and classifies the gesture as a letter or control token.

### Dynamic Recognition

Dynamic recognition focuses on words or signs that involve movement. The system receives a sequence of frames, extracts temporal landmark information and classifies the full gesture sequence.

## Ethical and Social Considerations

This project treats sign language recognition as an accessibility support tool, not as a replacement for human interpreters.

Important concerns include:

* involvement of the deaf community in future validation;
* privacy and data protection;
* local processing whenever possible;
* dataset bias;
* accessibility in real-world environments;
* responsible use of AI.

## Limitations

The current prototype has some limitations:

* it is based on ASL rather than LGP;
* dynamic recognition is more sensitive to lighting, camera angle and movement quality;
* automatic gesture segmentation is still limited;
* real-world use would require broader testing with diverse users;
* the system should not be used in critical contexts without human supervision.

## Future Work

Possible future improvements include:

* collecting and annotating a Portuguese Sign Language dataset;
* improving continuous sign language recognition;
* adding automatic gesture segmentation;
* improving robustness in uncontrolled environments;
* expanding the dynamic gesture vocabulary;
* integrating more advanced Transformer-based models;
* testing with real users from the deaf community;
* supporting bidirectional translation from text/speech to sign language.

## Academic Context

Developed at:

**University of Minho**
**Master's Degree in Artificial Intelligence**
**Sensorization and Environment**
**Academic Year 2025/2026**

## License

This repository is intended for academic and educational purposes.
