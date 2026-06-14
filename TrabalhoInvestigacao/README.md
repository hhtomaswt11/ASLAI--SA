# Research Work — Sign Language Recognition and Translation using Computer Vision and Machine Learning

This folder contains the research component of the **ASLAI** project, developed for the **Sensorization and Environment** course at the **University of Minho**.

**Final Grade:** 19/20

## Overview

This research work presents a state-of-the-art analysis of automatic sign language recognition and translation systems using **Computer Vision** and **Machine Learning**.

The main goal of the work is to study how a conventional RGB camera can be used as a visual sensor to capture sign language gestures, extract relevant features and support the development of systems capable of translating gestures into **text and speech**.

The project is motivated by the need to reduce communication barriers between the deaf community and hearing people, especially in everyday contexts such as education, healthcare, transport, public services and general assistance.

## Academic Context

This work was developed as the first phase of the practical component of the **Sensorization and Environment** course.

The assignment required the development of a research paper or survey focused on the state of the art of the topic that would later support the practical implementation.

The research was written in the format of a scientific paper and was also presented orally as part of the evaluation process.

## Main Topic

The selected topic was:

**Sign Language Recognition and Translation to Text and Speech using Computer Vision and Machine Learning**

The work is framed within the area of **Human-Computer Interaction**, combining:

* visual sensing through a webcam;
* real-time landmark extraction;
* static gesture recognition;
* dynamic gesture recognition;
* Machine Learning classification models;
* text generation;
* speech synthesis;
* accessibility and inclusion.

## Motivation

Portuguese Sign Language is the natural language of the Portuguese deaf community. Although it is legally recognized, it is still unknown by a large part of the hearing population.

This creates communication barriers in several contexts:

* healthcare;
* education;
* public services;
* transport;
* emergency situations;
* general daily communication.

The research explores how Artificial Intelligence can support accessibility by enabling a technological system capable of interpreting sign language gestures through non-invasive and low-cost sensing.

## Why ASL instead of LGP?

The original motivation of the project is related to **Portuguese Sign Language (LGP)**.

However, one of the main limitations identified in the research is the lack of large-scale public annotated datasets for LGP. This makes it difficult to train robust Machine Learning models directly for Portuguese Sign Language.

For this reason, the proposed practical implementation is based on **American Sign Language (ASL)**, which has more mature and publicly available datasets, such as:

* ASL Alphabet Dataset;
* WLASL;
* ASL-LEX.

The work argues that the methodology developed for ASL can later be adapted to LGP if larger and more diverse datasets become available.

## State of the Art

The research reviews different approaches for capturing and recognizing sign language gestures.

Two main sensing approaches are discussed:

### Specialized Hardware

Some systems use gloves with flex sensors, accelerometers or other dedicated devices.

These approaches can provide accurate data, but they are often:

* expensive;
* intrusive;
* less natural for the user;
* dependent on specific hardware.

### Computer Vision

Computer Vision approaches use standard RGB cameras to capture gestures.

This approach is more accessible because it:

* does not require specialized hardware;
* is non-invasive;
* can run on common devices;
* is suitable for web-based or mobile applications.

## MediaPipe as a Sensing Framework

The research highlights **MediaPipe** as a suitable framework for sign language recognition.

### MediaPipe Hands

MediaPipe Hands can detect and track hands in real time using a standard camera.

For each detected hand, it extracts:

* 21 hand landmarks;
* 3D coordinates for each landmark;
* a compact vector representation of the hand pose.

This representation reduces the dimensionality of the input and avoids relying directly on raw image pixels.

### MediaPipe Holistic

For dynamic gestures involving hands, body and facial expressions, the research discusses MediaPipe Holistic.

MediaPipe Holistic combines:

* hand landmarks;
* body pose landmarks;
* facial landmarks.

This makes it more suitable for recognizing complete signs, words or phrases.

## Static Gesture Recognition

Static gesture recognition focuses on gestures that can be classified from a single frame.

In this project, static recognition is associated with the ASL alphabet.

The proposed static recognition pipeline is:

```txt
RGB Image
  ↓
MediaPipe Hands
  ↓
21 Hand Landmarks
  ↓
3D Landmark Vector
  ↓
MLP Classifier
  ↓
Predicted ASL Letter
```

The research proposes using a **Multi-Layer Perceptron (MLP)** for this task, since the input is a compact numerical vector extracted from the hand landmarks.

## Dynamic Gesture Recognition

Dynamic gesture recognition focuses on signs that involve movement over time.

Unlike static gestures, dynamic gestures require the analysis of frame sequences.

The proposed dynamic recognition pipeline is:

```txt
Video Sequence
  ↓
MediaPipe Holistic
  ↓
Temporal Landmark Sequence
  ↓
LSTM Model
  ↓
Predicted ASL Word
```

The research proposes using **Long Short-Term Memory networks (LSTM)** because they are suitable for modelling temporal dependencies in sequences.

The proposed dataset for this stage is **WLASL-100**, a subset of the Word-Level American Sign Language dataset containing 100 frequent ASL signs.

## Proposed System Architecture

The research proposes a complete system with the following stages:

```txt
Video Capture
  ↓
Landmark Extraction
  ↓
Gesture Classification
  ↓
Text Generation
  ↓
Speech Output
```

### 1. Video Capture

A conventional RGB webcam captures the user's gestures in real time.

### 2. Landmark Extraction

MediaPipe extracts hand, body and face landmarks from the captured frames.

### 3. Gesture Classification

Machine Learning models classify the detected gesture.

The system is divided into two recognition modules:

* static module for ASL alphabet gestures;
* dynamic module for ASL words.

### 4. Text Generation

The recognized gestures are converted into text.

For dynamic signs, the research also proposes using a language model to transform a sequence of recognized words into a grammatically coherent sentence.

### 5. Speech Output

The final sentence can be synthesized into speech, making the system more useful in communication scenarios between deaf and hearing people.

## Technical Decisions

| Component | Static Gestures         | Dynamic Gestures   |
| --------- | ----------------------- | ------------------ |
| Dataset   | ASL Alphabet            | WLASL-100          |
| Sensing   | MediaPipe Hands         | MediaPipe Holistic |
| Model     | MLP                     | LSTM               |
| Input     | Single-frame landmarks  | Landmark sequences |
| Output    | Letter or control token | Word or phrase     |

## Application Scenarios

The research discusses several possible application contexts.

### Education and Gamification

The system could be used as an educational tool for people learning sign language.

Possible features include:

* immediate feedback;
* scores;
* difficulty levels;
* progress tracking;
* classroom dashboards.

### Healthcare and Emergency Services

The system could support communication in hospitals, emergency rooms or triage situations where an interpreter is not immediately available.

However, this scenario requires high reliability, clear confidence indicators and human supervision.

### Smart Cities and Public Spaces

The system could be integrated into public information points, museums, transport stations, municipal services or other public spaces.

This would promote accessibility and support more inclusive smart city services.

## Ethical and Social Considerations

The research emphasizes that automatic sign language translation systems must be developed responsibly.

Important concerns include:

* involvement of the deaf community in design and validation;
* avoiding the replacement of professional interpreters;
* dataset bias;
* regional variation in sign language;
* privacy risks associated with video capture;
* responsible use of biometric data;
* transparency about system limitations.

The system should be understood as an accessibility support tool, not as a replacement for human interpreters.

## Privacy and Data Protection

Since the system uses a camera, privacy is a central concern.

The research proposes a privacy-by-design approach based on:

* local processing whenever possible;
* no unnecessary storage of video data;
* processing landmarks instead of raw video;
* clear user information about collected data;
* avoiding the transmission of visual data to external servers.

This is especially relevant in public, educational and healthcare contexts.

## Limitations

The main limitations identified in the research are:

* lack of large-scale public datasets for Portuguese Sign Language;
* difficulty in recognizing dynamic gestures;
* dependence on lighting and camera conditions;
* variability between different signers;
* difficulty in automatic gesture segmentation;
* possible bias in training datasets;
* need for validation with real users from the deaf community.

## Future Work

Future research and development may include:

* collecting and annotating a Portuguese Sign Language dataset;
* adapting the ASL-based methodology to LGP;
* improving continuous sign language recognition;
* adding automatic gesture segmentation;
* exploring Transformer-based models;
* improving phrase generation with language models;
* supporting bidirectional translation;
* testing the system with real users;
* evaluating the system in real-world environments.

## Folder Structure

```txt
TrabalhoInvestigacao/
├── README.md
├── TrabalhoInvestigacaoG12.pdf
├── Trabalho investigacao 2526.pdf
└── ApresentacaoTI.pdf
```

## Files

| File                             | Description                   |
| -------------------------------- | ----------------------------- |
| `TrabalhoInvestigacaoG12.pdf`    | Final research paper          |
| `Trabalho investigacao 2526.pdf` | Official assignment statement |
| `ApresentacaoTI.pdf`             | Research presentation slides  |
| `README.md`                      | Documentation for this folder |

## Final Grade

**19/20**

This grade reflects the quality of the research paper, the state-of-the-art analysis, the proposed implementation architecture and the discussion of ethical, privacy and accessibility issues.

## License

This work is intended for academic and educational purposes.
