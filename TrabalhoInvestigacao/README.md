# Sign Language Recognition and Translation using Computer Vision and Machine Learning

Academic research project developed for the **Sensorization and Environment** course, as part of the **Master's Degree in Artificial Intelligence / Computer Engineering** at the **University of Minho**.

**Final Grade:** 19/20

## Overview

This project explores the state of the art in automatic sign language recognition and translation using **Computer Vision** and **Machine Learning** techniques.

The main motivation is to reduce communication barriers between the deaf community and hearing people by designing a system capable of translating sign language gestures into **text and speech**.

Although the initial motivation is related to **Portuguese Sign Language (LGP)**, the practical implementation is based on **American Sign Language (ASL)** due to the greater availability of public annotated datasets.

## Main Goals

The research focuses on:

* Studying existing approaches for sign language recognition;
* Understanding the role of visual sensors in human-computer interaction;
* Exploring the use of webcam-based sensing instead of intrusive hardware;
* Analysing static and dynamic gesture recognition;
* Discussing ethical, privacy and accessibility implications;
* Proposing a possible implementation pipeline for a real-time recognition system.

## Proposed System

The proposed system follows a pipeline based on visual sensing and machine learning:

1. **Video Capture**
   A standard RGB webcam captures the user's hand gestures in real time.

2. **Landmark Extraction**
   MediaPipe is used to extract hand, face and body landmarks.

3. **Gesture Classification**
   Machine Learning models classify the detected gestures.

4. **Text Generation**
   Recognised signs are converted into readable text.

5. **Speech Output**
   The generated text can be converted into speech, improving accessibility.

## Technologies and Methods

The project discusses and proposes the use of:

* **Python**
* **OpenCV**
* **MediaPipe Hands**
* **MediaPipe Holistic**
* **Machine Learning**
* **Multi-Layer Perceptron (MLP)**
* **Convolutional Neural Networks (CNN)**
* **Long Short-Term Memory Networks (LSTM)**
* **Text-to-Speech**
* **Large Language Models for post-processing**

## Static Gesture Recognition

For static gestures, such as the ASL alphabet, the proposed approach uses:

* ASL Alphabet Dataset;
* MediaPipe Hands;
* 21 hand landmarks;
* 3D coordinates per landmark;
* A classification model such as an MLP.

This approach avoids using raw image pixels directly and instead relies on geometric hand features, making the system less dependent on lighting conditions, background and skin tone.

## Dynamic Gesture Recognition

For dynamic gestures, such as full words or short expressions, the project proposes:

* WLASL dataset;
* MediaPipe Holistic;
* Temporal sequences of landmarks;
* LSTM-based models.

Dynamic recognition is more complex because it requires analysing movement over time and identifying when a gesture starts and ends.

## Why ASL instead of LGP?

Portuguese Sign Language was the original motivation of the project. However, there are currently not enough large-scale public annotated datasets for LGP to train robust Machine Learning models.

For that reason, ASL was selected as the implementation target, since it provides widely used public datasets such as:

* ASL Alphabet Dataset;
* WLASL;
* ASL-LEX.

The methodology remains relevant for LGP and could be adapted in the future if larger and more diverse Portuguese Sign Language datasets become available.

## Ethical and Social Considerations

This project also reflects on the social impact of automatic sign language recognition systems.

Important concerns include:

* The need to involve the deaf community in the design and validation process;
* The risk of replacing, rather than supporting, human interpreters;
* Bias in training datasets;
* Privacy risks associated with video capture;
* The importance of on-device processing;
* Compliance with data protection principles.

The proposed system should be understood as an accessibility support tool, not as a replacement for professional interpreters.

## Potential Use Cases

Possible application scenarios include:

* Educational tools for learning sign language;
* Accessibility support in public services;
* Hospitals and emergency services;
* Museums and public information kiosks;
* Smart city accessibility systems;
* Assistive communication tools.

## Repository Structure

```txt
.
├── docs/
│   └── research-paper.pdf
├── src/
│   └── prototype/
├── models/
├── datasets/
├── README.md
└── requirements.txt
```

> Note: The structure above represents a possible organisation for future implementation work.

## Future Work

Future improvements may include:

* Collecting and annotating a Portuguese Sign Language dataset;
* Improving real-time gesture segmentation;
* Supporting continuous sign language recognition;
* Exploring Transformer-based architectures;
* Developing bidirectional translation;
* Creating a full web-based prototype;
* Testing the system with real users from the deaf community.

## Academic Context

This work was developed as a research phase for the **Sensorization and Environment** course.

The project combines topics from:

* Human-Computer Interaction;
* Computer Vision;
* Machine Learning;
* Mobile and environmental sensing;
* Accessibility;
* Ethics in Artificial Intelligence.

## Final Grade

**19/20**

This grade reflects the quality of the research, state-of-the-art analysis, technical proposal and critical discussion of ethical, social and privacy-related issues.

## License

This repository is intended for academic and educational purposes.
