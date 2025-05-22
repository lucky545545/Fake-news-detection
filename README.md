# link for the Globle vectors https://www.kaggle.com/datasets/bertcarremans/glovetwitter27b100dtxt

# 📰 Fake News Detection using LSTM and GloVe

This project aims to classify news articles as **real** or **fake** using deep learning techniques. A Long Short-Term Memory (LSTM) model was trained using pre-trained **GloVe word embeddings** to capture contextual meanings of words. The model achieves a classification accuracy of approximately **94–96%**.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Preprocessing Steps](#preprocessing-steps)
- [Training and Evaluation](#training-and-evaluation)
- [Usage](#usage)
- [Results](#results)
- [How to Run](#how-to-run)
- [License](#license)

---

## 📖 Introduction

Fake news has become a major concern in the modern digital world. This project is designed to automatically detect whether a piece of news is genuine or fabricated using natural language processing (NLP) and deep learning.

---

## 🗂 Project Structure

├── app.py 
├──Templates
    └── index.html
├── fake_new_det.keras
├── tokenizer.pkl
├── fake.csv
├── true.csv
├── fake_news.ipynb # Model training notebook
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 🛠 Technologies Used

- **Python 3**
- **TensorFlow / Keras**
- **NLTK**
- **Flask** (for API)
- **GloVe (Global Vectors for Word Representation)**

---

## 🧠 Model Architecture

The model consists of:

- **Embedding Layer** using GloVe vectors (100-dimensional)
- **2 LSTM Layers** with dropout
- **Dense Layer** with ReLU activation
- **Output Layer** with sigmoid activation for binary classification

