# 📱 SMS Spam Detection System (Deep Learning & RNN)

## 📄 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset & Data Preprocessing](#2-dataset--data-preprocessing)
3. [NLP Pipeline & Feature Engineering](#3-nlp-pipeline--feature-engineering)
4. [Model Architecture & Training](#4-model-architecture--training)
5. [Evaluation & Performance](#5-evaluation--performance)
6. [Tech Stack & Requirements](#6-tech-stack--requirements)
7. [Project Directory Structure](#7-project-directory-structure)
8. [Flask Web Application Deployment](#8-flask-web-application-deployment)

---

## 1. Project Overview
* This project implements an end-to-end Deep Learning-based Natural Language Processing (NLP) web application[cite: 1, 3, 5].
* Its primary objective is to classify incoming SMS messages automatically into **Spam** or **Ham (Normal)** categories[cite: 1, 3, 5].
* The system utilizes a Recurrent Neural Network (RNN) model integrated with a user-friendly Flask web interface[cite: 3, 5].

---

## 2. Dataset & Data Preprocessing
* The project uses the `spam_sms.csv` dataset, which initially contains 5,572 rows across two columns (`v1` and `v2`)[cite: 5].
* Data cleaning steps include dropping missing values and duplicate rows, resulting in a clean dataset of 5,169 rows[cite: 5].
* Columns are renamed to `label` and `text`[cite: 5]:
    * `label`: Contains category tags ('spam' or 'ham')[cite: 5].
    * `text`: Contains the raw SMS message content[cite: 5].
* Label encoding maps 'spam' to `1` and 'ham' to `0`[cite: 5].

---

## 3. NLP Pipeline & Feature Engineering
* **Text Cleaning (`clean_text`)**:
    * Converts all text strings to lowercase[cite: 5].
    * Removes special characters and digits using regular expressions (`re.sub(r'[^a-zA-Z]', ' ', text)`)[cite: 5].
* **Tokenization & Stemming**:
    * Sentences are tokenized into words using NLTK (`word_tokenize`)[cite: 5].
    * English stopwords are removed and words with a length greater than 2 are reduced to their root form using `PorterStemmer`[cite: 5].
* **Vectorization (Text to Sequences)**:
    * Keras `Tokenizer` is initialized with a vocabulary size of `num_words=5000` and an out-of-vocabulary token `oov_token="<OOV>"`[cite: 5].
    * Cleaned texts are converted into numerical integer sequences[cite: 5].
* **Padding**:
    * Sequences are standardized to a fixed length using `pad_sequences` (`maxlen=50`, `padding='post'`, `truncating='post'`), producing a final feature shape of `(5169, 50)`[cite: 5].

---

## 4. Model Architecture & Training
* **Train-Test Split**:
    * The dataset is split into training and testing sets using Scikit-Learn (`train_test_split` with `test_size=0.2`, `random_state=42`)[cite: 5].
    * Training features shape: `(4135, 50)` and Testing features shape: `(1034, 50)`[cite: 5].
* **Deep Learning Model (Sequential RNN)**:
    * **Embedding Layer**: `input_dim=5000`, `output_dim=32`, `input_length=50` to project words into dense vector spaces[cite: 5].
    * **SimpleRNN Layer**: `32 units` with `return_sequences=False` to capture sequential dependencies and context[cite: 5].
    * **Dense Layer**: `1 unit` with a `sigmoid` activation function for binary classification[cite: 5].
* **Compilation & Training**:
    * Compiled using `binary_crossentropy` loss, `adam` optimizer, and `accuracy` metrics[cite: 5].
    * Trained for `20 epochs` with a batch size of `32` and `validation_split=0.2`[cite: 5].
    * Model and tokenizer artifacts are serialized and stored as `spam_sms_model.h5` and `tokenizer.pkl`[cite: 3, 5].

---

## 5. Evaluation & Performance
* Evaluated on the independent test dataset (`x_test`, `y_test`)[cite: 1, 5]:
    * **Test Loss**: `0.1856`[cite: 1, 5]
    * **Test Accuracy**: `96.52%`[cite: 1, 5]
* Model accurately classifies incoming custom texts with high prediction probability confidence scores[cite: 1, 3, 5].

---

## 6. Tech Stack & Requirements
The project relies on the following core libraries stored in `requirements.txt`[cite: 1]:
* **Python**[cite: 1]
* **Flask**[cite: 1]
* **TensorFlow / Keras**[cite: 1]
* **NLTK**[cite: 1]
* **Pandas & NumPy**[cite: 1]
* **Scikit-Learn**[cite: 1]

---

## 7. Project Directory Structure
```text
SPAM DETECTION PROJECT/
│
├── EDA/
│   ├── spam_sms_model.h5        # Serialized Keras RNN model[cite: 3, 5]
│   └── tokenizer.pkl            # Serialized text tokenizer[cite: 3, 5]
│
├── templates/
│   └── index.html               # Frontend UI interface
│
├── SMS SPAM DATASET/
│   └── spam_sms.csv             # Raw source dataset[cite: 5]
│
├── app.py                       # Main Flask web application backend
├── requirements.txt             # Project dependencies specification[cite: 1]
└── README.md                    # Detailed documentation file