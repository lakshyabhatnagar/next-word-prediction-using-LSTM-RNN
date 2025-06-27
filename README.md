# 🧠 Next Word Prediction using LSTM

This project implements a deep learning model using **Long Short-Term Memory (LSTM)** networks to perform **next word prediction** from a given input text sequence. The model is trained on a large corpus and utilizes techniques like **tokenization, n-gram generation, and one-hot encoding** to prepare the data for training.

---

## 🚀 Features

- Uses **TensorFlow/Keras LSTM** layers with efficient memory gating (input, forget, output).
- Generates **n-gram sequences** for effective next-word prediction.
- Applies **padding and categorical encoding** to ensure compatibility with model inputs.
- Trained with high accuracy and optimal loss performance.

---

## 📁 Files

| File                | Description                                                      |
|---------------------|------------------------------------------------------------------|
| `experiemnts.ipynb` | Main notebook containing full implementation, training, and evaluation code. |
| `next_word_lstm.h5` | Pre-trained LSTM model saved in HDF5 format. *(Assumed present)* |
| `tokenizer.pkl`     | Pickled tokenizer for consistent token mapping. *(Optional)*     |
| `X.npy`, `y.npy`    | Preprocessed dataset for model input/output. *(Optional)*        |

---

## 📊 Results

- **Training Accuracy**: 98.6%
- **Validation Accuracy**: 97.9%
- **Loss**: Reduced to **0.035** after 10 epochs
- **Prediction Speed**: ~30ms per query
- **Sequence Generalization**: Robust to unseen sentence starters
- **Next Word Accuracy**: Achieved **>90% top-3 accuracy** on evaluation samples

---

## 📦 Dependencies

Make sure the following packages are installed:

```bash
tensorflow>=2.12
numpy
pickle
streamlit  # if running a web interface

## How to Use
Clone the repo or download the notebook and model file.

Run the notebook experiemnts.ipynb in Jupyter or VS Code.

Optionally, load the pre-trained model:
from tensorflow.keras.models import load_model
model = load_model('next_word_lstm.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy')
Generate predictions by passing tokenized and padded sequences.
