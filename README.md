# 📰 Fake News Detector using BERT

A deep learning NLP project that automatically classifies news articles as **FAKE** or **REAL** using Google's BERT model, fine-tuned on a dataset of 44,000+ news articles.

---

## 🎯 Problem Statement

Fake news is a growing threat to society — especially during elections, pandemics, and other critical events. Misinformation spreads faster than the truth and causes real harm. This project builds an AI-powered solution to detect fake news automatically, before it spreads.

---

## 🧠 How It Works

The project uses **BERT (Bidirectional Encoder Representations from Transformers)** — a state-of-the-art NLP model developed by Google. BERT reads the entire article bidirectionally (left-to-right and right-to-left simultaneously), giving it a deep understanding of context and meaning.

A single classification layer is added on top of BERT to output two probabilities:
- `P(FAKE)` — probability the article is fake
- `P(REAL)` — probability the article is real

```
Input Article
      ↓
BERT Tokenizer (WordPiece, max 512 tokens)
      ↓
BERT Encoder (12 layers, 768-dim, 110M parameters)
      ↓
[CLS] Token Output (768-dim)
      ↓
Linear Layer (768 → 2)
      ↓
Softmax → [P(Fake), P(Real)]
```

---

## 📁 Project Structure

```
fake-news-detector/
├── fake_news_detector.py     # Complete training pipeline
├── label_distribution.png    # Dataset visualization
├── README.md                 # Project documentation
└── .gitignore
```

---

## 📦 Dataset

**Fake and Real News Dataset** — Kaggle  
🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

| Split   | Count  |
|---------|--------|
| Total   | 44,898 |
| Fake    | 23,481 |
| Real    | 21,417 |
| Train   | 35,918 |
| Val     | 4,490  |
| Test    | 4,490  |

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Programming language |
| BERT (bert-base-uncased) | Pre-trained NLP model |
| HuggingFace Transformers | BERT implementation |
| PyTorch | Deep learning framework |
| Scikit-learn | Evaluation metrics |
| Pandas & NumPy | Data processing |
| Matplotlib & Seaborn | Visualization |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/harsh-thakur-rajput/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install transformers torch scikit-learn pandas seaborn tqdm
```

### 3. Download dataset
Download `Fake.csv` and `True.csv` from Kaggle and place them in the project folder.

### 4. Train the model
```bash
python fake_news_detector.py
```

### 5. Predict on new article
```python
result = predict("Your news article text here...", model, tokenizer, DEVICE)
print(result)
# Output: {'label': 'REAL', 'confidence': 99.85, 'prob_fake': 0.15, 'prob_real': 99.85}
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Train Accuracy | **100%** |
| Validation Accuracy | **100%** |
| Test Accuracy | **100%** |
| F1-Score (Fake) | **1.00** |
| F1-Score (Real) | **1.00** |

### Confusion Matrix
All 4,490 test articles correctly classified — zero misclassifications.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | bert-base-uncased |
| Max Token Length | 512 |
| Batch Size | 16 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Warmup Steps | 10% |
| Gradient Clipping | 1.0 |

---

## 💡 Key Insight

BERT achieves near-perfect accuracy on this dataset because fake news has very distinct linguistic patterns — sensationalist language, all-caps words, excessive punctuation, and conspiracy-style framing — which BERT's deep bidirectional attention mechanism picks up on very effectively.

> **Important:** Always provide the full article text, not just the headline, for best results. BERT needs context to make accurate predictions.

---

## 👨‍💻 Author

**Harsh Thakur**  
AI-Python Internship — Venura Tech  
GitHub: [@harsh-thakur-rajput](https://github.com/harsh-thakur-rajput)

---

## 📚 References

- Devlin et al. (2018) — [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- HuggingFace Transformers — https://huggingface.co/transformers
- Dataset — [Clément Bisaillon on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
