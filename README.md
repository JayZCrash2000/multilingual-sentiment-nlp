[multilingual_sentiment_README.md](https://github.com/user-attachments/files/26325985/multilingual_sentiment_README.md)
# 🌐 Multilingual Sentiment Classification — 13 Indian Languages

Fine-tuning `google/gemma-3-1b-it` with **4-bit QLoRA** for binary sentiment classification (Positive / Negative) across 13 Indian languages. Evaluated using **Macro F1-Score**.

---

## 🗣️ Languages Covered

| Code | Language | Code | Language |
|---|---|---|---|
| `as` | Assamese | `mr` | Marathi |
| `bd` | Bodo | `or` | Odia |
| `bn` | Bengali | `pa` | Punjabi |
| `gu` | Gujarati | `ta` | Tamil |
| `hi` | Hindi | `te` | Telugu |
| `kn` | Kannada | `ur` | Urdu |
| `ml` | Malayalam | | |

---

## 🧰 Tech Stack

| Component | Tool / Version |
|---|---|
| Base Model | `google/gemma-3-1b-it` |
| Fine-tuning | QLoRA via HuggingFace PEFT `0.12.0` |
| Quantization | 4-bit BitsAndBytes `0.45.3` |
| Training | `trl` SFTTrainer `0.11.4` |
| Transformers | `4.50.0` (first version with Gemma-3 support) |
| Environment | Kaggle (Tesla P100 16GB GPU) |
| Evaluation | Macro F1-Score (scikit-learn) |

---

## 📁 Project Structure

```
multilingual-sentiment-nlp/
│
├── Multilingual_Sentiment_Classification.ipynb   # Full pipeline notebook
├── requirements.txt
└── README.md
```

---

## 🏗️ Pipeline

```
Raw multilingual text (13 Indian languages)
        ↓
Unicode NFC normalization + HTML/URL cleaning
        ↓
Instruction prompt formatting (language-aware)
        ↓
Gemma-3-1B-IT — 4-bit quantized (BitsAndBytes)
        ↓
QLoRA adapters (r=16, target: q/k/v/o projections)
        ↓
SFTTrainer fine-tuning — 4 epochs
        ↓
Token-level label extraction → Macro F1 evaluation
```

---

## 🔑 Prompt Template

```
<start_of_turn>user
You are a sentiment analysis expert. Classify the sentiment of the following {language} sentence.
Respond with exactly one word: Positive or Negative.

Sentence: {text}
<end_of_turn>
<start_of_turn>model
{label}<end_of_turn>
```

---

## ⚙️ Key Configuration

```python
# 4-bit Quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# LoRA Config
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

# Training
epochs = 4          # beyond 4-5 epochs causes overfitting on private leaderboard
batch_size = 4
gradient_accumulation_steps = 4   # effective batch = 16
learning_rate = 2e-4
max_seq_len = 512
```

---

## 📊 Dataset

| Split | Samples | Classes |
|---|---|---|
| Train | 900 | Positive (456), Negative (444) |
| Test | 100 | — |

Languages are roughly balanced (~65–76 samples each in train).

---

## 💡 Key Findings & Lessons Learned

- **Explicit language context in the prompt** significantly improves cross-lingual generalization
- **Instruction-tuned base** (Gemma-IT) outperforms the base variant for classification tasks
- **4–5 epochs is optimal** — training beyond this caused private leaderboard degradation despite improving train metrics (overfitting)
- **NFC Unicode normalization** is critical for Indic scripts to avoid tokenization inconsistencies
- Loading tokenizer from **HuggingFace Hub** was necessary due to a corrupted `tokenizer.json` in the local Kaggle model copy

---

## 🚀 How to Run

> Requires GPU with ~16GB VRAM — Kaggle free T4/P100 recommended

```bash
pip install -r requirements.txt
# Set your HF_TOKEN in Kaggle secrets or as an environment variable
jupyter notebook Multilingual_Sentiment_Classification.ipynb
```

---



---

## 👤 Author

**Jeet Kumar** — [GitHub](https://github.com/JayZCrash2000)
