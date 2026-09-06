# 🎹 MusicTransformer Composer

> *Feed it the opening of Ode to Joy. Watch it compose the next 32 notes. Hear it play.*

A hands-on implementation of a **decoder-only Transformer** — the same architecture as GPT — trained on classic melodies and used to auto-regressively generate new music, playable directly in the notebook.

---

## 🧠 What You're Building

```
Token IDs  →  Embedding  →  PositionalEncoding
           →  TransformerBlock × 2
                 ├─ MultiHeadAttention (4 heads, causal mask)
                 └─ FeedForward (GELU)
                 ↑ both wrapped in: residual + LayerNorm
           →  Linear output head
           →  Logits (one score per note)
```

The model is trained on 50+ classic melodies (Ode to Joy, Twinkle Twinkle, Amazing Grace, …). Once trained, you provide a seed sequence and the model generates new notes one at a time via temperature sampling.

---

## 📋 TODOs at a Glance

| # | What | New Concept |
|---|------|-------------|
| 1 | Build vocab dict | same as class |
| 2 | `tokenize()` function | same as class |
| 3 | Positional Encoding (sin/cos) | 🆕 position fingerprints |
| 4 | Multi-Head Attention | 🆕 parallel attention heads |
| 5 | Residual + LayerNorm in TransformerBlock | 🆕 residual connections |
| 6 | Assemble MusicTransformer | 🆕 `nn.ModuleList` |
| 7 | Loss fn + optimiser | same as class |
| 8 | Training loop (backprop) | same as class |
| 9 | Temperature sampling | 🆕 sampling vs argmax |

---

## 🚀 Quick Start

### Option A — Google Colab (recommended, no install needed)

1. Upload `music_transformer_composer.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run cells **top to bottom**
3. Fill in each `___` blank (hints are provided in dropdowns)
4. Listen to your model compose music in ~3 minutes on CPU

### Option B — Local

```bash
git clone https://github.com/YOUR_USERNAME/music-transformer-composer.git
cd music-transformer-composer
pip install -r requirements.txt
jupyter notebook music_transformer_composer.ipynb
```

---

## 📦 Requirements

```
torch>=2.0
numpy
matplotlib
IPython
jupyter
```

See `requirements.txt` for pinned versions.

---

## 🎼 Dataset

50+ classic melodies encoded as space-separated note strings, covering:

| Octave 3 | Octave 4 | Octave 5 |
|----------|----------|----------|
| G3 A3 B3 | C4 D4 E4 F4 G4 A4 B4 | C5 D5 E5 |

Plus one special token: `REST` (silence).

Example:
```
"Ode to Joy": "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4 D4 D4"
```

---

## 🎵 Example Output

Given the seed `E4 E4 F4 G4 G4 F4 E4 D4` (Ode to Joy opening), the model generates the next 28 notes. Audio is rendered using a harmonic synthesis engine with piano envelope and reverb — no external audio libraries needed.

---

## 👁️ Bonus: Attention Visualisation

Step 7 visualises the 4 attention heads for any input context:

- **Bright diagonal** → each note attends mostly to itself
- **Off-diagonal brightness** → attending to earlier melodic context
- **Different patterns per head** → each head learned a different "musical question"

---

## 📁 Repo Structure

```
music-transformer-composer/
├── music_transformer_composer.ipynb   # Main notebook (fill in the blanks)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Python / Jupyter ignores
└── README.md                          # This file
```

---

## 📌 Rules

- Run cells **top to bottom**
- Replace every `___` with code — each blank has a hint comment above it
- Stuck? Open the `💡 Hint` dropdown in the notebook
- Nothing to install for Colab — runs on CPU in ~3 minutes

---

## 📄 License

MIT — feel free to use, modify, and share.
