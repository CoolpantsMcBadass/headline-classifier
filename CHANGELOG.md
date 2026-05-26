# Changelog

## [Unreleased] — 2026-05-26

### Added
- Integrated trained model into Clear Eyes Chrome extension (v0.7.0)
  - Offscreen document architecture for ONNX Runtime WASM inference
  - `ml_classifier.js` content script collects headlines and sends to background
  - `background.js` proxies ML_CLASSIFY to offscreen document
  - Confidence threshold: 0.40 (interim, pending retraining on rage-bait data)
  - Model served from HuggingFace Hub: `CoolpantsMcBadass/headline-classifier`

### Known limitations
- Model was trained primarily on clickbait-style headlines; top scores on Fox News content peak at ~0.46
- Rage-bait and political opinion editorial framing not yet well-represented in training data
- Retraining with `data/custom_examples.csv` planned to address Fox News-style manipulation patterns

---

## [v1.0] — 2026-05-26 (Training complete)

### Model
- Base: `distilbert-base-uncased` fine-tuned for binary text classification
- Task: manipulative headline detection (label 0 = legitimate, label 1 = manipulative)
- Eval F1: 0.998 (on held-out test split)
- Export: int8 quantized ONNX via `optimum` + `ORTQuantizer`
- Hosted at: `CoolpantsMcBadass/headline-classifier` on HuggingFace Hub
  - `onnx/model_quantized.onnx`
  - Tokenizer and config at repo root

### Training data (~57,000 examples)
| Source | Examples | Notes |
|--------|----------|-------|
| christinacdl/clickbait_detection_dataset | ~30,000 | Primary clickbait dataset |
| liamvbetts/sarcastic-news-headlines-1 | ~26,700 | Sarcasm proxy for rage-bait |
| daviddaubner/misinformation-detection | ~20,700 | Misinformation headlines |
| FredZhang7/all-scam-spam (filtered ≤200 chars) | ~9,700 | Scam/spam short texts |

### Training environment
- Google Colab (T4 GPU, free tier)
- Trainer: HuggingFace `transformers.Trainer`
- 3 epochs, warmup_steps=300, weight_decay=0.01

### Fixes applied during development
- `VideoReader` ImportError: added `pip install --upgrade datasets`
- `Trainer(tokenizer=...)` deprecated: renamed to `processing_class`
- `warmup_ratio` deprecated: replaced with `warmup_steps=300`
- LIAR dataset broken (script-based, not supported): skipped, empty DataFrame fallback
- Scam dataset label field is `is_spam` not `label`
- ONNX inference: `file_name='model_quantized.onnx'`, `provider='CPUExecutionProvider'`, `device=-1`
