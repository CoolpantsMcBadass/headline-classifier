# Headline Manipulation Classifier

A fine-tuned DistilBERT model that detects manipulative framing in news headlines and social media posts — clickbait, rage-bait, fear-bait, financial scam ads, and engagement bait.

Built as a replacement for regex-based pattern matching, which struggles with context. A transformer model understands that "tornadoes slam the Midwest" is legitimate weather reporting while "Senator SLAMS colleague" is manufactured outrage.

## How it works

1. Fine-tunes `distilbert-base-uncased` on the [Webis Clickbait Corpus 2017](https://webis.de/data/webis-clickbait-17.html) (38k labeled examples)
2. Supplements with hand-labeled examples from `data/custom_examples.csv`
3. Exports to ONNX (int8 quantized, ~65MB) for use in browser extensions via [Transformers.js](https://huggingface.co/docs/transformers.js)

**Output:** binary label — `manipulative` or `legitimate`

## Run it

Open in Google Colab (free T4 GPU, ~25 min to train):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOURUSERNAME/headline-classifier/blob/main/headline_classifier.ipynb)

> Runtime → Change runtime type → T4 GPU before running

## Contribute labeled examples

The model improves with more hand-labeled data. Add examples to `data/custom_examples.csv`:

```
text,label
"Your headline here",1
"A legitimate headline",0
```

`1 = manipulative`, `0 = legitimate`

PRs with new labeled examples are welcome — especially edge cases the model gets wrong.

## What counts as manipulative

| Type | Example |
|---|---|
| Clickbait | "You won't believe what happened next" |
| Rage-bait | "Senator SLAMS colleague in explosive hearing" |
| Fear-bait | "This food is silently destroying your gut" |
| Financial scam ad | "Over $70B in Unclaimed Funds — Enter Name & State" |
| Engagement bait | "Share this before they delete it" |
| Too-good-to-be-true | "Doctors HATE this one weird trick" |

## What does NOT count as manipulative

- Legitimate breaking news that uses strong verbs ("tornadoes slam Midwest")
- Analytical questions ("Is it a problem if the Fed speaks too much?")
- Straightforward trending summaries ("Why Dominion Energy is trending")
- Standard weather/crime reporting

## Output format (ONNX)

The exported model is compatible with Transformers.js for in-browser inference:

```js
import { pipeline } from '@xenova/transformers';

const classifier = await pipeline('text-classification', 'path/to/model');
const result = await classifier('You won't believe what happened next');
// { label: 'manipulative', score: 0.97 }
```

## Requirements

Training runs entirely in Google Colab. No local setup needed.

Dependencies (installed automatically in the notebook):
- `transformers`
- `datasets`
- `torch`
- `optimum[exporters]`
- `onnxruntime`
- `scikit-learn`
- `seaborn`
