# Transformer from Scratch: German → English Translation

A small encoder-decoder Transformer implemented from scratch in PyTorch, trained to
translate German sentences into English on the Multi30k dataset.

## Credits
- Based on the 'Modern Route-Full Stack GenerativeAI And Agentic AI Bootcamp' by Krish Naik & Sunny Savita.
- Github Link: https://github.com/sunnysavita10/Full-Stack-GenAI-Bootcamp-1.0
- Modified by Avantika Poddar.

## Research Papers for understanding Encoder-Decoder architecture
- Transformer-attention-all-you-need:
https://arxiv.org/pdf/1706.03762
- OG blog on the transformer:
https://jalammar.github.io/illustrated-transformer/

## Transformer Architecture to understand the code 

<img width="2026" height="1106" alt="image" src="https://github.com/user-attachments/assets/32fbd534-d231-49b7-96f6-aaaee5d8c95c" />




## What's inside
- Custom tokenization (spaCy) and vocabulary building
- Positional encoding, multi-head attention, feed-forward, encoder and decoder layers
- Padding and causal masks
- Training and evaluation loop with gradient clipping
- Greedy-decoding inference for translating new sentences

## Model
- ~13.8M parameters (d_model=256, 8 heads, 3 layers, d_ff=512)
- Adam optimizer, cross-entropy loss (padding ignored)

## Results (1 epoch, ~2 min on CPU)
| Metric | Value |
|---|---|
| Train loss / PPL | 4.539 / 93.6 |
| Validation loss / PPL | 3.734 / 41.8 |

After one epoch the model produces fluent English but often wrong meaning.
Longer training should improve this.

## How to run
1. Install dependencies: `pip install torch spacy tqdm`
2. Download spaCy models:
   `python -m spacy download de_core_news_sm`
   `python -m spacy download en_core_web_sm`
3. Download the Multi30k files into a `data/` folder and set the paths in the notebook.
4. Run the notebook.



## Future work
- Train for more epochs (about 10 to 30) with N_EPOCHS.
- Try a higher learning rate with warmup, or a larger model (D_MODEL, NUM_LAYERS).
- Use beam search instead of greedy decoding.
- Measure quality with BLEU score instead of only loss and perplexity.
