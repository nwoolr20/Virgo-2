# Neural Field Language Model (NFLM)

## True Generative Neural Field Architecture

This document describes the **true generative Neural Field Language Model** implementation in Virgo, which differs fundamentally from the original retrieval-based memory system.

## What is a Neural Field Language Model?

A neural field language model represents text as a continuous function in a learned coordinate space, where:
1. **Coordinates are learned** from the data (not hand-crafted)
2. **The field generates** token distributions (not retrieves embeddings)
3. **Interpolation is meaningful** in coordinate space

## Architecture Overview

```
Input tokens → CoordinateEncoder → Coordinates → GenerativeField → Token logits
                     ↑                                    ↓
                     └──────── Trained jointly ──────────┘
```

### Components

#### 1. CoordinateEncoder
Maps token sequences to continuous coordinates using a learned neural network:
- **Input**: Token IDs `[batch, seq_len]`
- **Processing**: Token embeddings → GRU → Coordinate projection
- **Output**: Coordinates `[batch, seq_len, coord_dim]` in [0,1]^coord_dim

Key: These coordinates are **learned** to make interpolation semantically meaningful.

#### 2. GenerativeField
Neural field that produces token distributions from coordinates:
- **Architecture**: SIREN network (sinusoidal activations)
- **Input**: Coordinates `[batch, seq_len, coord_dim]`
- **Output**: Token logits `[batch, seq_len, vocab_size]`

Key: This is a **continuous function** - you can query it at any point in coordinate space.

#### 3. NeuralFieldLM
Complete language model combining encoder and field:
- **Training**: Next-token prediction with cross-entropy loss
- **Generation**: Autoregressive sampling from field
- **Interpolation**: Linear interpolation in coordinate space

## Usage

### Basic Training and Generation

```python
from virgo import NeuralFieldLM, CharTokenizer, train_neural_field_lm
import torch

# Prepare data
texts = ["hello world", "hi there", "good morning"]
tokenizer = CharTokenizer()
tokenizer.build_vocab(texts)

# Create training data (next-token prediction)
train_data = []
for text in texts:
    tokens = tokenizer.encode(text, add_eos=False)
    if len(tokens) > 1:
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
        target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
        train_data.append((input_ids, target_ids))

# Create and train model
model = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=8)
train_neural_field_lm(model, train_data, epochs=10, lr=1e-3)

# Generate text
prompt = torch.tensor(tokenizer.encode("hello", add_eos=False), dtype=torch.long)
generated = model.generate(prompt, max_length=20, temperature=0.8)
print(tokenizer.decode(generated.tolist()))
```

### Sequence Interpolation

```python
# Interpolate between two sequences
seq1 = torch.tensor(tokenizer.encode("the cat", add_eos=False), dtype=torch.long)
seq2 = torch.tensor(tokenizer.encode("the dog", add_eos=False), dtype=torch.long)

# Interpolate at α=0.5
interpolated = model.interpolate_sequences(seq1, seq2, alpha=0.5)
print(tokenizer.decode(interpolated.tolist()))
```

## Key Differences: Old System vs. New NFLM

| Aspect | Original System | True Neural Field LM |
|--------|----------------|---------------------|
| **Purpose** | Retrieval-augmented memory | Text generation |
| **Coordinates** | Hand-crafted heuristics (6D) | Learned from data (8D) |
| **Field Output** | Pre-computed SBERT embeddings | Token logits (generative) |
| **Training** | Memorize fixed embeddings | Language modeling (next-token) |
| **Interpolation** | No semantic meaning | Produces coherent text |
| **Use Case** | Context retrieval | Text generation |

### Original System (Memory/Retrieval)
```python
# Stores pre-computed embeddings
system.store("My name is Alice", speaker_id=0)
system.fit_field()  # Field compresses SBERT embeddings

# Retrieves similar memories
results = system.retrieve("What is my name?", k=3)
# Returns: stored memories ranked by similarity
```

### New NFLM (Generative)
```python
# Learns from text data
model.train(texts)  # Field learns to generate

# Generates new text
generated = model.generate(prompt)
# Returns: newly generated text
```

## Demo

Run the comprehensive demo:
```bash
python scripts/demo_nflm.py
```

This demonstrates:
- Training on a small corpus
- Autoregressive generation
- Coordinate space interpolation
- Continuous semantic representations

## Performance Characteristics

### Model Size
- Parameters: ~2-5M (depends on hidden dimensions)
- Coordinate dimensions: 8 (default)
- Vocabulary: Flexible (character or word-level)

### Training
- Task: Next-token prediction
- Loss: Cross-entropy
- Optimization: Adam
- Typical training: 10-50 epochs on small datasets

### Generation
- Method: Autoregressive sampling
- Temperature control: Adjustable randomness
- Speed: ~10-50 tokens/second (CPU)

## Why This is a TRUE Neural Field LM

1. **Continuous Function**: The field is a continuous function approximator, not a lookup table
2. **Learned Coordinates**: Coordinates are learned to capture semantic structure
3. **Generative**: Creates new text, doesn't just retrieve existing text
4. **Interpolatable**: Linear interpolation in coordinate space produces meaningful results
5. **Joint Training**: Encoder and field trained together on language modeling

## Comparison with Traditional LMs

### vs. Transformer LMs (GPT, BERT)
- **NFLM**: Continuous coordinate space, smaller model, novel interpolation
- **Transformers**: Discrete attention, larger models, state-of-the-art performance

### vs. RNN/LSTM LMs
- **NFLM**: Explicit coordinate representation, SIREN field for generation
- **RNN/LSTM**: Implicit hidden state, recurrent connections

### vs. Original Virgo (Memory System)
- **NFLM**: Generative, learned coordinates, produces new text
- **Memory System**: Retrieval-based, hand-crafted coordinates, recalls stored text

## Future Enhancements

Possible improvements to explore:
1. **Larger Vocabulary**: Word-level or BPE tokenization
2. **Deeper Networks**: More SIREN layers for complex patterns
3. **Conditioning**: Add control signals (style, topic, etc.)
4. **Hybrid Models**: Combine with retrieval for RAG-like systems
5. **Multi-Modal**: Extend coordinates to image/audio inputs

## Research Context

Neural field language models represent a novel approach to language modeling that:
- Provides explicit geometric structure to language representations
- Enables smooth interpolation between linguistic concepts
- Offers interpretable coordinate spaces
- Bridges continuous function approximation and discrete language

This implementation serves as a foundation for exploring these ideas in practice.

## References

- SIREN: Implicit Neural Representations with Periodic Activation Functions
- Neural Fields: Continuous function approximation with neural networks
- Language Modeling: Next-token prediction for text generation

## License

MIT License - See LICENSE file for details
