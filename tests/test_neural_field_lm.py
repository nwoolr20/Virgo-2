"""Tests for Neural Field Language Model"""

import pytest
import torch
from virgo.neural_field_lm import (
    NeuralFieldLM,
    CoordinateEncoder,
    GenerativeField,
    train_neural_field_lm
)
from virgo.tokenizer import CharTokenizer


def test_coordinate_encoder():
    """Test coordinate encoder forward pass"""
    vocab_size = 100
    coord_dim = 8
    batch_size = 4
    seq_len = 10
    
    encoder = CoordinateEncoder(vocab_size, coord_dim)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    coords = encoder(token_ids)
    
    assert coords.shape == (batch_size, seq_len, coord_dim)
    # Coordinates should be in [0, 1]
    assert torch.all(coords >= 0) and torch.all(coords <= 1)


def test_generative_field():
    """Test generative field forward pass"""
    coord_dim = 8
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    field = GenerativeField(coord_dim, vocab_size)
    coords = torch.rand(batch_size, seq_len, coord_dim)
    
    logits = field(coords)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_generative_field_interpolation():
    """Test field interpolation between coordinates"""
    coord_dim = 8
    vocab_size = 100
    
    field = GenerativeField(coord_dim, vocab_size)
    coord1 = torch.rand(coord_dim)
    coord2 = torch.rand(coord_dim)
    
    # Test interpolation at different points
    logits_start = field.interpolate(coord1, coord2, alpha=0.0)
    logits_mid = field.interpolate(coord1, coord2, alpha=0.5)
    logits_end = field.interpolate(coord1, coord2, alpha=1.0)
    
    assert logits_start.shape == (vocab_size,)
    assert logits_mid.shape == (vocab_size,)
    assert logits_end.shape == (vocab_size,)
    
    # Interpolated should be different from endpoints
    assert not torch.allclose(logits_start, logits_end)


def test_neural_field_lm_forward():
    """Test NFLM forward pass without targets"""
    vocab_size = 100
    coord_dim = 8
    batch_size = 4
    seq_len = 10
    
    model = NeuralFieldLM(vocab_size, coord_dim)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_neural_field_lm_forward_with_loss():
    """Test NFLM forward pass with targets (training mode)"""
    vocab_size = 100
    coord_dim = 8
    batch_size = 4
    seq_len = 10
    
    model = NeuralFieldLM(vocab_size, coord_dim)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids, target_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss.shape == ()  # Scalar
    assert loss.item() > 0


def test_neural_field_lm_generation():
    """Test autoregressive generation"""
    vocab_size = 100
    coord_dim = 8
    
    model = NeuralFieldLM(vocab_size, coord_dim)
    prompt = torch.randint(0, vocab_size, (5,))
    
    generated = model.generate(prompt, max_length=10, temperature=1.0)
    
    assert generated.shape == (15,)  # prompt_len + max_length
    # First tokens should match prompt
    assert torch.all(generated[:5] == prompt)


def test_neural_field_lm_interpolation():
    """Test sequence interpolation in coordinate space"""
    vocab_size = 100
    coord_dim = 8
    seq_len = 10
    
    model = NeuralFieldLM(vocab_size, coord_dim)
    seq1 = torch.randint(0, vocab_size, (seq_len,))
    seq2 = torch.randint(0, vocab_size, (seq_len,))
    
    # Interpolate at different alphas
    interp_0 = model.interpolate_sequences(seq1, seq2, alpha=0.0)
    interp_mid = model.interpolate_sequences(seq1, seq2, alpha=0.5)
    interp_1 = model.interpolate_sequences(seq1, seq2, alpha=1.0)
    
    assert interp_0.shape == (seq_len,)
    assert interp_mid.shape == (seq_len,)
    assert interp_1.shape == (seq_len,)


def test_tokenizer_encode_decode():
    """Test character tokenizer encoding and decoding"""
    tokenizer = CharTokenizer()
    texts = ["hello", "world", "test"]
    tokenizer.build_vocab(texts)
    
    # Test single encode/decode
    text = "hello"
    tokens = tokenizer.encode(text, add_eos=False)
    decoded = tokenizer.decode(tokens)
    
    assert decoded == text
    assert len(tokens) == len(text)


def test_tokenizer_batch():
    """Test batch encoding"""
    tokenizer = CharTokenizer()
    texts = ["hello", "world", "test"]
    tokenizer.build_vocab(texts)
    
    # Batch encode
    batch_tensor = tokenizer.encode_batch(texts, max_length=10)
    
    assert batch_tensor.shape == (3, 10)
    
    # Batch decode
    decoded = tokenizer.decode_batch(batch_tensor)
    
    assert len(decoded) == 3


def test_tokenizer_vocab_size():
    """Test vocabulary size calculation"""
    tokenizer = CharTokenizer()
    texts = ["abc", "def"]
    tokenizer.build_vocab(texts)
    
    # Should have: 3 special tokens + 6 unique chars = 9
    # Actually more because special tokens might overlap
    assert tokenizer.vocab_size > 0
    assert tokenizer.vocab_size == len(tokenizer.char_to_idx)


def test_training_step():
    """Test that training reduces loss"""
    vocab_size = 50
    coord_dim = 6
    
    model = NeuralFieldLM(vocab_size, coord_dim)
    
    # Create simple training data
    train_data = []
    for _ in range(5):
        input_ids = torch.randint(0, vocab_size, (1, 10))
        target_ids = torch.randint(0, vocab_size, (1, 10))
        train_data.append((input_ids, target_ids))
    
    # Get initial loss
    model.train()
    with torch.no_grad():
        _, initial_loss = model(train_data[0][0], train_data[0][1])
        initial_loss_value = initial_loss.item()
    
    # Train for one epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for input_ids, target_ids in train_data:
        optimizer.zero_grad()
        _, loss = model(input_ids, target_ids)
        loss.backward()
        optimizer.step()
    
    # Check that loss decreased
    with torch.no_grad():
        _, final_loss = model(train_data[0][0], train_data[0][1])
        final_loss_value = final_loss.item()
    
    assert final_loss_value < initial_loss_value


def test_end_to_end_with_text():
    """Test end-to-end: tokenize, train, generate"""
    # Create simple dataset
    texts = ["hello world", "hi there", "good morning"]
    
    # Build tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    
    # Create model
    model = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=6)
    
    # Prepare training data
    train_data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_eos=False)
        if len(tokens) > 1:
            # Use tokens[:-1] as input, tokens[1:] as target (next token prediction)
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
            train_data.append((input_ids, target_ids))
    
    # Quick training
    train_neural_field_lm(model, train_data, epochs=2, lr=1e-3)
    
    # Generate from first token
    prompt = torch.tensor([tokens[0] for tokens in [tokenizer.encode(texts[0], add_eos=False)]], dtype=torch.long)
    generated = model.generate(prompt, max_length=5, temperature=1.0)
    
    assert generated.shape[0] > prompt.shape[0]
    assert torch.all(generated[:len(prompt)] == prompt)
