"""Integration tests for neural field language model."""

import pytest
import torch
from virgo import NeuralFieldLM, CharTokenizer, train_neural_field_lm


def test_end_to_end_training_and_generation():
    """Complete end-to-end test of training and generation."""
    # Create small training data
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew in the sky"
    ]
    
    # Build tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    
    # Prepare training data
    train_data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_eos=False)
        if len(tokens) > 1:
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
            train_data.append((input_ids, target_ids))
    
    # Create and train model
    model = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=8)
    train_neural_field_lm(model, train_data, epochs=5, lr=1e-3)
    
    # Test generation
    prompt = tokenizer.encode("the", add_eos=False)
    input_ids = torch.tensor(prompt, dtype=torch.long)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=20, temperature=1.0)
    
    generated_text = tokenizer.decode(output_ids.tolist())
    
    # Should generate something
    assert len(generated_text) > len("the")
    assert generated_text.startswith("the")


def test_model_save_and_load():
    """Test model checkpoint saving and loading."""
    import tempfile
    from pathlib import Path
    
    # Create and train small model
    texts = ["hello world", "test text"]
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    
    model1 = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=4)
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.pt"
        
        checkpoint = {
            'model_state_dict': model1.state_dict(),
            'vocab_size': tokenizer.vocab_size,
            'coord_dim': 4,
            'char_to_idx': tokenizer.char_to_idx,
            'idx_to_char': tokenizer.idx_to_char,
        }
        torch.save(checkpoint, save_path)
        
        # Load checkpoint
        loaded = torch.load(save_path, map_location='cpu', weights_only=False)
        
        model2 = NeuralFieldLM(vocab_size=loaded['vocab_size'], coord_dim=loaded['coord_dim'])
        model2.load_state_dict(loaded['model_state_dict'])
        
        # Verify models are equivalent
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

