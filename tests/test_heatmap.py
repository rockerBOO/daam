from daam.heatmap import GlobalHeatMap, WordHeatMap
import torch
import pytest


# Emulating hugging face tokenizer
class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt):
        return self.tokenizer(prompt)


@pytest.fixture
def sample_tokenizer():
    # Sample tokenizer for testing
    def tokenize(x):
        return x.split()

    return Tokenizer(tokenize)


@pytest.fixture
def sample_heat_maps():
    # Sample heat maps tensor for testing
    return [torch.randn((4, 5, 5)), torch.randn((4, 5, 5))]


@pytest.fixture
def sample_global_heatmap(sample_tokenizer, sample_heat_maps):
    # Sample GlobalHeatMap for testing
    return GlobalHeatMap(sample_tokenizer, "Test prompt word test", sample_heat_maps)


def test_compute_word_heat_map(sample_global_heatmap):
    # Test compute_word_heat_map method
    word_heat_map = sample_global_heatmap.compute_word_heat_map("word")
    assert isinstance(word_heat_map, WordHeatMap)
    assert torch.is_tensor(word_heat_map.heatmap)
    assert word_heat_map.word == "word"


def test_compute_word_heat_map_with_indices(sample_global_heatmap):
    # Test compute_word_heat_map method with specified indices
    word_heat_map = sample_global_heatmap.compute_word_heat_map(
        "word", word_idx=1, offset_idx=2, batch_idx=1
    )
    assert isinstance(word_heat_map, WordHeatMap)
    assert torch.is_tensor(word_heat_map.heatmap)
    assert word_heat_map.word == "word"
