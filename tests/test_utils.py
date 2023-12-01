import pytest
from daam.utils import compute_token_merge_indices


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


def test_compute_token_merge_indices_basic(sample_tokenizer):
    # Test basic functionality
    prompt = "This is a sample prompt"
    word = "sample"
    merge_idx, word_idx = compute_token_merge_indices(sample_tokenizer, prompt, word)
    assert merge_idx == [3]
    assert word_idx is None


def test_compute_token_merge_indices_with_indices(sample_tokenizer):
    # Test with specified indices
    prompt = "This is a sample prompt"
    word = "sample"
    merge_idx, word_idx = compute_token_merge_indices(
        sample_tokenizer, prompt, word, word_idx=2, offset_idx=1
    )
    assert merge_idx == [2]
    assert word_idx == 2


def test_compute_token_merge_indices_word_not_found(sample_tokenizer):
    # Test when the word is not found in the prompt
    prompt = "This is a sample prompt"
    word = "not_in_prompt"
    with pytest.raises(
        ValueError, match="Word 'not_in_prompt' not found in the prompt."
    ):
        compute_token_merge_indices(sample_tokenizer, prompt, word)


def test_compute_token_merge_indices_empty_prompt(sample_tokenizer):
    # Test with an empty prompt
    prompt = ""
    word = "sample"
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        compute_token_merge_indices(sample_tokenizer, prompt, word)


def test_compute_token_merge_indices_multiple_occurrences(sample_tokenizer):
    # Test when the word has multiple occurrences in the prompt
    prompt = "This is a sample prompt with sample words. Another sample is here."
    word = "sample"
    merge_idx, word_idx = compute_token_merge_indices(sample_tokenizer, prompt, word)
    assert merge_idx == [3, 6, 9]
    assert word_idx is None


def test_compute_token_merge_indices_offset(sample_tokenizer):
    # Test with a non-zero offset
    prompt = "This is a sample prompt"
    word = "sample"
    merge_idx, word_idx = compute_token_merge_indices(
        sample_tokenizer, prompt, word, offset_idx=2
    )
    assert merge_idx == [5]
    assert word_idx is None


def test_compute_token_merge_indices_word_case_insensitive(sample_tokenizer):
    # Test case-insensitive word matching
    prompt = "This is a Sample prompt"
    word = "sAmPle"
    merge_idx, word_idx = compute_token_merge_indices(sample_tokenizer, prompt, word)
    assert merge_idx == [3]
    assert word_idx is None


def test_compute_token_merge_indices_word_not_found_case_insensitive(sample_tokenizer):
    # Test when the case-insensitive word is not found in the prompt
    prompt = "This is a Sample prompt"
    word = "not_in_prompt"
    with pytest.raises(
        ValueError, match="Word 'not_in_prompt' not found in the prompt."
    ):
        compute_token_merge_indices(sample_tokenizer, prompt, word)
