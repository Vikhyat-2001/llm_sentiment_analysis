from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_data(dataset_name="imdb", model_name="bert-base-uncased"):
    """
    Load and preprocess a sentiment analysis dataset.
    """
    # Load dataset
    dataset = load_dataset(dataset_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset