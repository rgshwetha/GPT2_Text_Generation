from datasets import load_dataset
from transformers import GPT2Tokenizer

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
dataset = load_dataset(
    "text",
    data_files={"train": "dataset/quotes.txt"}
)

print("Tokenizing...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Tokenization complete!")
print(tokenized_dataset)