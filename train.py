print("Training script started...")
# ==========================================================
# GPT-2 Fine-Tuning on Custom Dataset
# ==========================================================

from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ==========================================================
# Step 1: Load Tokenizer
# ==========================================================

print("Loading GPT-2 Tokenizer...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("Tokenizer loaded")

# GPT-2 doesn't have a padding token by default
tokenizer.pad_token = tokenizer.eos_token

# ==========================================================
# Step 2: Load Pre-trained GPT-2 Model
# ==========================================================

print("Loading GPT-2 Model...")

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.pad_token_id

print("Model loaded")

# ==========================================================
# Step 3: Load Dataset
# ==========================================================

print("Loading Dataset...")

dataset = load_dataset(
    "text",
    data_files={"train": "dataset/quotes.txt"}
)
print("Dataset loaded")
# ==========================================================
# Step 4: Tokenize Dataset
# ==========================================================

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
print("Dataset tokenized")
# ==========================================================
# Step 5: Prepare Data for GPT-2
# ==========================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ==========================================================
# Step 6: Training Configuration
# ==========================================================

training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    report_to="none",
)

# ==========================================================
# Step 7: Create Trainer
# ==========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)
print("Trainer created")
# ==========================================================
# Step 8: Train the Model
# ==========================================================

print("\nStarting Training...\n")

trainer.train()

# ==========================================================
# Step 9: Save Model
# ==========================================================

print("\nSaving Model...\n")

trainer.save_model("./model")
tokenizer.save_pretrained("./model")

# ==========================================================
# Step 10: Training Completed
# ==========================================================

print("=" * 50)
print(" GPT-2 Fine-Tuning Completed Successfully! ")
print("=" * 50)
print("Model saved inside the 'model' folder.")

