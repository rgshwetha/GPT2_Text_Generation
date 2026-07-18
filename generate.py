from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./model"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

prompt = input("Enter a prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=80,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

print("\nGenerated Text:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))