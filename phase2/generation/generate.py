import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def generate_answer(prompt, max_tokens=256):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            # temperature=0.0  # deterministic
        )

    answer = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return answer
