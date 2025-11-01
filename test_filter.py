import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

with open("val_dataset.json", "r") as f:
    data = json.load(f)

model_name = "nandansarkar/qwen3_0-6B_filter_13_epochs"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

with open("qwen3_0-6B_filter_13_epochs_outputs.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "text", "model_output"])

    for i, example in enumerate(tqdm(data)):
        instruction = example["instruction"]
        input_text = example["input"]
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{instruction}\n{input_text}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=0.7, top_p=0.9)
        output_text = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
        writer.writerow([i, text, output_text])
