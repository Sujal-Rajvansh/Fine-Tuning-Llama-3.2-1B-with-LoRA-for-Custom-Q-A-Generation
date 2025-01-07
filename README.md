# 🦙 Fine-Tuning Llama-3.2-1B with LoRA for Custom Q&A Generation 🚀

This project demonstrates fine-tuning **Meta's Llama-3.2-1B model** using **LoRA (Low-Rank Adaptation)** for efficient and lightweight customization. The resulting model can be used for **custom question-answering generation** or other text-based tasks. The fine-tuning process is resource-efficient, leveraging LoRA for scalable parameter updates without requiring full model re-training.

## 🌟 Key Features

- **Efficient Fine-Tuning with LoRA**: Minimal hardware requirements for effective customization.  
- **Custom Q&A Dataset**: Fine-tune the model to generate domain-specific answers.  
- **Easy Integration**: Compatible with Hugging Face's `transformers` and `pipeline` for text generation.  
- **Google Drive Integration**: Save and load fine-tuned models for reuse and sharing.

## 🛠️ Installation

Install the necessary libraries:
```bash
pip install --upgrade transformers torch bitsandbytes peft
Log in to Hugging Face:

python
Copy code
from huggingface_hub import login
login("your_huggingface_token")
🚀 Fine-Tuning Process
1️⃣ Load the Pre-Trained Model

python
Copy code
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
2️⃣ Apply LoRA for Efficient Fine-Tuning

python
Copy code
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
3️⃣ Prepare the Dataset

python
Copy code
data = [{"question": "What is AI?", "answer": "AI is the simulation of human intelligence in machines."}]
train_texts = [f"Q: {item['question']} A: {item['answer']}" for item in data]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
Create a PyTorch dataset:

python
Copy code
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        self.encodings["labels"] = self.encodings["input_ids"].clone()

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = QADataset(train_texts, tokenizer)
4️⃣ Train the Model

python
Copy code
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
5️⃣ Save the Fine-Tuned Model

python
Copy code
model.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
🧠 Using the Fine-Tuned Model
Load the Fine-Tuned Model

python
Copy code
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "/content/drive/MyDrive/fine_tuned_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
Generate Text

python
Copy code
prompt = "The key to life is"
generated_text = generator(prompt, max_length=50)
print(generated_text)
📚 Example Interaction
Input: What is AI?
Output: AI is the simulation of human intelligence in machines.

Prompt: The key to life is
Generated: The key to life is to be true to yourself. If you are true to yourself, you will understand yourself better.

📦 Model Storage and Reuse
Save the fine-tuned model to Google Drive:

python
Copy code
model.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
Load the fine-tuned model:

python
Copy code
model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/fine_tuned_model")
🎯 Customization Options
Expand Dataset: Add more questions and answers to improve the model's responses.
Adjust Training Arguments: Modify num_train_epochs, learning_rate, or batch_size for better performance.
Target Specific Layers: Use LoRA on specific layers for fine-tuned optimization.
