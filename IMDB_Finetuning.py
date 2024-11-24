from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
import os

# Loading the dataset
dataset = load_dataset(os.environ.get('Dataset'))
print(dataset)

# Calculating the proportion of positive labels in the training set
positive_ratio = np.array(dataset['train']['label']).sum() / len(dataset['train']['label'])
print("Positive label ratio:", positive_ratio)

# Defining the model checkpoint
model_checkpoint = os.environ.get('LLM_Model')

# Mapping of labels
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

# Loading the pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

# Initializing the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# Adding padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenization function
def tokenize(examples):
    text = examples["text"]
    tokenizer.truncation_side = "left"
    return tokenizer(text, return_tensors="np", truncation=True, max_length=512)

# Applying tokenization to the datasets
tokenized_data = dataset.map(tokenize, batched=True)

# Initializing data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Loading accuracy metric
accuracy = evaluate.load("accuracy")

# Function to compute metrics
def compute_metrics(predictions_and_labels):
    predictions, labels = predictions_and_labels
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

# Sample texts for prediction
sample_texts = [
    "It was good.",
    "Not a fan, don't recommend.",
    "Better than the first one.",
    "This is not worth watching even once.",
    "This one is a pass."
]

# Displaying untrained model predictions
print("Untrained model predictions:")
for text in sample_texts:
    inputs = tokenizer.encode(text, return_tensors="pt")
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(f"{text} - {id2label[predictions.item()]}")

# Setting up LoRA configuration for model fine-tuning
lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=['q_lin']
)

# Applying LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training hyperparameters
learning_rate = 1e-3
batch_size = 4
epochs = 10

# Defining training arguments
training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-text-classification",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Creating Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Training the model
trainer.train()

# Testing the trained model
model.to('cuda')
print("Trained model predictions:")
for text in sample_texts:
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda")
    logits = model(inputs).logits
    predictions = torch.max(logits, 1).indices
    print(f"{text} - {id2label[predictions.item()]}")

# Pushing the model to Hugging Face Hub
model_name = 'samadpls'
model_id = f"{model_name}/sentiment-analysis"
model.push_to_hub(model_id)