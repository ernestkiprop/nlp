# Install required libraries
!pip install --upgrade pip
!pip install --upgrade typing_extensions
!pip install wandb
!pip install transformers datasets wandb evaluate pynvml scipy scikit-learn transformers[torch] accelerate>=0.26.0
!pip install --upgrade transformers


import os
os._exit(00) 
import wandb, os
###############################

os.environ["WANDB_API_KEY"] = "b3c53303f76ce37498666700f945f85807f5db44"
wandb.login()

!pip install transformers datasets wandb evaluate pynvml

# Import necessary libraries
import wandb
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
import numpy as np
import evaluate
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# Initialize wandb
wandb.login()

# Function to log GPU memory usage
def log_gpu_memory():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # GPU index 0
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = {
            "memory_used_MB": info.used // 1024**2,
            "memory_free_MB": (info.total - info.used) // 1024**2,
            "memory_total_MB": info.total // 1024**2,
        }
        wandb.log(gpu_memory)
    except Exception as e:
        print(f"Error logging GPU memory: {e}")

# Custom callback to log memory and time during training
class MemoryTimeCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_init_end(self, args, state, control, **kwargs):
        print("Trainer initialization complete.")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("Training started.")

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time  # Time since training began

        # Log GPU memory usage if a GPU is available
        if torch.cuda.is_available():
            log_gpu_memory()

        # Log elapsed time per step
        wandb.log({"train/step_time_seconds": elapsed_time}, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        print("Training completed.")

# Mapping of GLUE tasks to their input keys
task_to_keys = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# List of all GLUE tasks (excluding 'ax' since it's test-only)
glue_tasks = list(task_to_keys.keys())

# Loop through each task in the GLUE benchmark
for task in glue_tasks:
    print(f"Running GLUE task: {task}")
    
    # Load dataset for the current task
    dataset = load_dataset("glue", task)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Get the appropriate keys for the current task
    sentence1_key, sentence2_key = task_to_keys[task]
    
    # Tokenize the dataset dynamically based on the task's input keys
    def tokenize_function(examples):
        if sentence2_key is None:  # Single-sentence tasks (e.g., cola, sst2)
            return tokenizer(examples[sentence1_key], truncation=True)
        else:  # Sentence-pair tasks (e.g., mrpc, qqp)
            return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch tensors
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Load pre-trained model for sequence classification
    num_labels = len(dataset["train"].features["label"].names) if task != "stsb" else 1  # Regression for STS-B
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    
    # Define evaluation metric(s)
    metric = evaluate.load("glue", task)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task == "stsb":  # For STS-B regression, use Pearson/Spearman correlation
            predictions = logits[:, 0]
        else:  # For classification tasks, use argmax for predictions
            predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Data collator to dynamically pad batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define training arguments with wandb integration and logging steps for memory/time tracking
    training_args = TrainingArguments(
        output_dir=f"./{task}-results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="wandb",
        run_name=f"bert-{task}"
    )
    
    # Instantiate the custom callback for memory/time logging
    memory_time_callback = MemoryTimeCallback()
    
    # Initialize Trainer object with the custom callback integrated
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[memory_time_callback],
    )
    
    # Start training (metrics are automatically logged to wandb)
    trainer.train()

# Finish the wandb run when all tasks are done (optional)
wandb.finish()
