from datasets import load_dataset
from transformers import OPTForCausalLM, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from transformers import default_data_collator
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


import torch
import torch.nn.functional as F
import math

block_size = 256

def preprocess_function(examples):
    results = tokenizer(examples["text"], truncation=True)
    return results


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    results = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    results["labels"] = results["input_ids"][:]
    for key, val in results.items():
        results[key] = val[:-1]
    return results


lambada = load_dataset("wikitext", "wikitext-2-v1")
#lambada = load_dataset("allenai/c4", "en", split="train")

model_name = "facebook/opt-1.3b"
#tokenizer = GPT2Tokenizer.from_pretrained(model_name,model_max_length=256)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = OPTForCausalLM.from_pretrained(model_name)

lm_dataset = lambada.map(
    preprocess_function,
    batched=True,
    remove_columns=lambada["train"].column_names,
)
lm_dataset = lm_dataset.map(
    group_texts,
    batched=True,
)

loader = DataLoader(lm_dataset['train'], collate_fn=data_collator, batch_size=8)
for data in loader:
    continue


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



def ce_loss(logits, labels):  # Default to square root (k=2)
    # Calculate the cross entropy loss
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return ce_loss

def root_loss(output, target, k, m):
    # Calculate the RLO loss
    n = target.shape[0]
    prob = F.softmax(output, dim=1)
    root = torch.pow((prob[range(n), target]), 1 / k)
    root = m * (1 - root)
    loss = torch.mean(root)
    return loss



class RLOTrainer(Trainer):
    def __init__(self, *args, k=5, m=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.m = m

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        _, _, vocab_size = shift_logits.size()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Compute custom loss
        loss = root_loss(shift_logits, shift_labels, k=5, m=5)
        #loss = ce_loss(shift_logits, shift_labels)
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir='opt_1.3b_wiki_k5',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,  # Reduced learning rate
    warmup_steps=500,  # Increased warmup steps
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    gradient_accumulation_steps=2,
    fp16=True,  # Enable mixed precision training
    seed=42,
)



# Initialize the standard trainer with custom loss function
trainer = RLOTrainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset['train'],
    eval_dataset=lm_dataset['validation'],
    #compute_loss=compute_loss
)

# Start training
trainer.train()
eval_results = trainer.evaluate()

# Standard perplexity calculation based on cross-entropy loss
eval_loss = eval_results['eval_loss']
perplexity = math.exp(eval_loss)

# Print evaluation results including standard perplexity
print(f"Eval_loss: {eval_loss}")
print(f"Perplexity: {perplexity:.2f}")

trainer.push_to_hub()
from huggingface_hub import login

login('')
repo_name = ""
add_to_git_credential=True
tokenizer.push_to_hub(repo_name)
