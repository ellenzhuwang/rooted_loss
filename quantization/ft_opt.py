from datasets import load_dataset
from transformers import OPTForCausalLM, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from transformers import default_data_collator
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

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


# training_args = TrainingArguments(
#     output_dir="QuIP/opt125_wiki_rlo_k50",
#     num_train_epochs=1,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=100,
#     evaluation_strategy="epoch"
# )

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=lm_dataset["train"],
#    eval_dataset=lm_dataset["test"],
#    data_collator=data_collator,
#)

import torch
import torch.nn.functional as F

# def root_loss(outputs, labels, k, m):
#     """
#     Compute the root loss for sequence outputs.
    
#     Args:
#         outputs (torch.Tensor): The logits from the model of shape (batch_size, sequence_length, num_classes).
#         labels (torch.Tensor): The ground-truth labels of shape (batch_size, sequence_length).
#         k (float): The exponent for the root calculation.
#         m (float): The scaling factor for the loss adjustment.
        
#     Returns:
#         torch.Tensor: The mean root loss calculated over all elements in the batch and sequence.
#     """
#     batch_size, sequence_length, num_classes = outputs.shape
#     prob = F.softmax(outputs, dim=2)  # Apply softmax across the class dimension

#     # Create a new range tensor for indexing: [batch_size, sequence_length]
#     batch_range = torch.arange(batch_size).unsqueeze(1).expand(batch_size, sequence_length)
#     sequence_range = torch.arange(sequence_length).expand(batch_size, sequence_length)

#     # Index to gather the probabilities corresponding to the labels at each sequence position
#     selected_prob = prob[batch_range, sequence_range, labels]
    
#     # Compute the root loss
#     root = torch.pow(selected_prob, 1/k)
#     root = m * (1 - root)
#     loss = torch.mean(root)
#     return loss

# def rooted_loss(logits, labels, k=1):
#     # Flatten logits and labels
#     logits = logits.view(-1, logits.size(-1))
#     labels = labels.view(-1)
    
#     # Ensure labels are within the valid range
#     labels = torch.clamp(labels, 0, logits.size(-1) - 1)
    
#     # Convert labels to one-hot encoding
#     one_hot_labels = F.one_hot(labels, num_classes=logits.size(-1)).float()
    
#     # Compute the linear combination of weights and inputs
#     xw = torch.sum(logits * one_hot_labels, dim=1)
    
#     # Compute the custom loss
#     exp_term = torch.exp(-xw)
#     loss = k * (1 + exp_term) ** (1 / k)
#     return loss.mean()

def ce_loss(logits, labels):  # Default to square root (k=2)
    # Calculate the cross entropy loss
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    # Apply the k-th root to the cross entropy loss
    #rooted_ce_loss = ce_loss ** (1 / k)
    #return rooted_ce_loss
    return ce_loss

def root_loss(output, target, k, m):

    n = target.shape[0]
    prob = F.softmax(output, dim=1)
    root = torch.pow((prob[range(n), target]), 1 / k)
    root = m * (1 - root)
    loss = torch.mean(root)
    return loss


from transformers import Trainer, TrainingArguments

# class RLOTrainer(Trainer):
#     def __init__(self, *args, k=10, m=10, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.k = k
#         self.m = m

#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         Custom method to compute the loss using the `root_loss` function.

#         Args:
#             model: the model being trained.
#             inputs: dictionary of inputs to the model. This must include 'labels'.
#             return_outputs: if True, this method returns a tuple of (loss, outputs).
        
#         Returns:
#             loss or (loss, outputs) depending on the value of `return_outputs`.
#         """
#         labels = inputs.pop('labels')
#         outputs = model(**inputs)
#         logits = outputs.get('logits')
#         loss = root_loss(logits, labels, self.k, self.m)
#         return (loss, outputs) if return_outputs else loss
    
# class RLOTrainer(Trainer):
#     def __init__(self, *args, root_loss_func=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.root_loss_func = root_loss_func

#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop('labels')
#         outputs = model(**inputs)
#         logits = outputs.get('logits')
#         if model.training:  # Correctly check if it's the training phase
#             loss = self.root_loss_func(logits, labels, k=50, m=100
#                                        ) if self.root_loss_func else outputs.loss
#         else:
#             # Use standard cross-entropy loss for evaluation
#             loss_fct = torch.nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

#         return (loss, outputs) if return_outputs else loss

class RLOTrainer(Trainer):
    def __init__(self, *args, k=5, m=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.m = m

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.get("labels")
    #     # Forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # Compute custom loss
    #     loss = rooted_loss(logits, labels, self.k)
    #     return (loss, outputs) if return_outputs else loss

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


from transformers import Trainer
import torch


# rlo_trainer = RLOTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=lm_dataset["train"],
#     eval_dataset=lm_dataset["test"],
#     data_collator=data_collator,
# )


# rlo_trainer.train()

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
#trainer.push_to_hub()

import math

# Standard perplexity calculation based on cross-entropy loss
eval_loss = eval_results['eval_loss']
perplexity = math.exp(eval_loss)

# Print evaluation results including standard perplexity
print(f"Eval_loss: {eval_loss}")
print(f"Perplexity: {perplexity:.2f}")

trainer.push_to_hub()
from huggingface_hub import login

login('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')
repo_name = "ellen625/opt_1.3b_wiki_k5"
add_to_git_credential=True
tokenizer.push_to_hub(repo_name)
