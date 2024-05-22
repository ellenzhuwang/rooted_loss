from transformers import OPTForCausalLM, OPTConfig

# Load the configuration and model
config = OPTConfig.from_pretrained('ellen625/opt125_wiki_ce')
model = OPTForCausalLM.from_pretrained('ellen625/opt125_wiki_ce', config=config)

# Set the model to evaluation mode
model.eval()

# Print the data types of each parameter
for name, param in model.named_parameters():
    print(f"{name} has dtype {param.dtype}")

import torch
import torch.nn.functional as F

# Create dummy data
input = torch.randn(1, 10).float()  # Make sure it is float32
weight = torch.randn(10, 5).float()
bias = torch.randn(5).float()

# Apply linear
output = F.linear(input, weight, bias)
print("Output:", output)

