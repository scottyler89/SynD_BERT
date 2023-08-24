import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoTokenizer, AutoModel

# Transformer Regressor
class TransformerRegressor(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=8, dim_feedforward=2048):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1) # Output a single value

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.fc(output.mean(dim=1))
        return output

# Custom Model combining BERT with Transformer Regressor
class CustomModel(nn.Module):
    def __init__(self, bert_model, max_length=512):
        super().__init__()
        self.bert_model = bert_model
        self.max_length = max_length
        self.regressor = TransformerRegressor() # 8-layer transformer regressor

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        #print(outputs)
        # Returns a tuple of 2 tensors, but the first is the hidden state vects
        bert_output = outputs[0]
        padded_output = self.pad_to_fixed_size(bert_output)
        regression_output = self.regressor(padded_output)
        return regression_output

    def pad_to_fixed_size(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.size()
        padded_states = torch.zeros((batch_size, self.max_length, 768), device=hidden_states.device)
        padded_states[:, :seq_length, :] = hidden_states
        return padded_states

# Instantiate DNABERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
bert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)

# Create custom model
custom_model = CustomModel(bert_model=bert_model).to(device)






#############################
import random

def generate_sequence(length=100):
    return ''.join(random.choice('ACTG') for _ in range(length))

def percentage_of_As(sequence):
    return sequence.count('A') / len(sequence)

# Generate synthetic dataset
sequences = [generate_sequence() for _ in range(1000)]
targets = [percentage_of_As(seq) for seq in sequences]

from torch.utils.data import DataLoader, TensorDataset

# Tokenize sequences
inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=False).to(device)

# Prepare DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(targets))
loader = DataLoader(dataset, batch_size=32, shuffle=True)
###############################

from torch.optim import Adam

# Loss and optimizer
loss_function = nn.MSELoss()
optimizer = Adam(custom_model.parameters(), lr=0.001)

# Training loop
for epoch in range(10): # Adjust the number of epochs as needed
    for input_ids, attention_mask, target in loader:
        custom_model.zero_grad()
        output = custom_model(input_ids, attention_mask).squeeze()
        loss = loss_function(output, target.float())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")
