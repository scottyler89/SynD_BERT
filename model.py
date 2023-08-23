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
        bert_output = outputs.last_hidden_state
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
bert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Create custom model
custom_model = CustomModel(bert_model=bert_model)

# Example input sequence
input_sequence = "ACTGACTG"

# Tokenize the input
inputs = tokenizer(input_sequence, return_tensors="pt", padding=True, truncation=True)

# Run the custom model
output = custom_model(**inputs)

# Output contains the single floating-point regression value
