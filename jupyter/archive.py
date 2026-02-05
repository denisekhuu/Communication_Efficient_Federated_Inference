import torch
import torch.nn as nn

import torch

import torch

def generate_unique_row_binary_tensor(n):
    tensor = torch.randint(0, 2, (n, n))

    # Keep checking for duplicates until all rows are unique
    while True:
        # Convert rows to tuples for easy uniqueness check
        rows = [tuple(row.tolist()) for row in tensor]

        # Find duplicates by counting occurrences
        duplicates = [row for row in set(rows) if rows.count(row) > 1]

        if not duplicates:
            break  # all rows unique

        # For each duplicate, randomly pick one occurrence to change
        for dup in duplicates:
            indices = [i for i, row in enumerate(rows) if row == dup]
            # Keep one occurrence intact, change others
            for idx in indices[1:]:
                tensor[idx] = torch.randint(0, 2, (n,))

    return tensor


class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length = 4,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.row_drop_prob = 0.3
        self.context_length = context_length
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        if self.training and self.row_drop_prob > 0.0 and num_tokens == self.context_length:
            self.mask = generate_unique_row_binary_tensor(self.context_length)
            print(self.mask)
        else:
            self.mask = torch.zeros(self.context_length, self.context_length)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) if self.training else attn_weights

        context_vec = attn_weights @ values
        context_vec = torch.nan_to_num(context_vec, nan=0.0)
        return context_vec
    
import torch 
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2
torch.manual_seed(123)

context_length = batch.shape[1]
ca = SelfAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)