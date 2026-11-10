# %% [markdown]
# ## Preparing Dataset for Model Training

# %%
import pandas as pd
import json
import torch
from onehotencoding import OneHotEncoder

# %%
# Load Datasets
train_data_path = '../data/clean/deeploc_train.csv'
val_data_path = '../data/clean/deeploc_val.csv'
test_data_path = '../data/clean/deeploc_test.csv'
aa_idx_path = '../configs/aa_to_idx.json'

train_df = pd.read_csv(train_data_path)
val_df = pd.read_csv(val_data_path)
test_df = pd.read_csv(test_data_path)
with open(aa_idx_path) as file:
    aa_to_idx = json.load(file)


# %%
train_df.head()

# %%
# Author: Dhruv

# Sequence to Tensor Encoding
def encode_sequences(seq, aa_to_idx):
    token_ids = [aa_to_idx[aa] for aa in seq]
    return torch.tensor(token_ids, dtype=torch.long)

# Encode all Sequencecs in a dataframe
def encode_split(df, aa_to_idx):
    tensors = []
    for seq, label in zip(df['sequence'], df['label']):
        tensor = encode_sequences(seq, aa_to_idx)
        tensors.append((tensor, label))
    return tensors

# Encode our 3 df splits
train_data = encode_split(train_df, aa_to_idx)
val_data = encode_split(val_df, aa_to_idx)
test_data = encode_split(test_df, aa_to_idx)

# %%
# Author: Dhruv

# Workflow: prepare batches by length --> pad batch by padding each tensor

def seq_length(tensor_label_pair):
    return len(tensor_label_pair[0])

def prepare_batches(train_data):
    train_batches = []
    BATCH_SIZE = 32
    sorted_train = sorted(train_data, key=seq_length)
    for i in range(0, len(sorted_train), BATCH_SIZE):
        train_batches.append(sorted_train[i:i+BATCH_SIZE])
    return train_batches

def pad_tensor(tensor, target_length, aa_to_idx):
    padded_tensor = tensor.tolist()
    
    if target_length < len(tensor):
        raise ValueError("target length shorter than seq length")
    
    pad_length = target_length - len(tensor)

    for _ in range(pad_length):
        padded_tensor.append(aa_to_idx['PAD'])

    return torch.tensor(padded_tensor, dtype=torch.long)

def pad_batch(train_batch, aa_to_idx):
    padded_batch = []
    target_length = seq_length(train_batch[-1]) # length of longest tensor
    for tensor, label in train_batch:
        padded_tensor = pad_tensor(tensor, target_length, aa_to_idx)
        padded_batch.append((padded_tensor, label))
    return padded_batch

# %%
# Author: AI (generated test)

train_batches = prepare_batches(train_data)
padded_batch = pad_batch(train_batches[0], aa_to_idx)

sequences, labels = zip(*padded_batch)
inputs = torch.stack(sequences)
targets = torch.tensor(labels, dtype=torch.long)

print(inputs.shape)
print(targets.shape)

# %%
one_hot_encoder = OneHotEncoder(aa_to_idx)
one_hot_inputs = one_hot_encoder.encode_batch(inputs)

# %%
one_hot_inputs.shape


