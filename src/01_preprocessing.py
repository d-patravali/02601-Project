# %%
from Bio import SeqIO
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths configured by AI
project_root = Path('..')
file_path = project_root / 'data' / 'raw' / 'deeploc_data.fasta'
aa_to_idx_path = project_root / 'configs' / 'aa_to_idx.json'
label_to_idx_path = project_root / 'configs' / 'label_to_idx.json'
train_data_path = project_root / 'data' / 'clean' / 'deeploc_train.csv'
val_data_path = project_root / 'data' / 'clean' / 'deeploc_val.csv'
test_data_path = project_root / 'data' / 'clean' / 'deeploc_test.csv'

# %%
# Author: Dhruv

proteins = []
file_type = 'fasta'
for protein in SeqIO.parse(file_path, file_type):
    protein_seq = str(protein.seq)

    # header
    header = protein.description

    proteins.append({
        "protein_id": protein.id,
        "sequence": protein_seq,
        "header": header
    })

df = pd.DataFrame(proteins)
print(df.head())

# %%
# Author: Dhruv
def parse_header(header):
    '''
    Extract the subcellular location (output label), membrane_status, and train/test label from the header
    Args:
        header(str): header for row
    Returns:
        Tuple of subcellular loc, membrane status, and train test label

    '''
    header_parts = header.split()

    subcellular_location = header_parts[1][:-2]
    membrane_status = header_parts[1][-1]

    if len(header_parts) > 2:
        train_test = header_parts[2]
    else:
        train_test = 'train'
    
    return (subcellular_location, membrane_status, train_test)

new_cols = df['header'].apply(parse_header).apply(pd.Series)
df[['subcellular_location', 'membrane_status', 'train_test']] = new_cols

# drop headers column: info has been parsed into new cols
df.drop(columns=['header'], inplace=True)

print(df.head())

# %%
df['train_test'].value_counts()

# %%
# check missing values
print(df.isnull().sum())
# check duplicates
print("Duplicate IDs:", df['protein_id'].duplicated().sum())
print("Duplicate sequences:", df['sequence'].duplicated().sum())

# %%
# Author: Dhruv

# check unusual amino acids
usual_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
unusual_amino_acids = set("XBZUOJ")

# check if a sequence contains unusual amino acids
def count_unusual(seq):
    count = 0
    for aa in seq:
        if aa in unusual_amino_acids:
            count += 1
    return count

df['num_unusual_aa'] = df['sequence'].apply(count_unusual)
print((df['num_unusual_aa'] > 0).sum())

# %%
# Author: Dhruv

unusual_rows = df[df['num_unusual_aa'] > 0]

counts = Counter()

for seq in unusual_rows['sequence']:
    for aa in seq:
        if aa in unusual_amino_acids:
            counts[aa] += 1

print(counts)

# %%
# Author: Dhruv

# standard amino acids
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {'PAD': 0}

for i, aa in enumerate(amino_acids):
    aa_to_idx[aa] = i + 1

# unknown amino acids
aa_to_idx['UNK'] = 21

for aa in "XBZUOJ":
    aa_to_idx[aa] = aa_to_idx["UNK"]

# %%
# Author: Dhruv

# check variation in sequence lengths
df['seq_length'] = df['sequence'].str.len()
print(df['seq_length'].describe())

# %%
# Author: Dhruv

# inspect duplicates
duplicates = df[df.duplicated('sequence', keep=False)]
duplicates = duplicates.sort_values('sequence')
print(f"duplicate labels: {duplicates['train_test'].value_counts()}")

print(duplicates[['protein_id', 'subcellular_location', 'membrane_status', 'train_test']])

cols = ['subcellular_location', 'membrane_status']

# check which duplicates have mismatch data: potential problem
conflicts = df.groupby('sequence')[cols].nunique()
conflicting_sequences = conflicts[(conflicts > 1).any(axis=1)]

# retreive df rows of duplicates
conflicting_rows = df[df['sequence'].isin(conflicting_sequences.index)]

print(conflicting_rows[['protein_id', 'subcellular_location', 'membrane_status', 'train_test']])

# %%
# Author: Dhruv

# separate train and test dataframes
train_df = df[df['train_test'] == 'train'].copy()
test_df = df[df['train_test'] == 'test'].copy()

# remove duplicates from training data
train_df = train_df.drop_duplicates(subset='sequence')

# train and test dataset sizes
print("Training set size:", len(train_df))
print("Test set size:", len(test_df))
train_percent = 100 * len(train_df) / len(df)
test_percent = 100 * len(test_df) / len(df)
print("Train/Test Split:", train_percent, ':', test_percent)

# %%
train_df.head()

# %%
test_df.head()

# %%
# inspect output class balance

print(train_df["subcellular_location"].value_counts())
print(train_df["subcellular_location"].value_counts(normalize=True))

# %%
# Author: Dhruv

# split the training data into true training set and validation set, leaving the test set untouched

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["subcellular_location"]
)

# %%
# Author: AI

# check duplicates within each dataset
print("Train duplicates:", train_df['sequence'].duplicated().sum())
print("Validation duplicates:", val_df['sequence'].duplicated().sum())
print("Test duplicates:", test_df['sequence'].duplicated().sum())

# check sequence overlap between datasets
train_seqs = set(train_df['sequence'])
val_seqs = set(val_df['sequence'])
test_seqs = set(test_df['sequence'])

print("Train/Val overlap:", len(train_seqs & val_seqs))
print("Train/Test overlap:", len(train_seqs & test_seqs))
print("Val/Test overlap:", len(val_seqs & test_seqs))

# %%
# Author: AI

# check duplicates in test set
test_duplicates = test_df[test_df['sequence'].duplicated(keep=False)]
print(test_duplicates[['protein_id', 'subcellular_location', 'membrane_status']])

conflicts = test_df.groupby('sequence')[['subcellular_location', 'membrane_status']].nunique()
print(conflicts[(conflicts > 1).any(axis=1)])

# don't want to change test set at all because it should be the same as DeepLoc's test set which we will use as a bench mark

# %%
# check lengths
print(len(train_df))
print(len(val_df))
print(len(test_df))

# %%
# summarize sequence lengths by localization
length_stats = train_df.groupby('subcellular_location')['seq_length'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])

print(length_stats.sort_values('median'))

# %%
# Author: AI

# compare output class proportions of each dataset split 

train_props = train_df['subcellular_location'].value_counts(normalize=True)
val_props = val_df['subcellular_location'].value_counts(normalize=True)
test_props = test_df['subcellular_location'].value_counts(normalize=True)

props_df = pd.DataFrame({
    'train': train_props,
    'val': val_props,
    'test': test_props
}).fillna(0)

props_df = props_df.sort_index()

props_df.plot(kind='bar', figsize=(12, 6))
plt.xlabel('Subcellular Location')
plt.ylabel('Proportion')
plt.title('Class Proportion Across Train, Validation, and Test Sets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
test_df['subcellular_location'].value_counts().get('Cytoplasm-Nucleus', 0)

# %%
# Author: Dhruv

# omit Cytoplasm-Nucleus from training and validation data since it is not included in the predetermined test set

train_df = train_df[train_df['subcellular_location'] != 'Cytoplasm-Nucleus'].copy()

val_df = val_df[val_df['subcellular_location'] != 'Cytoplasm-Nucleus'].copy()

# %%
# Author: Dhruv

# subcellular localization labels
classes = sorted(train_df['subcellular_location'].unique())

label_to_idx = {}

for i in range(len(classes)):
    label_to_idx[classes[i]] = i

train_df['label'] = train_df['subcellular_location'].map(label_to_idx)
val_df['label'] = val_df['subcellular_location'].map(label_to_idx)
test_df['label'] = test_df['subcellular_location'].map(label_to_idx)

# check labels were encoded
print(train_df['label'].isnull().sum())
print(val_df['label'].isnull().sum())
print(test_df['label'].isnull().sum())

# %%
# Author: Dhruv

with open(aa_to_idx_path, 'w') as f:
    json.dump(aa_to_idx, f, indent=4)

with open(label_to_idx_path, 'w') as f:
    json.dump(label_to_idx, f, indent=4)

# %%
train_df.to_csv(train_data_path, index=False)
val_df.to_csv(val_data_path, index=False)
test_df.to_csv(test_data_path, index=False)


