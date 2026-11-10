import numpy as np
import pandas as pd
import torch

"""
One-hot encoding for protein sequences.

Author: Josh & AI
"""

class OneHotEncoder:
    """
    Class to perform one-hot encoding of protein sequences.

    Author: AI
    """
    def __init__(self, aa_to_idx):
            self.aa_to_index = aa_to_idx
            self.num_channels = max(aa_to_idx.values()) + 1 # multiple letter share UNK id

    def encode(self, seq: str) -> np.ndarray:
        one_hot = np.zeros((len(seq), self.num_channels), dtype=np.float32)

        for i, aa in enumerate(seq):
            if aa in self.aa_to_index:
                one_hot[i, self.aa_to_index[aa]] = 1.0
            elif aa in ("X", "U", "B", "Z", "O", "J"):
                one_hot[i, self.aa_to_index['UNK']] = 1.0
            else:
                one_hot[i, self.aa_to_index['UNK']] = 1.0

        return one_hot

    def encode_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        # Author: Dhruv
        encoded_batch = []

        for seq in inputs.tolist():
            encoded_seq = []

            for aa_id in seq:
                vector = [0.0] * self.num_channels
                vector[aa_id] = 1.0
                encoded_seq.append(vector)

            encoded_batch.append(encoded_seq)

        return torch.tensor(encoded_batch, dtype=torch.float32)


def one_hot_encoding(sequence: str, aa_to_idx) -> np.ndarray:
    """
    Perform one-hot encoding on a CSV file containing protein sequences.
    
    Parameters:
    - filepath: str, path to the input CSV file. The file should contain a column named "sequence".

    Author: Josh
    """
    encoder = OneHotEncoder(aa_to_idx)
    return encoder.encode(sequence)
