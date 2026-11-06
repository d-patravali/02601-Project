import numpy as np
import pandas as pd

"""
One-hot encoding for protein sequences.

Author: Josh & AI
"""

class OneHotEncoder:
    """
    Class to perform one-hot encoding of protein sequences.

    Author: AI
    """
    def __init__(self):
        # 20 standard amino acids
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

        # Special categories
        self.special_unusual = "UNUSUAL"   # X, U, B, Z
        self.special_padding = "PADDING"   # anything else is counted as padding

        # Build index mapping
        self.aa_to_index = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.aa_to_index[self.special_unusual] = len(self.aa_to_index)
        self.aa_to_index[self.special_padding] = len(self.aa_to_index)

        self.num_channels = len(self.aa_to_index)  # 22 total

    def encode(self, seq: str) -> np.ndarray:
        one_hot = np.zeros((len(seq), self.num_channels), dtype=np.float32)

        for i, aa in enumerate(seq):
            if aa in self.aa_to_index:
                one_hot[i, self.aa_to_index[aa]] = 1.0
            elif aa in ("X", "U", "B", "Z"):
                one_hot[i, self.aa_to_index[self.special_unusual]] = 1.0
            else:
                one_hot[i, self.aa_to_index[self.special_padding]] = 1.0

        return one_hot


def one_hot_encoding(filepath: str) -> None:
    """
    Perform one-hot encoding on a CSV file containing protein sequences.
    
    Parameters:
    - filepath: str, path to the input CSV file. The file should contain a column named "sequence".

    Author: Josh
    """
    encoder = OneHotEncoder()
    df = pd.read_csv(filepath)
    df["one_hot"] = df["sequence"].apply(encoder.encode)
    df.to_csv("deeploc_data_onehot.csv", index=False)
