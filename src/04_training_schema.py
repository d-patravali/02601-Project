# %% [markdown]
# # **Training Schema Implementation**
# 
# Author: Josh Chen + AI

# %% [markdown]
# ### **1st training schema: class-balanced curriculum learning**
# 
# Summary: exposing models to different proportions of classes at different times.\
# Phase 1: balanced distribution of classes (equal # of classes)\
# Phase 2: mixed distribution between balanced and real distribution\
# Phase 3: real distribution of classes

# %%
from collections import defaultdict
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# %%
def build_class_index(df, label_col="localization"):
    """
    Build a dictionary, mapping class labels to their indices in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - label_col: str, the name of the column containing class labels.

    Returns:
    - class_to_indices: dict, mapping each class label to a list of indices where that label occurs in the DataFrame.

    Author: Josh
    """

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(df[label_col]):
        class_to_indices[label].append(idx)
    return class_to_indices

# %%
df = pd.read_csv(r"C:\Users\josh2\OneDrive\Desktop\CMU Fall 26\02-601 Programming for Scientists\02601-Project\deeploc_data_onehot.csv")
print(df['subcellular_location'].value_counts())

class_to_indices = build_class_index(df, label_col="subcellular_location")
print({k: len(v) for k, v in class_to_indices.items()})

# %%
class CurriculumSchedule:
    """
    Class to manage the curriculum schedule for training

    Author: Josh
    """
    def __init__(self, total_epochs: int, phase1_frac=0.33, phase2_frac=0.33):
        """
        Initialize the curriculum schedule. 

        Parameters:
        - total_epochs: int, total number of epochs for training.
        - phase1_frac: float, fraction of total epochs for phase 1.
        - phase2_frac: float, fraction of total epochs for phase 2.

        Author: Josh
        """

        self.total_epochs = total_epochs
        self.phase1_end = int(total_epochs * phase1_frac)
        self.phase2_end = int(total_epochs * (phase1_frac + phase2_frac))

    def get_phase(self, epoch: int):
        """
        Returns the phase of the curriculum based on current epoch.

        Parameters:
        - epoch: int, current epoch number.

        Returns:
        - int, phase number (1, 2, or 3).

        Author: Josh
        """
        if epoch < self.phase1_end:
            return 1
        elif epoch < self.phase2_end:
            return 2
        else:
            return 3

# %%
class ClassSampler:
    """
    Class to sample batches of data according to the curriculum schedule.

    Author: Josh + AI
    """

    def __init__(self, class_to_indices: dict, curriculum_schedule: CurriculumSchedule, batch_size: int, mix_alpha=0.5):
        """
        Initializes the ClassSampler.

        Parameters:
        - class_to_indices: dict, mapping class labels to their indices in the dataset.
        - curriculum_schedule: CurriculumSchedule object, defines the phases of training.
        - batch_size: int, number of samples per batch.
        """
        self.class_to_indices = class_to_indices
        self.curriculum_schedule = curriculum_schedule
        self.classes = list(class_to_indices.keys())
        self.batch_size = batch_size
        self.mix_alpha = mix_alpha

        # Compute class probabilities
        counts = {cls: len(idxs) for cls, idxs in class_to_indices.items()}
        total = sum(counts.values())
        self.class_probs = {cls: counts[cls] / total for cls in counts}

    def sample(self, epoch: int):
        """
        Sample a batch of data. Approach (balanced, mixed, or realistic) depends on the current phase of training.

        Parameters:
        - epoch: int, current epoch number.

        Returns:
        - list, indices of the sampled data points.
        """
        phase = self.curriculum_schedule.get_phase(epoch)

        if phase == 1:
            return self.sample_balanced()

        elif phase == 2:
            return self.sample_mixed()

        else:
            return self.sample_realistic()

    
    def sample_balanced(self):
        """
        Phase 1: Balanced sampling. Each class is sampled equally, with oversampling for rare classes.

        Returns:
        - list, indices of the sampled data points.
        """
        num_classes = len(self.classes)
        per_class = max(1, self.batch_size // num_classes)

        batch_indices = []
        for cls in self.classes:
            indices = self.class_to_indices[cls]
            chosen = random.choices(indices, k=per_class)
            batch_indices.extend(chosen)

        # if batch size smaller than desired
        if len(batch_indices) < self.batch_size:
            difference = self.batch_size - len(batch_indices)
            all_indices = [i for idxs in self.class_to_indices.values() for i in idxs]
            filler = random.choices(all_indices, k=difference)
            batch_indices.extend(filler)

        # if batch size larger than desired
        if len(batch_indices) > self.batch_size:
            batch_indices = batch_indices[:self.batch_size]

        random.shuffle(batch_indices)
        return batch_indices


    def sample_mixed(self):
        """
        Phase 2: Mixed sampling. A mix of balanced and realistic sampling.
        
        Returns:
        - list, indices of the sampled data points.
        """

        k_balanced = int(self.batch_size * self.mix_alpha)
        k_realistic = self.batch_size - k_balanced

        balanced_indices = self.sample_balanced()[:k_balanced]

        realistic_indices = self.sample_realistic()[:k_realistic]

        batch_indices = balanced_indices + realistic_indices
        random.shuffle(batch_indices)
        return batch_indices


    def sample_realistic(self):
        """
        Phase 3: Realistic sampling. Samples according to the actual class distribution in the dataset.

        Returns:
        - list, indices of the sampled data points.
        """
        batch_indices = []

        probs = [self.class_probs[c] for c in self.classes]

        for _ in range(self.batch_size):
            cls = random.choices(self.classes, weights=probs)[0]
            idx = random.choice(self.class_to_indices[cls])
            batch_indices.append(idx)

        return batch_indices


# %%
sampler = ClassSampler(class_to_indices, CurriculumSchedule(total_epochs=100), batch_size=32)
print(sampler.sample(epoch=10))  # Phase 1
print("size of batch:", len(sampler.sample(epoch=10)))  # Phase 1
print(sampler.sample(epoch=40))  # Phase 2
print("size of batch:", len(sampler.sample(epoch=40)))  # Phase 2
print(sampler.sample(epoch=80))  # Phase 3
print("size of batch:", len(sampler.sample(epoch=80)))  # Phase 3

# %% [markdown]
# ### **2nd training schema: Dynamic Class Re-weighting**
# 
# Summary: changing class weights during training to focus on classes the model is currently bad at evaluating.\
# Accuracy and loss are tracked, weights for classes with low accuracy and high loss are increased.\
# Initial class weights: uniform\
# How often to update weights: every epoch\
# Which metric to use when updating weights: per-class loss (classes w/ higher loss get higher weight)\
# Number of Epochs: 20 epochs (~20k sequences)\
# Early stopping: yes
# 

# %%
class DynamicClassReweightingSchema:
    def __init__(self, num_classes: int, device="cpu", update_every_epochs=1, epsilon=1e-3):
        self.num_classes = num_classes
        self.device = device
        self.update_every_epochs = update_every_epochs
        self.epsilon = epsilon

        # Start with uniform weights
        self.class_weights = torch.ones(num_classes, device=device)

        # Metrics
        self.reset_metrics()

        # Loss function (will be updated)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def reset_metrics(self):
        self.class_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.class_loss_sum = np.zeros(self.num_classes, dtype=np.float64)

    def update_metrics(self, logits, targets, loss):
        targets_np = targets.detach().cpu().numpy()
        batch_loss = loss.detach().cpu().item()

        # Add loss to each target class (approximate)
        for t in targets_np:
            self.class_counts[t] += 1
            self.class_loss_sum[t] += batch_loss / len(targets_np)

    def compute_class_metrics(self):
        counts = np.maximum(self.class_counts, 1)
        avg_losses = self.class_loss_sum / counts
        return avg_losses

    def update_class_weights(self, epoch):
        if (epoch + 1) % self.update_every_epochs != 0:
            return

        avg_losses = self.compute_class_metrics()

        # Higher loss → higher weight
        raw_weights = avg_losses + self.epsilon

        # Normalize
        raw_weights = raw_weights / raw_weights.sum()

        # Convert to tensor
        self.class_weights = torch.tensor(raw_weights, dtype=torch.float32, device=self.device)

        # Update loss function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Reset metrics
        self.reset_metrics()

    def compute_loss(self, logits, targets):
        loss = self.criterion(logits, targets)
        self.update_metrics(logits, targets, loss)
        return loss



