# %%
import torch
import torch.nn as nn

# %%
class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes=10,
        cnn_filters=128,
        kernel_sizes=(3, 5, 7),
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.3
    ):
        super().__init__()

        # --- CNN BLOCK ---
        convs = []
        for k in kernel_sizes:
            convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=22,       # one-hot depth
                        out_channels=cnn_filters,
                        kernel_size=k,
                        padding=k // 2        # keep same length
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        self.convs = nn.ModuleList(convs)

        # --- BiLSTM BLOCK ---
        self.bilstm = nn.LSTM(
            input_size=cnn_filters * len(kernel_sizes),
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # --- CLASSIFIER ---
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, 22)
        mask: (batch, seq_len) with 1 for real tokens, 0 for padding
        """

        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, 22, seq_len)

        # Apply each convolution
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))  # each: (batch, filters, seq_len)

        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch, filters*K, seq_len)

        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, 2*lstm_hidden)

        # Masked global max pooling
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            lstm_out = lstm_out * mask

        pooled, _ = torch.max(lstm_out, dim=1)  # (batch, 2*lstm_hidden)

        # Classifier
        logits = self.classifier(pooled)
        return logits



