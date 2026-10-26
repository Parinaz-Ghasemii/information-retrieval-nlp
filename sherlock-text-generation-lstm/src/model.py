import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    LSTM-based Encoder–Decoder (Seq2Seq) model.
    Encoder processes src sequence and produces context (hidden, cell).
    Decoder generates outputs conditioned on encoder context.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        # 1. Shared embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )

        # 3. Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )

        # 4. Output projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        """
        src: (B, src_len)
        trg: (B, trg_len)

        Returns:
            logits: (B, trg_len, vocab_size)
            (hidden, cell): final decoder states
        """

        # ---- Encoder ----
        # Embed source
        src_embedded = self.embedding(src)  # (B, src_len, emb_dim)

        # Encode
        _, (hidden, cell) = self.encoder_lstm(src_embedded)
        # hidden, cell: (num_layers, B, hidden_size)

        # ---- Decoder ----
        # Embed target
        trg_embedded = self.embedding(trg)  # (B, trg_len, emb_dim)

        # Decode using encoder context as initial state
        decoder_outputs, (hidden, cell) = self.decoder_lstm(
            trg_embedded, (hidden, cell)
        )
        # decoder_outputs: (B, trg_len, hidden_size)

        # ---- Projection ----
        logits = self.fc_out(decoder_outputs)
        # logits: (B, trg_len, vocab_size)

        return logits, (hidden, cell)
