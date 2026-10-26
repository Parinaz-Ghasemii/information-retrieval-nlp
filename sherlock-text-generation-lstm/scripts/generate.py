import sys
import os
import re
import json
import torch
import torch.nn.functional as F
from typing import List, Tuple

# Fix Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import ProjectConfig
from src.model import EncoderDecoder
from src import utils
class TextGenerator:
    """Encapsulates the logic for generating text with the Encoder-Decoder model."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load paths consistent with ProjectConfig
        self.experiment_dir = utils.get_experiment_dir(config)
        model_path = "/content/sherlock_project/checkpoints/best_model.pth"

        # Load Vocabulary
        with open(utils.VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.token_to_id = vocab_data["token_to_id"]
            self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> EncoderDecoder:
        model = EncoderDecoder(
            vocab_size=self.config.data.max_vocab_size,
            embedding_dim=self.config.model.embedding_dim,
            hidden_size=self.config.model.lstm_units,
            num_layers=self.config.model.num_layers,
            dropout_prob=self.config.model.dropout_rate,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _clean_prompt(self, text: str) -> List[int]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\,\!\?\;\:\'\-]", " ", text)
        tokens = text.split()
        return [self.token_to_id.get(t, self.token_to_id.get("<UNK>", 3)) for t in tokens]

    def generate(
        self,
        start_prompt: str,
        max_length: int = 50,
        strategy: str = 'sample',
        temperature: float = 0.7,
        repetition_penalty: float = 2, # Higher value = stronger repetition prevention
        top_k: int = 20                 # Limits options to most likely words
    ):
        """Generates text using the encoder-decoder hidden state."""
        with torch.no_grad():
            # 1. Encode context
            input_ids = self._clean_prompt(start_prompt)
            src_tensor = torch.tensor([input_ids], device=self.device)
            src_embedded = self.model.embedding(src_tensor)
            _, (h, c) = self.model.encoder_lstm(src_embedded)

            generated_words = []
            tokens_seen = []

            # Start with <SOS>
            curr_token = torch.tensor([[self.token_to_id.get("<SOS>", 1)]], device=self.device)
            state = (h, c)

            for _ in range(max_length):
                embedded = self.model.embedding(curr_token)
                output, state = self.model.decoder_lstm(embedded, state)
                logits = self.model.fc_out(output).squeeze(1) # (1, vocab_size)

                # --- 🛑 Apply Repetition Penalty ---
                for token_id in set(tokens_seen):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

                if strategy == 'greedy':
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                else:
                    # --- 🎯 Apply Top-K Filtering ---
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')

                    # Apply temperature
                    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()

                # Stop if <EOS> is reached
                if token_id == self.token_to_id.get("<EOS>", 2):
                    break

                tokens_seen.append(token_id)
                word = self.id_to_token.get(token_id, "<UNK>")
                generated_words.append(word)
                curr_token = next_token

            return start_prompt + " " + " ".join(generated_words)
