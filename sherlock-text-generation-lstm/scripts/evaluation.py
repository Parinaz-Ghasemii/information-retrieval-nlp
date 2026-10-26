import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from nltk.util import ngrams
from typing import List, Tuple
import json

# Ensure the root directory is in the path for src imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import ProjectConfig
from src.dataset import SherlockDataset
from src.model import EncoderDecoder
from src import utils

class ModelEvaluator:
    """
    Handles evaluation using Perplexity and Distinct-n metrics.
    Fixes the massive perplexity issue by ignoring padding in calculations.
    """
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Consistent with your main.py and project structure
        self.experiment_dir = utils.get_experiment_dir(config)
        model_path = "/content/sherlock_project/checkpoints/best_model.pth"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}. Train the model first.")

        # Load the vocabulary to get special token IDs
        with open(utils.VOCAB_PATH, 'r', encoding='utf-8') as f:
            self.vocab = torch.load(utils.VOCAB_PATH) if utils.VOCAB_PATH.endswith('.pth') else json.load(f)

        self.model = self._load_model(model_path)
        self.val_loader = self._get_val_loader()

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

    def _get_val_loader(self) -> DataLoader:
        """Fixes the 'DataConfig has no len()' error by loading the tensor first."""
        if not os.path.exists(utils.SEQUENCES_PATH):
            raise FileNotFoundError(f"Sequences not found at {utils.SEQUENCES_PATH}")

        # Load the actual data tensor (usually saved during preprocessing)
        data_tensor = torch.load(utils.SEQUENCES_PATH)

        # Initialize dataset with the tensor, NOT the config object
        full_dataset = SherlockDataset(data_tensor, self.config.data.sequence_length)

        # Use a fixed seed to ensure validation set is distinct from training
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        _, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        return DataLoader(val_dataset, batch_size=self.config.train.batch_size, shuffle=False)

    def calculate_perplexity(self) -> float:
        """Calculates token-level perplexity, ignoring the <PAD> index (0)."""
        # ignore_index=0 is critical for valid perplexity scores
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for src, trg in tqdm(self.val_loader, desc="Evaluating Perplexity"):
                src, trg = src.to(self.device), trg.to(self.device)
                output, _ = self.model(src, trg)

                # Flatten the output and target for cross entropy
                loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))
                total_loss += loss.item()
                # Only count non-padding tokens
                total_tokens += (trg != 0).sum().item()

        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()

    @staticmethod
    def calculate_distinct_n(text: str) -> Tuple[float, float]:
        """Calculates unigram (D1) and bigram (D2) diversity."""
        tokens = text.lower().split()
        if not tokens: return 0.0, 0.0

        d1 = len(set(ngrams(tokens, 1))) / len(tokens)
        bigrams = list(ngrams(tokens, 2))
        d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        return d1, d2

    def calculate_diversity_from_samples(self, generator_obj=None, prompts=None) -> Tuple[float, float]:
            """
            Calculates average Distinct-1 and Distinct-2 diversity.
            Modified to work without arguments to match the call in main.py.
            """
            # 1. Handle missing generator
            if generator_obj is None:
                from scripts.generate import TextGenerator
                # self.config is already available in ModelEvaluator's __init__
                generator_obj = TextGenerator(self.config)

            # 2. Handle missing prompts
            if prompts is None:
                prompts = [
                    "Sherlock Holmes was a",
                    "The mystery of the",
                    "London was a dark",
                    "He looked at the evidence",
                    "The case was finally"
                ]

            d1_scores = []
            d2_scores = []

            print(f"\nEvaluating diversity on {len(prompts)} samples...")

            for prompt in prompts:
                # Generate a sample using the 'sample' strategy for variety
                generated_text = generator_obj.generate(
                    prompt,
                    max_length=50,
                    strategy='sample'
                )

                # Calculate metrics
                d1, d2 = self.calculate_distinct_n(generated_text)
                d1_scores.append(d1)
                d2_scores.append(d2)

            avg_d1 = sum(d1_scores) / len(prompts)
            avg_d2 = sum(d2_scores) / len(prompts)

            return avg_d1, avg_d2

def evaluate(config: ProjectConfig):
    """Entry point called by main.py."""
    print("\n--- Step 3: Model Evaluation ---")
    evaluator = ModelEvaluator(config)

    # Perplexity
    ppl = evaluator.calculate_perplexity()
    print(f"✅ Perplexity on validation set: {ppl:.4f}")

    # Diversity
    from scripts.generate import TextGenerator
    generator = TextGenerator(config.experiment_name)
    sample_prompts = ["Sherlock Holmes was", "The case of the", "Watson looked at"]
    d1, d2 = evaluator.calculate_diversity_from_samples(generator, sample_prompts)

    print(f"✅ Distinct-1 (Diversity): {d1:.4f}")
    print(f"✅ Distinct-2 (Diversity): {d2:.4f}")





