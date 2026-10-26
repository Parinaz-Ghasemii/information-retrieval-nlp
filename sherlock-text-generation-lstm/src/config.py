import os
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

from src import utils

@dataclass
class DataConfig:
    """Parameters related to data loading and preprocessing."""
    sequence_length: int = 100
    max_vocab_size: int = 10000

@dataclass
class ModelConfig:
    """Hyperparameters for the LSTM model architecture."""
    embedding_dim: int = 128
    lstm_units: int = 256
    num_layers: int = 2
    dropout_rate: float = 0.5

@dataclass
class TrainConfig:
    """Parameters for the training process."""
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 0.001
    grad_clip_norm: Optional[float] = 1.0       # Clip gradients to prevent explosion
    amp: bool = False                           # Automatic Mixed Precision (for modern GPUs)
    monitor_metric: str = "val_loss"            # Metric to monitor for saving the best model
    ckpt_every_n_epochs: int = 5                # Save a checkpoint every N epochs
    early_stop_patience: Optional[int] = 3     # Patience for early stopping, None to disable

@dataclass
class GenerateConfig:
    """Parameters for text generation/inference."""
    beam_width: int = 5
    sampling_temperature: float = 0.8

@dataclass
class ProjectConfig:
    """
    Main configuration class that aggregates all other configs.
    Includes methods to save and load the configuration from a JSON file.
    """
    experiment_name: str = "sherlock_lstm_base"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)

    def save(self, filepath: str):
        """Saves the configuration to a JSON file."""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"[{self.__class__.__name__}] Saved configuration to {filepath}")



    @classmethod
    def load(cls, filepath: str) -> "ProjectConfig":
        """Loads the configuration from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        return cls(
            experiment_name=data.get("experiment_name"),
            data=DataConfig(**data.get("data")),
            model=ModelConfig(**data.get("model")),
            train=TrainConfig(**data.get("train")),
            generate=GenerateConfig(**data.get("generate"))
        )



# Example of how to use this structure
if __name__ == '__main__':

    config = ProjectConfig(experiment_name="my_first_experiment")
    
    config.train.epochs = 75
    config.model.lstm_units = 512
    
    experiment_dir = os.path.join(utils.CHECKPOINT_DIR, config.experiment_name)
    config_save_path = os.path.join(experiment_dir, "config.json")
    
    config.save(config_save_path)
    
    loaded_config = ProjectConfig.load(config_save_path)
    
    print("\n--- Loaded Config ---")
    print(loaded_config)
    assert loaded_config.train.epochs == 75
    assert loaded_config.model.lstm_units == 512
    print("\nVerification successful!")
