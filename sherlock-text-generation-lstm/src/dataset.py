import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from src import utils
from src.config import TrainConfig, DataConfig


class SherlockDataset(Dataset):
    """
    Each sample is a contiguous chunk of length seq_length.
    The chunk is split into:
      - src: first half  (encoder input)
      - trg: second half (decoder input / target)
    """

    def __init__(self, data_tensor: torch.Tensor, sequence_length: int):
        if sequence_length % 2 != 0:
            raise ValueError("Sequence length must be even for encoder-decoder split.")

        self.data = data_tensor
        self.sequence_length = sequence_length
        self.half = sequence_length // 2

        # Number of non-overlapping chunks
        self.n_samples = len(self.data) // self.sequence_length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.sequence_length
        end = start + self.sequence_length

        chunk = self.data[start:end]

        # Split into source and target
        src = chunk[:self.half]
        trg = chunk[self.half:]

        return src, trg


def create_dataloaders(train_config: TrainConfig, data_config: DataConfig):
    """
    Creates train and validation dataloaders from the processed tensor.
    """

    if not os.path.exists(utils.SEQUENCES_PATH):
        raise FileNotFoundError("Processed data missing. Run preprocessing first.")

    data_tensor = torch.load(utils.SEQUENCES_PATH)

    dataset = SherlockDataset(data_tensor, data_config.sequence_length)

    # Train / validation split
    val_size = int(len(dataset) * 0.2) # val_split = 0.2
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader
