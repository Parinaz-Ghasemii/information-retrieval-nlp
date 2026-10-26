import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.config import ProjectConfig
from src import utils
from src.dataset import create_dataloaders
from src.model import EncoderDecoder

class ModelTrainer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = utils.get_device()
        self.logger = utils.get_logger("trainer")

        # Load data
        self.train_loader, self.val_loader = create_dataloaders(config.train, config.data)

        # Initialize Model
        self.model = EncoderDecoder(
            vocab_size=config.data.max_vocab_size,
            embedding_dim=config.model.embedding_dim,
            hidden_size=config.model.lstm_units,
            num_layers=config.model.num_layers,
            dropout_prob=config.model.dropout_rate
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.learning_rate)
        # ignore_index=0 ensures we don't calculate loss on <PAD> tokens
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train_one_epoch(self, epoch_index: int):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index} [Train]", leave=False)

        for src, trg in pbar:
            src, trg = src.to(self.device), trg.to(self.device)

            # 1. Zero Grad
            self.optimizer.zero_grad()

            # 2. Forward pass
            # Note: Decoder uses trg as input for teacher forcing
            outputs, _ = self.model(src, trg)

            # 3. Calculate Loss
            # outputs: (Batch, Seq_Len, Vocab_Size) -> (Batch * Seq_Len, Vocab_Size)
            # trg: (Batch, Seq_Len) -> (Batch * Seq_Len)
            V = outputs.size(-1)
            loss = self.criterion(outputs.view(-1, V), trg.view(-1))

            # 4. Backward & Step
            loss.backward()

            # Gradient Clipping (prevents exploding gradients in LSTMs)
            if self.config.train.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate(self, epoch_index: int):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_index} [Val]", leave=False)

        with torch.no_grad():
            for src, trg in pbar:
                src, trg = src.to(self.device), trg.to(self.device)

                outputs, _ = self.model(src, trg)
                V = outputs.size(-1)
                loss = self.criterion(outputs.view(-1, V), trg.view(-1))

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

    def train(self):
        os.makedirs(utils.CHECKPOINT_DIR, exist_ok=True)
        self.config.save(os.path.join(utils.CHECKPOINT_DIR, 'config.json'))

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.config.train.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss
            }

            # Regular checkpoint
            if epoch % self.config.train.ckpt_every_n_epochs == 0:
                torch.save(state, os.path.join(utils.CHECKPOINT_DIR, f'checkpoint_{epoch}.pth'))

            # Save best model and handle Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(state, os.path.join(utils.CHECKPOINT_DIR, 'best_model.pth'))
                self.logger.info(f"⭐ New best model saved at epoch {epoch}")
            else:
                patience_counter += 1

            if self.config.train.early_stop_patience and patience_counter >= self.config.train.early_stop_patience:
                self.logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                break
