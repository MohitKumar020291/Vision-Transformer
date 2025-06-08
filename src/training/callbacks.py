# early stopping - Stop training if validation loss doesn't improve for N epochs.
# Checkpointing	Save model when it performs best (e.g., lowest val loss).
# LR schedulers	ReduceLROnPlateau, cosine annealing restarts, warm-up logic.
# Custom logging	To file, WandB, or TensorBoard.
# Gradient logging	Visualize or clip gradients.
# Evaluation	Intermediate val/test performance tracking.

import torch

class Callback:
    def on_epoch_start(self, epoch, logs=None): ...
    def on_batch_end(self, batch, logs=None): ...
    def on_epoch_end(self, epoch, logs=None): ...

# Each time it will print loss whenever called
# could be calling for the epoch or batch _{start/end}
class PrintLossCallback(Callback):
    def on_batch_end(self, batch: int, logs=None):
        print(f"[Batch {batch}] Loss: {logs['loss']:.4f}")


class ModelCheckpoint(Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', float('inf'))
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(self.model.state_dict(), self.path)
            print(f"Saved new best model at epoch {epoch}, loss: {val_loss}")
