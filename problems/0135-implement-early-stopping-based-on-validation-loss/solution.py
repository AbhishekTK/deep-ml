from typing import Tuple

from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_loss = val_losses[0]
    best_epoch = 0
    wait = 0

    for i in range(1, len(val_losses)):
        # Check if current loss improved by more than min_delta
        if best_loss - val_losses[i] > min_delta:
            best_loss = val_losses[i]
            best_epoch = i
            wait = 0  # Reset patience counter
        else:
            wait += 1  # Increment counter on no improvement
            if wait >= patience:
                return (i, best_epoch)

    # If training completes without triggering early stopping
    return (len(val_losses) - 1, best_epoch)

            