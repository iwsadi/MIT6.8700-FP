import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    csv_path = Path("KT_loss_1e-05_1e-05_200_0.99_2000")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Update the path/filename.")

    df = pd.read_csv(csv_path)
    y_cols = ["Total_Loss", "Soft_Target_Loss", "Label_Loss"]
    ax = df.plot(y=y_cols)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Transformer Distillation Losses")
    plt.tight_layout()
    out_path = Path("transfer_losses.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

