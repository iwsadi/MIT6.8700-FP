import os
import glob
import torch

# Use a non-interactive backend (important on Windows / headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_plot(metrics, title, filename, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    ys = []
    for m in metrics:
        if isinstance(m, torch.Tensor):
            ys.append(m.detach().float().cpu().item())
        else:
            ys.append(float(m))
    plt.figure()
    plt.plot(ys)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def main():
    ckpts = glob.glob(os.path.join("checkpoints", "gcpn_prior_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found under checkpoints/gcpn_prior_epoch_*.pt")

    def epoch_num(p):
        # checkpoints/gcpn_prior_epoch_19.pt -> 19
        base = os.path.basename(p)
        return int(base.split("_")[-1].split(".")[0])

    latest = max(ckpts, key=epoch_num)
    print(f"Loading checkpoint: {latest}")
    ckpt = torch.load(latest, map_location="cpu")

    losses = ckpt.get("epoch_losses", None)
    stop_acc = ckpt.get("epoch_stop_acc", None)
    atom_acc = ckpt.get("epoch_add_node_acc", None)

    if losses is None or stop_acc is None or atom_acc is None:
        raise KeyError("Checkpoint missing one of: epoch_losses, epoch_stop_acc, epoch_add_node_acc")

    save_plot(losses, "Training Loss", "loss.png")
    save_plot(stop_acc, "Stop Prediction Accuracy", "stop_accuracy.png")
    save_plot(atom_acc, "Add Atom Prediction Accuracy", "atom_accuracy.png")

    print("Saved plots to figures/: loss.png, stop_accuracy.png, atom_accuracy.png")


if __name__ == "__main__":
    main()


