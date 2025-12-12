import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelDistillation:
    def __init__(
        self,
        optimizer,
        alpha,
        batch_size,
        n_steps,
        temp,
        version_name: str = None,
        results_dir: str = "distillation_runs",
        progress_interval: int = 1000,
        checkpoint_interval: int = 2000,
    ):
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.version_name = version_name or f"pretrain_{timestamp}"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.temp = temp
        self.optimizer = optimizer
        self.progress_interval = progress_interval
        self.checkpoint_interval = checkpoint_interval

        self._plot_path = self.results_dir / f"{self.version_name}_loss.png"
        self._csv_path = self.results_dir / f"{self.version_name}_losses.csv"
        self._metadata_path = self.results_dir / f"{self.version_name}_metadata.json"
        self._start_time: float | None = None
        self._last_log_time: float | None = None
        self._checkpoint_dir = self.results_dir / "checkpoints"
        self._checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def transfer(self, student, teacher):
        self.loss_history = []
        self.soft_target_loss_history = []
        self.label_loss_history = []
        self.accuracy_history = []
        self.min_loss_history = []
        self.max_loss_history = []

        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._log("Starting knowledge transfer", step=0)
        for step in range(self.n_steps):
            teacher_seq, teacher_logits = teacher.teach(self.batch_size)
            student_logits = student.q_values(teacher_seq)[:, :, :]

            teacher_soft = F.softmax(teacher_logits / self.temp, dim=-1)
            student_soft = F.log_softmax(student_logits / self.temp, dim=-1)
            flat_target = teacher_seq[:, 1:].reshape(-1)
            flat_logits = student_logits.reshape(-1, student_logits.size(-1))

            soft_target_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temp ** 2)
            label_loss = F.cross_entropy(flat_logits, flat_target)

            loss = self.alpha * soft_target_loss + (1 - self.alpha) * label_loss
            if loss.dim() > 0:
                loss = loss.mean()

            # Accuracy
            predictions = torch.argmax(flat_logits, dim=-1)
            correct_predictions = (predictions == flat_target).sum().item()
            accuracy = correct_predictions / flat_target.numel() * 100

            if step % 10 == 0:
                print(f"Step: {step}")
                print(f"Teacher sequence: {teacher_seq[:1,1:]}")
                print(f"Teacher logit: {torch.argmax(teacher_logits,dim=-1)[:1,:]}")
                print(f"Student sequence: {torch.argmax(student_logits,dim=-1)[:1,:]}")
                print(f"Step {step}/{self.n_steps}, Loss: {loss.item():.4f}, KL: {soft_target_loss.item():.4f}, CE: {label_loss.item():.4f}, Acc: {accuracy:.2f}%")

            self.loss_history.append(loss.item())
            self.soft_target_loss_history.append(soft_target_loss.item())
            self.label_loss_history.append(label_loss.item())
            self.accuracy_history.append(accuracy)

            if len(self.min_loss_history) == 0 or loss.item() < self.min_loss_history[-1]:
                self.min_loss_history.append(loss.item())
            else:
                self.min_loss_history.append(self.min_loss_history[-1])

            if len(self.max_loss_history) == 0 or loss.item() > self.max_loss_history[-1]:
                self.max_loss_history.append(loss.item())
            else:
                self.max_loss_history.append(self.max_loss_history[-1])

            self.update_params(loss)

            if (step + 1) % self.progress_interval == 0 or step == self.n_steps - 1:
                self._log_progress(step)
            if (step + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(student, step + 1)

        print("\n--- Knowledge Transfer Summary ---")
        print(f"Final Loss: {self.loss_history[-1]:.4f}")
        print(f"Min Loss: {min(self.loss_history):.4f}")
        print(f"Max Loss: {max(self.loss_history):.4f}")
        print(f"Average Accuracy: {sum(self.accuracy_history) / len(self.accuracy_history):.2f}%")
        print("----------------------------------\n")
        self._log_progress(self.n_steps - 1, force=True)
        self._save_metadata()
        self._save_final(student)

    def update_params(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_losses_to_csv(self, filename=None):
        target_path = Path(filename) if filename else self._csv_path
        if not self.loss_history:
            return
        with open(target_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Total Loss", "Soft Target Loss", "Label Loss", "Accuracy", "Min Loss", "Max Loss"])
            for step, (total_loss, soft_loss, label_loss, accuracy, min_loss, max_loss) in enumerate(
                zip(
                    self.loss_history,
                    self.soft_target_loss_history,
                    self.label_loss_history,
                    self.accuracy_history,
                    self.min_loss_history,
                    self.max_loss_history,
                )
            ):
                writer.writerow([step, total_loss, soft_loss, label_loss, accuracy, min_loss, max_loss])

    def _save_plot(self, filename=None):
        if not self.loss_history:
            return
        target_path = Path(filename) if filename else self._plot_path
        plt.figure(figsize=(6, 4))
        plt.plot(self.loss_history, label="Total")
        plt.plot(self.soft_target_loss_history, label="KL")
        plt.plot(self.label_loss_history, label="CE")
        ticks = list(range(0, len(self.loss_history), 1000))
        if len(self.loss_history) - 1 not in ticks:
            ticks.append(len(self.loss_history) - 1)
        plt.xticks(ticks)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(target_path, dpi=200)
        plt.close()

    def _save_metadata(self):
        metadata = {
            "version": self.version_name,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "n_steps": self.n_steps,
            "temperature": self.temp,
            "results_dir": str(self.results_dir),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(self._metadata_path, "w") as metafile:
            json.dump(metadata, metafile, indent=2)

    def _log(self, message, step=None):
        prefix = f"[distill:{self.version_name}]"
        step_info = f" Step {step}/{self.n_steps}" if step is not None else ""
        print(f"{prefix}{step_info} {message}", flush=True)

    def _log_progress(self, step, force=False):
        steps_done = min(step + 1, self.n_steps)
        now = time.time()
        if self._start_time is None:
            return
        elapsed = now - self._start_time
        avg_per_step = elapsed / steps_done if steps_done > 0 else 0
        remaining = max(self.n_steps - steps_done, 0)
        eta = timedelta(seconds=int(avg_per_step * remaining))
        if force or (step + 1) % self.progress_interval == 0 or step == self.n_steps - 1:
            self._log(f"ETA {eta}", step=steps_done)
            self._save_plot()
            self.save_losses_to_csv()

    def get_output_paths(self):
        return {
            "plot": str(self._plot_path),
            "csv": str(self._csv_path),
            "metadata": str(self._metadata_path),
        }

    def _save_checkpoint(self, student, step):
        name = f"{self.version_name}_step_{step}.pth"
        path = self._checkpoint_dir / name
        student.save_to_file(str(path))
        self._log(f"Checkpoint saved: {path}", step=step)

    def _save_final(self, student):
        name = f"{self.version_name}_weights_final.pth"
        path = self.results_dir / name
        student.save_to_file(str(path))
        self._log(f"Final weights saved: {path}", step=self.n_steps)
