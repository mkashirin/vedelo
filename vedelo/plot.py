import polars as pl
import matplotlib.pyplot as plt


RESULTS = "artifacts/vedelo-v1s/results.csv"

if __name__ == "__main__":
    df = pl.read_csv(RESULTS)
    epochs = df["epoch"].to_numpy()

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, df["train/box_loss"].to_numpy(), label="Train Box")
    plt.plot(epochs, df["val/box_loss"].to_numpy(), label="Val Box")
    plt.plot(epochs, df["train/cls_loss"].to_numpy(), label="Train Cls")
    plt.plot(epochs, df["val/cls_loss"].to_numpy(), label="Val Cls")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, df["metrics/mAP50(B)"].to_numpy(), label="mAP50")
    plt.plot(epochs, df["metrics/mAP50-95(B)"].to_numpy(), label="mAP50-95")
    plt.title("mAP Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, df["metrics/precision(B)"].to_numpy(), label="Precision")
    plt.plot(epochs, df["metrics/recall(B)"].to_numpy(), label="Recall")
    plt.title("Precision & Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, df["lr/pg0"].to_numpy(), label="LR")
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("static/training_results.png", dpi=300)
    plt.close()

    print("Saved training_results.png")
