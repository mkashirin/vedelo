"""Plot the profiling measurements documented in the README."""

from pathlib import Path

import matplotlib.pyplot as plt

type Measurement = tuple[str, str, str, float]


def measurements(readme: Path) -> dict[str, list[Measurement]]:
    results = readme.read_text().split("### Results\n", 1)[1]
    results = results.split("\n## ", 1)[0]
    groups: dict[str, list[Measurement]] = {}
    for section in results.split("\n#### ")[1:]:
        title, _, table = section.partition("\n")
        rows: list[Measurement] = []
        for line in table.splitlines():
            if not line.startswith(("| CPU |", "| CUDA |", "| MPS |")):
                continue
            device, decode, encode, _, fps = (
                cell.strip() for cell in line.strip("|").split("|")
            )
            rows.append((device, decode, encode, float(fps)))
        groups[title] = rows
    return groups


def main() -> None:
    plt.switch_backend("Agg")
    root = Path(__file__).resolve().parents[1]
    groups = measurements(root / "README.md")
    colors = {"CPU": "#b45309", "CUDA": "#047857", "MPS": "#2563eb"}
    figure, axes = plt.subplots(
        len(groups),
        1,
        figsize=(10, 7.4),
        sharex=True,
        height_ratios=[len(rows) for rows in groups.values()],
        layout="constrained",
    )
    figure.suptitle("Runtime throughput", fontsize=18, fontweight="bold")
    figure.supylabel("Inference / Decode / Encode", fontsize=11)
    for axis, (title, rows) in zip(axes, groups.items(), strict=True):
        labels = [
            f"{device} / {decode} / {encode}" for device, decode, encode, _ in rows
        ]
        bars = axis.barh(
            labels,
            [fps for _, _, _, fps in rows],
            color=[colors[device] for device, _, _, _ in rows],
            edgecolor="#6b7280",
            linewidth=0.8,
            height=0.65,
            zorder=3,
        )
        axis.bar_label(bars, fmt="%.2f", padding=5, fontsize=10)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
        axis.set_xlim(0, 40)
        axis.set_facecolor("#f5f6f8")
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#d9dde3", linewidth=0.7)
        axis.tick_params(axis="both", length=0, labelsize=10, pad=8)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#9ca3af")
            spine.set_linewidth(0.8)
    axes[-1].set_xlabel("Frames per second (higher is better)", labelpad=10)
    figure.savefig(root / "assets/profile.png", dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
