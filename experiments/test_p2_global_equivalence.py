from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta2 import global_equalization_own


def main():
    project_directory = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    image_path = (
        project_directory
        / "images"
        / "propias"
        / "fruit.jpg"
    )

    output_directory = (
        project_directory
        / "results"
        / "pregunta2"
        / "global_equivalence"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    image_gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image_gray is None:
        raise FileNotFoundError(
            f"No se pudo leer: {image_path}"
        )

    equalized_own = global_equalization_own(
        image_gray,
        num_bins=256
    )

    # solo como referencia
    equalized_opencv = cv2.equalizeHist(
        image_gray
    )

    difference = np.abs(
        equalized_own.astype(np.int16)
        - equalized_opencv.astype(np.int16)
    )

    maximum_difference = (
        difference.max()
    )

    mean_difference = (
        difference.mean()
    )

    different_pixels = np.count_nonzero(
        difference
    )

    total_pixels = difference.size

    different_percentage = (
        100.0
        * different_pixels
        / total_pixels
    )

    print("----------------------------------------")
    print("Global equalization equivalence")
    print("----------------------------------------")

    print(
        "Image shape:",
        image_gray.shape
    )

    print(
        "Maximum absolute difference:",
        maximum_difference
    )

    print(
        "Mean absolute difference:",
        f"{mean_difference:.6f}"
    )

    print(
        "Different pixels:",
        different_pixels
    )

    print(
        "Different pixels percentage:",
        f"{different_percentage:.6f}%"
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5)
    )

    axes[0].imshow(
        image_gray,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0].set_title(
        "Original"
    )

    axes[1].imshow(
        equalized_own,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        "Nuestra ecualizacion global"
    )

    axes[2].imshow(
        equalized_opencv,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[2].set_title(
        "OpenCV equalizeHist"
    )

    difference_plot = axes[3].imshow(
        difference,
        cmap="gray"
    )

    axes[3].set_title(
        "Diferencia absoluta"
    )

    figure.colorbar(
        difference_plot,
        ax=axes[3]
    )

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    output_path = (
        output_directory
        / "global_equivalence.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(figure)

    print()
    print(
        "Resultado guardado en:",
        output_path
    )


if __name__ == "__main__":
    main()