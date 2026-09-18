from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta2 import (
    build_region_grid,
    compute_region_weights,
    global_equalization_own,
    local_histogram_equalization
)


def test_region_weights():
    image_shape = (
        64,
        64
    )

    region_grid = build_region_grid(
        image_shape=image_shape,
        region_size=64,
        region_step=64
    )

    region = region_grid[0]

    region_weights = compute_region_weights(
        region
    )

    minimum_weight = (
        region_weights.min()
    )

    maximum_weight = (
        region_weights.max()
    )

    print("----------------------------------------")
    print("Test de pesos")
    print("----------------------------------------")

    print(
        "Minimum weight:",
        f"{minimum_weight:.6f}"
    )

    print(
        "Maximum weight:",
        f"{maximum_weight:.6f}"
    )

    assert minimum_weight > 0
    assert maximum_weight <= 1

    print(
        "Todos los pesos son positivos: OK"
    )


def test_uniform_image():
    uniform_image = np.full(
        (160, 220),
        120,
        dtype=np.uint8
    )

    equalized_uniform = (
        local_histogram_equalization(
            uniform_image,
            region_size=64,
            region_step=32,
            num_bins=256
        )
    )

    difference = np.abs(
        equalized_uniform.astype(np.int16)
        - uniform_image.astype(np.int16)
    )

    maximum_difference = (
        difference.max()
    )

    print()
    print("----------------------------------------")
    print("Test de imagen uniforme")
    print("----------------------------------------")

    print(
        "Maximum difference:",
        maximum_difference
    )

    assert maximum_difference == 0

    print(
        "La imagen uniforme queda igual: OK"
    )


def run_local_experiment(
    image_path,
    image_name,
    output_directory
):
    image_gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image_gray is None:
        raise FileNotFoundError(
            f"No se pudo leer: {image_path}"
        )

    print()
    print("----------------------------------------")
    print(
        f"Imagen: {image_name}"
    )
    print("----------------------------------------")

    print(
        "Image shape:",
        image_gray.shape
    )

    single_region_size = max(
        image_gray.shape
    )

    local_single_region = (
        local_histogram_equalization(
            image_gray,
            region_size=single_region_size,
            region_step=single_region_size,
            num_bins=256
        )
    )

    global_image = (
        global_equalization_own(
            image_gray,
            num_bins=256
        )
    )

    single_difference = np.abs(
        local_single_region.astype(np.int16)
        - global_image.astype(np.int16)
    )

    print(
        "Maximum difference "
        "local/global:",
        single_difference.max()
    )

    assert single_difference.max() == 0

    print(
        "Una sola region reproduce "
        "la ecualizacion global: OK"
    )

    region_size = 64

    local_step_equal_size = (
        local_histogram_equalization(
            image_gray,
            region_size=region_size,
            region_step=64,
            num_bins=256
        )
    )

    local_overlap_50 = (
        local_histogram_equalization(
            image_gray,
            region_size=region_size,
            region_step=32,
            num_bins=256
        )
    )

    difference_step_equal = np.abs(
        local_step_equal_size.astype(np.int16)
        - image_gray.astype(np.int16)
    )

    difference_overlap = np.abs(
        local_overlap_50.astype(np.int16)
        - image_gray.astype(np.int16)
    )

    print()
    print(
        "region_size=64, "
        "region_step=64"
    )

    print(
        "Mean difference from original:",
        f"{difference_step_equal.mean():.3f}"
    )

    print()
    print(
        "region_size=64, "
        "region_step=32"
    )

    print(
        "Mean difference from original:",
        f"{difference_overlap.mean():.3f}"
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
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
        local_step_equal_size,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        "Local unlimited\n"
        "size=64, step=64"
    )

    axes[2].imshow(
        local_overlap_50,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[2].set_title(
        "Local unlimited\n"
        "size=64, step=32"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        image_name
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"local_unlimited_{image_name}.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        "Resultado guardado en:",
        output_path
    )


def main():
    project_directory = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    output_directory = (
        project_directory
        / "results"
        / "pregunta2"
        / "local_unlimited"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    test_region_weights()
    test_uniform_image()

    image_configurations = [
        {
            "name": "profesor",
            "path": (
                project_directory
                / "images"
                / "profesor"
                / "pregunta2.tif"
            )
        },
        {
            "name": "fruit",
            "path": (
                project_directory
                / "images"
                / "propias"
                / "fruit.jpg"
            )
        }
    ]

    for current_image in image_configurations:

        run_local_experiment(
            image_path=current_image[
                "path"
            ],
            image_name=current_image[
                "name"
            ],
            output_directory=output_directory
        )

    print()
    print("----------------------------------------")
    print(
        "Todos los tests terminaron "
        "correctamente."
    )
    print("----------------------------------------")


if __name__ == "__main__":
    main()