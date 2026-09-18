from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta2 import (
    build_region_grid,
    local_histogram_equalization
)


def compute_average_coverage(
    image_shape,
    region_grid
):
    coverage_map = np.zeros(
        image_shape,
        dtype=np.int32
    )

    for region in region_grid:

        top = region["top"]
        bottom = region["bottom"]
        left = region["left"]
        right = region["right"]

        coverage_map[
            top:bottom,
            left:right
        ] += 1

    average_coverage = (
        coverage_map.mean()
    )

    maximum_coverage = (
        coverage_map.max()
    )

    return (
        average_coverage,
        maximum_coverage
    )


def run_mesh_density_experiment(
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

    region_size = 128

    region_steps = [
        128,
        96,
        64,
        32
    ]

    num_bins = 256
    contrast_strength = 1.0

    output_images = []

    print()
    print("----------------------------------------")
    print(
        f"Mesh density: {image_name}"
    )
    print("----------------------------------------")

    for region_step in region_steps:

        overlap_percentage = (
            100.0
            * (
                region_size
                - region_step
            )
            / region_size
        )

        region_grid = build_region_grid(
            image_shape=image_gray.shape,
            region_size=region_size,
            region_step=region_step
        )

        (
            average_coverage,
            maximum_coverage
        ) = compute_average_coverage(
            image_gray.shape,
            region_grid
        )

        output_image = (
            local_histogram_equalization(
                image_gray,
                region_size=region_size,
                region_step=region_step,
                num_bins=num_bins,
                contrast_strength=contrast_strength
            )
        )

        output_images.append(
            output_image
        )

        print(
            f"step={region_step}"
            f" | overlap={overlap_percentage:.1f}%"
            f" | regions={len(region_grid)}"
            f" | average coverage="
            f"{average_coverage:.2f}"
            f" | max coverage="
            f"{maximum_coverage}"
        )

    figure, axes = plt.subplots(
        1,
        len(region_steps) + 1,
        figsize=(18, 5)
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

    for image_index, region_step in enumerate(
        region_steps
    ):
        overlap_percentage = (
            100.0
            * (
                region_size
                - region_step
            )
            / region_size
        )

        axes[
            image_index + 1
        ].imshow(
            output_images[
                image_index
            ],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[
            image_index + 1
        ].set_title(
            f"step={region_step}\n"
            f"overlap={overlap_percentage:.0f}%"
        )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        (
            f"{image_name} | "
            f"region_size={region_size} | "
            "Local unlimited | "
            f"num_bins={num_bins}"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"mesh_density_{image_name}.png"
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
        / "mesh_density"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

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

        run_mesh_density_experiment(
            image_path=current_image[
                "path"
            ],
            image_name=current_image[
                "name"
            ],
            output_directory=output_directory
        )


if __name__ == "__main__":
    main()