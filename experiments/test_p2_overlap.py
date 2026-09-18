from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src.pregunta2 import (
    build_region_grid,
    local_histogram_equalization
)


def run_overlap_experiment(
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

    # mantenr la separacion nominal entre regiones
    region_step = 64

    region_sizes = [
        64,
        80,
        96,
        128
    ]

    num_bins = 256
    contrast_strength = 1.0

    output_images = []

    print()
    print("----------------------------------------")
    print(
        f"Overlap: {image_name}"
    )
    print("----------------------------------------")

    for region_size in region_sizes:

        overlap_pixels = (
            region_size
            - region_step
        )

        overlap_percentage = (
            100.0
            * overlap_pixels
            / region_size
        )

        region_grid = build_region_grid(
            image_shape=image_gray.shape,
            region_size=region_size,
            region_step=region_step
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
            f"size={region_size}"
            f" | step={region_step}"
            f" | overlap={overlap_pixels} px"
            f" ({overlap_percentage:.1f}%)"
            f" | regions={len(region_grid)}"
        )

    figure, axes = plt.subplots(
        1,
        len(region_sizes) + 1,
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

    for image_index, region_size in enumerate(
        region_sizes
    ):
        overlap_pixels = (
            region_size
            - region_step
        )

        overlap_percentage = (
            100.0
            * overlap_pixels
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
            f"size={region_size}, "
            f"step={region_step}\n"
            f"overlap={overlap_percentage:.0f}%"
        )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        (
            f"{image_name} | "
            "Local unlimited | "
            f"region_step={region_step} | "
            f"num_bins={num_bins}"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"overlap_{image_name}.png"
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
        / "overlap"
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

        run_overlap_experiment(
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