from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src.pregunta2 import (
    local_histogram_equalization
)


def run_region_size_experiment(
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

    region_sizes = [
        32,
        64,
        128,
        256
    ]

    num_bins = 256
    contrast_strength = 1.0

    output_images = []

    print()
    print("----------------------------------------")
    print(
        f"Region size: {image_name}"
    )
    print("----------------------------------------")

    for region_size in region_sizes:

        region_step = (
            region_size
            // 2
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
            f"region_size={region_size}, "
            f"region_step={region_step}: OK"
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
        region_step = (
            region_size
            // 2
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
            f"size={region_size}\n"
            f"step={region_step}"
        )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        (
            f"{image_name} | "
            "Local unlimited | "
            "50% overlap | "
            f"num_bins={num_bins}"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"region_size_{image_name}.png"
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
        / "region_size"
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

        run_region_size_experiment(
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