from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.pregunta2 import (
    local_histogram_equalization
)


def crop_around_point(
    image_gray,
    center_x,
    center_y,
    crop_size
):
    half_size = (
        crop_size
        // 2
    )

    left = (
        center_x
        - half_size
    )

    right = (
        center_x
        + half_size
    )

    top = (
        center_y
        - half_size
    )

    bottom = (
        center_y
        + half_size
    )

    image_crop = image_gray[
        top:bottom,
        left:right
    ]

    return image_crop


def run_boundary_experiment(
    image_path,
    image_name,
    boundary_point,
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
    num_bins = 256
    contrast_strength = 1.0

    region_steps = [
        128,
        96,
        64,
        32
    ]

    crop_size = 256

    output_images = []
    output_crops = []

    for region_step in region_steps:

        output_image = (
            local_histogram_equalization(
                image_gray,
                region_size=region_size,
                region_step=region_step,
                num_bins=num_bins,
                contrast_strength=contrast_strength
            )
        )

        image_crop = crop_around_point(
            output_image,
            center_x=boundary_point[
                "x"
            ],
            center_y=boundary_point[
                "y"
            ],
            crop_size=crop_size
        )

        output_images.append(
            output_image
        )

        output_crops.append(
            image_crop
        )

    original_crop = crop_around_point(
        image_gray,
        center_x=boundary_point["x"],
        center_y=boundary_point["y"],
        crop_size=crop_size
    )

    print()
    print("----------------------------------------")
    print(
        f"Boundary artifacts: {image_name}"
    )
    print("----------------------------------------")

    print(
        "Boundary point:",
        boundary_point
    )

    print(
        "Crop size:",
        crop_size
    )

    for region_step in region_steps:

        overlap_percentage = (
            100.0
            * (
                region_size
                - region_step
            )
            / region_size
        )

        print(
            f"step={region_step}"
            f" | overlap="
            f"{overlap_percentage:.1f}%"
        )

    figure, axes = plt.subplots(
        2,
        5,
        figsize=(18, 8)
    )

    axes[0, 0].imshow(
        image_gray,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    half_size = (
        crop_size
        // 2
    )

    rectangle = Rectangle(
        (
            boundary_point["x"]
            - half_size,
            boundary_point["y"]
            - half_size
        ),
        crop_size,
        crop_size,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    axes[0, 0].add_patch(
        rectangle
    )

    axes[0, 0].set_title(
        "Original + crop"
    )

    for column_index, region_step in enumerate(
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
            0,
            column_index + 1
        ].imshow(
            output_images[
                column_index
            ],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[
            0,
            column_index + 1
        ].set_title(
            f"step={region_step}\n"
            f"overlap={overlap_percentage:.0f}%"
        )

    axes[1, 0].imshow(
        original_crop,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 0].set_title(
        "Crop original"
    )

    for column_index, region_step in enumerate(
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
            1,
            column_index + 1
        ].imshow(
            output_crops[
                column_index
            ],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        # marcar punto ref
        axes[
            1,
            column_index + 1
        ].axvline(
            crop_size / 2,
            linewidth=1
        )

        axes[
            1,
            column_index + 1
        ].axhline(
            crop_size / 2,
            linewidth=1
        )

        axes[
            1,
            column_index + 1
        ].set_title(
            f"Crop {overlap_percentage:.0f}%"
        )

    for axis_row in axes:
        for axis in axis_row:
            axis.axis("off")

    figure.suptitle(
        (
            f"{image_name} | "
            f"region_size={region_size} | "
            "Local unlimited"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"boundaries_{image_name}.png"
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
        / "boundaries"
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
            ),
            "boundary_point": {
                "x": 640,
                "y": 512
            }
        },
        {
            "name": "fruit",
            "path": (
                project_directory
                / "images"
                / "propias"
                / "fruit.jpg"
            ),
            "boundary_point": {
                "x": 768,
                "y": 640
            }
        }
    ]

    for current_image in image_configurations:

        run_boundary_experiment(
            image_path=current_image[
                "path"
            ],
            image_name=current_image[
                "name"
            ],
            boundary_point=current_image[
                "boundary_point"
            ],
            output_directory=output_directory
        )


if __name__ == "__main__":
    main()