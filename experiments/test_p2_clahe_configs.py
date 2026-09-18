from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta2 import (
    global_equalization_own,
    local_histogram_equalization
)


def mean_absolute_difference(
    image_a,
    image_b
):
    difference = np.abs(
        image_a.astype(np.int16)
        - image_b.astype(np.int16)
    )

    return difference.mean()


def compute_clahe_grid(
    image_shape,
    approximate_tile_size
):
    image_height = image_shape[0]
    image_width = image_shape[1]

    tiles_x = max(
        1,
        round(
            image_width
            / approximate_tile_size
        )
    )

    tiles_y = max(
        1,
        round(
            image_height
            / approximate_tile_size
        )
    )

    return (
        tiles_x,
        tiles_y
    )


def run_clahe_configurations(
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

    configurations = [
        {
            "name": "soft",
            "region_size": 64,
            "region_step": 32,
            "contrast_strength": 0.25,
            "clahe_clip_limit": 1.0,
            "clahe_tile_size": 64
        },
        {
            "name": "medium",
            "region_size": 64,
            "region_step": 32,
            "contrast_strength": 0.50,
            "clahe_clip_limit": 2.0,
            "clahe_tile_size": 64
        },
        {
            "name": "strong",
            "region_size": 64,
            "region_step": 32,
            "contrast_strength": 0.75,
            "clahe_clip_limit": 4.0,
            "clahe_tile_size": 64
        },
        {
            "name": "large_regions",
            "region_size": 128,
            "region_step": 64,
            "contrast_strength": 0.50,
            "clahe_clip_limit": 2.0,
            "clahe_tile_size": 128
        }
    ]

    num_bins = 256

    global_equalized = (
        global_equalization_own(
            image_gray,
            num_bins=num_bins
        )
    )

    figure, axes = plt.subplots(
        len(configurations),
        5,
        figsize=(20, 14)
    )

    print()
    print("----------------------------------------")
    print(
        f"CLAHE configurations: {image_name}"
    )
    print("----------------------------------------")

    for row_index, current_config in enumerate(
        configurations
    ):
        region_size = current_config[
            "region_size"
        ]

        region_step = current_config[
            "region_step"
        ]

        contrast_strength = current_config[
            "contrast_strength"
        ]

        clahe_clip_limit = current_config[
            "clahe_clip_limit"
        ]

        clahe_tile_size = current_config[
            "clahe_tile_size"
        ]

        tiles_x, tiles_y = compute_clahe_grid(
            image_gray.shape,
            clahe_tile_size
        )

        local_unlimited = (
            local_histogram_equalization(
                image_gray,
                region_size=region_size,
                region_step=region_step,
                num_bins=num_bins,
                contrast_strength=1.0
            )
        )

        local_controlled = (
            local_histogram_equalization(
                image_gray,
                region_size=region_size,
                region_step=region_step,
                num_bins=num_bins,
                contrast_strength=contrast_strength
            )
        )

        # CLAHE solo como referencia
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(
                tiles_x,
                tiles_y
            )
        )

        clahe_image = clahe.apply(
            image_gray
        )

        controlled_difference = (
            mean_absolute_difference(
                local_controlled,
                image_gray
            )
        )

        clahe_difference = (
            mean_absolute_difference(
                clahe_image,
                image_gray
            )
        )

        print()
        print(
            "Configuration:",
            current_config["name"]
        )

        print(
            f"Our: size={region_size}, "
            f"step={region_step}, "
            f"alpha={contrast_strength:.2f}"
        )

        print(
            f"CLAHE: clip={clahe_clip_limit:.1f}, "
            f"tiles={tiles_x}x{tiles_y}"
        )

        print(
            "Mean difference - our method:",
            f"{controlled_difference:.3f}"
        )

        print(
            "Mean difference - CLAHE:",
            f"{clahe_difference:.3f}"
        )

        images = [
            image_gray,
            global_equalized,
            local_unlimited,
            local_controlled,
            clahe_image
        ]

        titles = [
            "Original",
            "Global",
            (
                "Local unlimited\n"
                f"size={region_size}, "
                f"step={region_step}"
            ),
            (
                "Nuestro control\n"
                f"alpha={contrast_strength:.2f}"
            ),
            (
                "CLAHE\n"
                f"clip={clahe_clip_limit:.1f}, "
                f"tiles={tiles_x}x{tiles_y}"
            )
        ]

        for column_index in range(5):

            axes[
                row_index,
                column_index
            ].imshow(
                images[
                    column_index
                ],
                cmap="gray",
                vmin=0,
                vmax=255
            )

            axes[
                row_index,
                column_index
            ].set_title(
                titles[
                    column_index
                ]
            )

            axes[
                row_index,
                column_index
            ].axis("off")

        axes[
            row_index,
            0
        ].set_ylabel(
            current_config["name"],
            fontsize=12
        )

    figure.suptitle(
        (
            f"{image_name} | "
            "Nuestro metodo vs CLAHE"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"clahe_configurations_{image_name}.png"
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
        / "clahe_configs"
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

        run_clahe_configurations(
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