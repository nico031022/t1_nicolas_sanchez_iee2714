from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta2 import (
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


def run_bins_experiment(
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

    region_configurations = [
        {
            "name": "small",
            "region_size": 64,
            "region_step": 32
        },
        {
            "name": "large",
            "region_size": 256,
            "region_step": 128
        }
    ]

    num_bins_values = [
        16,
        32,
        64,
        128,
        256
    ]

    contrast_strength = 1.0

    all_results = {}

    print()
    print("----------------------------------------")
    print(
        f"Numero de bins: {image_name}"
    )
    print("----------------------------------------")

    for current_config in (
        region_configurations
    ):

        region_size = current_config[
            "region_size"
        ]

        region_step = current_config[
            "region_step"
        ]

        config_name = current_config[
            "name"
        ]

        output_images = {}

        print()
        print(
            f"Configuracion: {config_name}"
        )

        print(
            f"region_size={region_size}, "
            f"region_step={region_step}"
        )

        for num_bins in num_bins_values:

            output_image = (
                local_histogram_equalization(
                    image_gray,
                    region_size=region_size,
                    region_step=region_step,
                    num_bins=num_bins,
                    contrast_strength=contrast_strength
                )
            )

            output_images[
                num_bins
            ] = output_image

            print(
                f"num_bins={num_bins}: OK"
            )

        reference_image = (
            output_images[256]
        )

        print(
            "Diferencia respecto a 256 bins:"
        )

        for num_bins in num_bins_values:

            current_difference = (
                mean_absolute_difference(
                    output_images[num_bins],
                    reference_image
                )
            )

            print(
                f"num_bins={num_bins:3d}"
                f" | mean difference="
                f"{current_difference:.3f}"
            )

        all_results[
            config_name
        ] = output_images

    figure, axes = plt.subplots(
        2,
        len(num_bins_values) + 1,
        figsize=(20, 8)
    )

    for row_index, current_config in enumerate(
        region_configurations
    ):

        config_name = current_config[
            "name"
        ]

        region_size = current_config[
            "region_size"
        ]

        region_step = current_config[
            "region_step"
        ]

        axes[
            row_index,
            0
        ].imshow(
            image_gray,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[
            row_index,
            0
        ].set_title(
            (
                "Original\n"
                f"size={region_size}, "
                f"step={region_step}"
            )
        )

        for column_index, num_bins in enumerate(
            num_bins_values
        ):

            current_image = (
                all_results[
                    config_name
                ][
                    num_bins
                ]
            )

            axes[
                row_index,
                column_index + 1
            ].imshow(
                current_image,
                cmap="gray",
                vmin=0,
                vmax=255
            )

            axes[
                row_index,
                column_index + 1
            ].set_title(
                f"bins={num_bins}"
            )

    for axis_row in axes:
        for axis in axis_row:
            axis.axis("off")

    figure.suptitle(
        (
            f"{image_name} | "
            "Local unlimited | "
            "50% overlap"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"bins_{image_name}.png"
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
        / "bins"
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

        run_bins_experiment(
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