from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta2 import (
    local_histogram_equalization
)


def run_contrast_experiment(
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

    region_size = 64
    region_step = 32
    num_bins = 256

    contrast_strength_values = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0
    ]

    output_images = []

    print()
    print("----------------------------------------")
    print(
        f"Control de contraste: {image_name}"
    )
    print("----------------------------------------")

    print(
        "Image shape:",
        image_gray.shape
    )

    for contrast_strength in (
        contrast_strength_values
    ):

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

        difference = np.abs(
            output_image.astype(np.int16)
            - image_gray.astype(np.int16)
        )

        mean_difference = (
            difference.mean()
        )

        maximum_difference = (
            difference.max()
        )

        print(
            f"alpha={contrast_strength:.2f}"
            f" | mean difference="
            f"{mean_difference:.3f}"
            f" | max difference="
            f"{maximum_difference}"
        )

    # alpha=0 debe ser identidad
    identity_difference = np.abs(
        output_images[0].astype(np.int16)
        - image_gray.astype(np.int16)
    )

    assert identity_difference.max() == 0

    print(
        "alpha=0 mantiene "
        "la imagen original: OK"
    )

    unlimited_reference = (
        local_histogram_equalization(
            image_gray,
            region_size=region_size,
            region_step=region_step,
            num_bins=num_bins,
            contrast_strength=1.0
        )
    )

    unlimited_difference = np.abs(
        output_images[-1].astype(np.int16)
        - unlimited_reference.astype(np.int16)
    )

    assert unlimited_difference.max() == 0

    print(
        "alpha=1 reproduce "
        "el metodo no limitado: OK"
    )

    figure, axes = plt.subplots(
        1,
        len(
            contrast_strength_values
        ) + 1,
        figsize=(21, 5)
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

    for image_index, contrast_strength in enumerate(
        contrast_strength_values
    ):

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
            "Control local\n"
            f"alpha={contrast_strength:.2f}"
        )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        (
            f"{image_name} | "
            f"region_size={region_size}, "
            f"region_step={region_step}, "
            f"num_bins={num_bins}"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"contrast_strength_{image_name}.png"
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
        / "contrast_control"
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

        run_contrast_experiment(
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
        "Experimentos de control "
        "terminados correctamente."
    )
    print("----------------------------------------")


if __name__ == "__main__":
    main()