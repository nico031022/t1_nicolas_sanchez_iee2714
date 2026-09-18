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


def run_main_comparison(
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
    contrast_strength = 0.50

    clahe_clip_limit = 2.0

    image_height = image_gray.shape[0]
    image_width = image_gray.shape[1]

    # buscar tiles de tamano parecido a regiones
    clahe_tiles_x = max(
        1,
        round(
            image_width
            / region_size
        )
    )

    clahe_tiles_y = max(
        1,
        round(
            image_height
            / region_size
        )
    )

    global_equalized = (
        global_equalization_own(
            image_gray,
            num_bins=num_bins
        )
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
            clahe_tiles_x,
            clahe_tiles_y
        )
    )

    clahe_image = clahe.apply(
        image_gray
    )

    global_difference = (
        mean_absolute_difference(
            global_equalized,
            image_gray
        )
    )

    unlimited_difference = (
        mean_absolute_difference(
            local_unlimited,
            image_gray
        )
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

    approximate_tile_width = (
        image_width
        / clahe_tiles_x
    )

    approximate_tile_height = (
        image_height
        / clahe_tiles_y
    )

    print()
    print("----------------------------------------")
    print(
        f"Comparacion principal: {image_name}"
    )
    print("----------------------------------------")

    print(
        "Image shape:",
        image_gray.shape
    )

    print()
    print("Nuestro metodo")
    print(
        "region_size:",
        region_size
    )
    print(
        "region_step:",
        region_step
    )
    print(
        "num_bins:",
        num_bins
    )
    print(
        "contrast_strength:",
        contrast_strength
    )

    print()
    print("CLAHE")
    print(
        "clipLimit:",
        clahe_clip_limit
    )
    print(
        "tileGridSize:",
        (
            clahe_tiles_x,
            clahe_tiles_y
        )
    )
    print(
        "Approximate tile size:",
        (
            f"{approximate_tile_width:.1f}",
            f"{approximate_tile_height:.1f}"
        )
    )

    print()
    print("Mean difference from original")
    print(
        "Global:",
        f"{global_difference:.3f}"
    )
    print(
        "Local unlimited:",
        f"{unlimited_difference:.3f}"
    )
    print(
        "Our controlled:",
        f"{controlled_difference:.3f}"
    )
    print(
        "CLAHE:",
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
            f"clip={clahe_clip_limit}, "
            f"tiles={clahe_tiles_x}x"
            f"{clahe_tiles_y}"
        )
    ]

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 5)
    )

    for axis, image, title in zip(
        axes,
        images,
        titles
    ):
        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axis.set_title(
            title
        )

        axis.axis("off")

    figure.suptitle(
        (
            f"{image_name} | "
            f"num_bins={num_bins}"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"main_comparison_{image_name}.png"
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
        / "clahe"
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

        run_main_comparison(
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
        "Comparaciones principales "
        "terminadas correctamente."
    )
    print("----------------------------------------")


if __name__ == "__main__":
    main()