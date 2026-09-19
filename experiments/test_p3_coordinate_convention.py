from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta3 import (
    bilinear_value,
    calculate_output_size,
    resize_image,
)


SCALE_FACTORS = [
    0.6,
    1.7,
]


CROP_REGION = {
    "top": 1220,
    "bottom": 1740,
    "left": 0,
    "right": 650,
}


def map_output_coordinate_direct(
    output_coordinate,
    input_size,
    output_size,
):

    scale = output_size / input_size

    input_coordinate = (
        output_coordinate / scale
    )

    # coordenada dentro de image
    input_coordinate = np.clip(
        input_coordinate,
        0.0,
        input_size - 1,
    )

    return input_coordinate


def resize_image_direct_mapping(
    image,
    scale_factor,
):

    output_height, output_width = (
        calculate_output_size(
            image.shape,
            scale_factor,
        )
    )

    input_height = image.shape[0]
    input_width = image.shape[1]

    if image.ndim == 2:

        output_image = np.zeros(
            (output_height, output_width),
            dtype=image.dtype,
        )

    else:

        output_image = np.zeros(
            (
                output_height,
                output_width,
                image.shape[2],
            ),
            dtype=image.dtype,
        )

    mapped_rows = np.zeros(
        output_height,
        dtype=np.float64,
    )

    mapped_cols = np.zeros(
        output_width,
        dtype=np.float64,
    )

    for output_row in range(output_height):

        mapped_rows[output_row] = (
            map_output_coordinate_direct(
                output_row,
                input_height,
                output_height,
            )
        )

    for output_col in range(output_width):

        mapped_cols[output_col] = (
            map_output_coordinate_direct(
                output_col,
                input_width,
                output_width,
            )
        )

    for output_row in range(output_height):
        for output_col in range(output_width):

            input_row = (
                mapped_rows[output_row]
            )

            input_col = (
                mapped_cols[output_col]
            )

            pixel_value = bilinear_value(
                image,
                input_row,
                input_col,
            )

            if np.issubdtype(
                image.dtype,
                np.integer,
            ):

                dtype_limits = np.iinfo(
                    image.dtype
                )

                pixel_value = np.rint(
                    pixel_value
                )

                pixel_value = np.clip(
                    pixel_value,
                    dtype_limits.min,
                    dtype_limits.max,
                )

            output_image[
                output_row,
                output_col,
            ] = pixel_value

    return output_image


def calculate_mean_absolute_difference(
    image_a,
    image_b,
):

    image_a_float = image_a.astype(
        np.float64
    )

    image_b_float = image_b.astype(
        np.float64
    )

    absolute_difference = np.abs(
        image_a_float
        - image_b_float
    )

    mean_difference = np.mean(
        absolute_difference
    )

    return mean_difference


def create_difference_image(
    image_a,
    image_b,
):

    image_a_float = image_a.astype(
        np.float64
    )

    image_b_float = image_b.astype(
        np.float64
    )

    difference = np.abs(
        image_a_float
        - image_b_float
    )

    # amplificar la dif para verla mejor
    difference_visible = np.clip(
        difference * 5.0,
        0,
        255,
    )

    difference_visible = (
        difference_visible.astype(
            np.uint8
        )
    )

    return difference_visible


def main():

    image_path = Path(
        "images/profesor/pregunta3.tif"
    )

    original_image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if original_image is None:
        raise ValueError(
            f"No se pudo leer {image_path}"
        )

    crop = original_image[
        CROP_REGION["top"]:CROP_REGION["bottom"],
        CROP_REGION["left"]:CROP_REGION["right"],
    ]

    output_directory = Path(
        "results/pregunta3/coordinate_convention"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    for row_index, scale_factor in enumerate(
        SCALE_FACTORS
    ):

        pixel_center_image = resize_image(
            crop,
            scale_factor,
            "bilinear",
        )

        corner_image = (
            resize_image_direct_mapping(
                crop,
                scale_factor,
            )
        )

        mean_difference = (
            calculate_mean_absolute_difference(
                pixel_center_image,
                corner_image,
            )
        )

        difference_image = (
            create_difference_image(
                pixel_center_image,
                corner_image,
            )
        )

        print(
            f"\nFactor s = {scale_factor}"
        )

        print(
            f"Shape: "
            f"{pixel_center_image.shape}"
        )

        print(
            f"Diferencia media absoluta: "
            f"{mean_difference:.4f}"
        )

        pixel_center_rgb = cv2.cvtColor(
            pixel_center_image,
            cv2.COLOR_BGR2RGB,
        )

        corner_rgb = cv2.cvtColor(
            corner_image,
            cv2.COLOR_BGR2RGB,
        )

        difference_rgb = cv2.cvtColor(
            difference_image,
            cv2.COLOR_BGR2RGB,
        )

        axes[row_index, 0].imshow(
            pixel_center_rgb,
            interpolation="nearest",
        )

        axes[row_index, 0].set_title(
            f"Pixel-center, s = {scale_factor}"
        )

        axes[row_index, 1].imshow(
            corner_rgb,
            interpolation="nearest",
        )

        axes[row_index, 1].set_title(
            f"Direct mapping, s = {scale_factor}"
        )

        axes[row_index, 2].imshow(
            difference_rgb,
            interpolation="nearest",
        )

        axes[row_index, 2].set_title(
            "Absolute difference x5"
        )

    for axis in axes.flat:
        axis.axis("off")

    figure.tight_layout()

    output_path = (
        output_directory
        / "coordinate_convention_profesor.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"\nGuardado: {output_path}"
    )


if __name__ == "__main__":
    main()