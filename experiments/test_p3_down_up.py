from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt

from src.pregunta3 import resize_image


DOWN_SCALE = 0.6


CROP_REGIONS = {
    "profesor": {
        "top": 1050,
        "bottom": 1700,
        "left": 50,
        "right": 800,
    },
    "fruit": {
        "top": 250,
        "bottom": 620,
        "left": 620,
        "right": 1080,
    },
}


ORIGINAL_IMAGES = {
    "profesor": Path(
        "images/profesor/pregunta3.tif"
    ),
    "fruit": Path(
        "images/propias/fruit.jpg"
    ),
}


def compute_up_scale():

    up_scale = 1.0 / DOWN_SCALE

    return up_scale


def crop_image(
    image,
    crop_region,
):

    image_height = image.shape[0]
    image_width = image.shape[1]

    top = min(
        crop_region["top"],
        image_height,
    )

    bottom = min(
        crop_region["bottom"],
        image_height,
    )

    left = min(
        crop_region["left"],
        image_width,
    )

    right = min(
        crop_region["right"],
        image_width,
    )

    crop = image[
        top:bottom,
        left:right,
    ]

    return crop


def process_method(
    original_image,
    method,
):

    # primero reducir
    reduced_image = resize_image(
        original_image,
        DOWN_SCALE,
        method,
    )

    up_scale = compute_up_scale()

    # volver a ampliar
    recovered_image = resize_image(
        reduced_image,
        up_scale,
        method,
    )

    return reduced_image, recovered_image, up_scale


def create_down_up_figure(
    image_name,
    output_directory,
):

    original_path = ORIGINAL_IMAGES[
        image_name
    ]

    original_image = cv2.imread(
        str(original_path),
        cv2.IMREAD_COLOR,
    )

    if original_image is None:
        raise ValueError(
            f"No se pudo leer {original_path}"
        )

    print(f"\nImagen: {image_name}")
    print(
        f"Shape original: {original_image.shape}"
    )

    nearest_start = time.perf_counter()

    nearest_reduced, nearest_recovered, nearest_up_scale = process_method(
        original_image,
        "nearest",
    )

    nearest_time = (
        time.perf_counter()
        - nearest_start
    )

    bilinear_start = time.perf_counter()

    bilinear_reduced, bilinear_recovered, bilinear_up_scale = process_method(
        original_image,
        "bilinear",
    )

    bilinear_time = (
        time.perf_counter()
        - bilinear_start
    )

    print(
        f"Nearest reduced: {nearest_reduced.shape}"
    )

    print(
        f"Nearest recovered: {nearest_recovered.shape}"
    )

    print(
        f"Nearest up scale: {nearest_up_scale:.6f}"
    )

    print(
        f"Nearest total time: {nearest_time:.2f} s"
    )

    print(
        f"Bilinear reduced: {bilinear_reduced.shape}"
    )

    print(
        f"Bilinear recovered: {bilinear_recovered.shape}"
    )

    print(
        f"Bilinear up scale: {bilinear_up_scale:.6f}"
    )

    print(
        f"Bilinear total time: {bilinear_time:.2f} s"
    )

    crop_region = CROP_REGIONS[
        image_name
    ]

    original_crop = crop_image(
        original_image,
        crop_region,
    )

    nearest_crop = crop_image(
        nearest_recovered,
        crop_region,
    )

    bilinear_crop = crop_image(
        bilinear_recovered,
        crop_region,
    )

    original_crop_rgb = cv2.cvtColor(
        original_crop,
        cv2.COLOR_BGR2RGB,
    )

    nearest_crop_rgb = cv2.cvtColor(
        nearest_crop,
        cv2.COLOR_BGR2RGB,
    )

    bilinear_crop_rgb = cv2.cvtColor(
        bilinear_crop,
        cv2.COLOR_BGR2RGB,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        original_crop_rgb,
        interpolation="nearest",
    )

    axes[0].set_title(
        "Original"
    )

    axes[1].imshow(
        nearest_crop_rgb,
        interpolation="nearest",
    )

    axes[1].set_title(
        "Nearest\n0.6 -> up"
    )

    axes[2].imshow(
        bilinear_crop_rgb,
        interpolation="nearest",
    )

    axes[2].set_title(
        "Bilinear\n0.6 -> up"
    )

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    output_path = (
        output_directory
        / f"down_up_{image_name}.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Guardado: {output_path}"
    )


def main():

    output_directory = Path(
        "results/pregunta3/down_up"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image_name in ORIGINAL_IMAGES:

        create_down_up_figure(
            image_name,
            output_directory,
        )


if __name__ == "__main__":
    main()