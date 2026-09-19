from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta3 import resize_image


SUCCESSIVE_FACTORS = [
    0.8,
    1.25,
    1.1,
]


CROP_REGION = {
    "top": 250,
    "bottom": 620,
    "left": 620,
    "right": 1080,
}


def calculate_equivalent_factor():

    equivalent_factor = 1.0

    for scale_factor in SUCCESSIVE_FACTORS:
        equivalent_factor *= scale_factor

    return equivalent_factor


def apply_successive_resizing(
    image,
    method,
):

    current_image = image

    intermediate_shapes = []

    for scale_factor in SUCCESSIVE_FACTORS:

        current_image = resize_image(
            current_image,
            scale_factor,
            method,
        )

        intermediate_shapes.append(
            current_image.shape
        )

    return current_image, intermediate_shapes


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


def crop_original(
    image,
    crop_region,
):

    crop = image[
        crop_region["top"]:crop_region["bottom"],
        crop_region["left"]:crop_region["right"],
    ]

    return crop


def crop_resized(
    image,
    crop_region,
    original_shape,
):

    original_height = original_shape[0]
    original_width = original_shape[1]

    resized_height = image.shape[0]
    resized_width = image.shape[1]

    scale_y = resized_height / original_height
    scale_x = resized_width / original_width

    top = int(
        round(crop_region["top"] * scale_y)
    )

    bottom = int(
        round(crop_region["bottom"] * scale_y)
    )

    left = int(
        round(crop_region["left"] * scale_x)
    )

    right = int(
        round(crop_region["right"] * scale_x)
    )

    crop = image[
        top:bottom,
        left:right,
    ]

    return crop


def create_successive_comparison():

    image_path = Path(
        "images/propias/fruit.jpg"
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"No se pudo leer {image_path}"
        )

    equivalent_factor = (
        calculate_equivalent_factor()
    )

    print(
        f"Shape original: {image.shape}"
    )

    print(
        f"Factores sucesivos: {SUCCESSIVE_FACTORS}"
    )

    print(
        f"Factor equivalente: {equivalent_factor:.6f}"
    )

    results = {}

    for method in [
        "nearest",
        "bilinear",
    ]:

        successive_start = time.perf_counter()

        successive_image, intermediate_shapes = (
            apply_successive_resizing(
                image,
                method,
            )
        )

        successive_time = (
            time.perf_counter()
            - successive_start
        )

        direct_start = time.perf_counter()

        direct_image = resize_image(
            image,
            equivalent_factor,
            method,
        )

        direct_time = (
            time.perf_counter()
            - direct_start
        )

        mean_difference = (
            calculate_mean_absolute_difference(
                successive_image,
                direct_image,
            )
        )

        results[method] = {
            "successive": successive_image,
            "direct": direct_image,
        }

        print(
            f"\nMetodo: {method}"
        )

        print(
            f"Shapes intermedios: "
            f"{intermediate_shapes}"
        )

        print(
            f"Shape sucesivo: "
            f"{successive_image.shape}"
        )

        print(
            f"Shape directo: "
            f"{direct_image.shape}"
        )

        print(
            f"Diferencia media absoluta: "
            f"{mean_difference:.4f}"
        )

        print(
            f"Tiempo sucesivo: "
            f"{successive_time:.2f} s"
        )

        print(
            f"Tiempo directo: "
            f"{direct_time:.2f} s"
        )

    original_crop = crop_original(
        image,
        CROP_REGION,
    )

    nearest_direct_crop = crop_resized(
        results["nearest"]["direct"],
        CROP_REGION,
        image.shape,
    )

    nearest_successive_crop = crop_resized(
        results["nearest"]["successive"],
        CROP_REGION,
        image.shape,
    )

    bilinear_direct_crop = crop_resized(
        results["bilinear"]["direct"],
        CROP_REGION,
        image.shape,
    )

    bilinear_successive_crop = crop_resized(
        results["bilinear"]["successive"],
        CROP_REGION,
        image.shape,
    )

    images = [
        original_crop,
        nearest_direct_crop,
        nearest_successive_crop,
        bilinear_direct_crop,
        bilinear_successive_crop,
    ]

    titles = [
        "Original",
        "Nearest\nDirect s = 1.1",
        "Nearest\nSuccessive",
        "Bilinear\nDirect s = 1.1",
        "Bilinear\nSuccessive",
    ]

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5),
    )

    for axis, current_image, title in zip(
        axes,
        images,
        titles,
    ):

        current_image_rgb = cv2.cvtColor(
            current_image,
            cv2.COLOR_BGR2RGB,
        )

        axis.imshow(
            current_image_rgb,
            interpolation="nearest",
        )

        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()

    output_directory = Path(
        "results/pregunta3/successive"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "successive_fruit.png"
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


def main():

    create_successive_comparison()


if __name__ == "__main__":
    main()