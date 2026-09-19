from pathlib import Path

import cv2
import matplotlib.pyplot as plt


SCALE_FACTORS = [
    0.6,
    0.8,
    1.3,
    1.7,
]


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


def calculate_scaled_crop(
    crop_region,
    original_shape,
    resized_shape,
):

    original_height = original_shape[0]
    original_width = original_shape[1]

    resized_height = resized_shape[0]
    resized_width = resized_shape[1]

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

    return top, bottom, left, right


def create_comparison(
    image_name,
    scale_factor,
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

    scale_text = str(
        scale_factor
    ).replace(".", "_")

    nearest_path = Path(
        "results/pregunta3/scaling"
    ) / image_name / (
        f"{image_name}_nearest_s{scale_text}.png"
    )

    bilinear_path = Path(
        "results/pregunta3/scaling"
    ) / image_name / (
        f"{image_name}_bilinear_s{scale_text}.png"
    )

    nearest_image = cv2.imread(
        str(nearest_path),
        cv2.IMREAD_COLOR,
    )

    bilinear_image = cv2.imread(
        str(bilinear_path),
        cv2.IMREAD_COLOR,
    )

    if nearest_image is None:
        raise ValueError(
            f"No se pudo leer {nearest_path}"
        )

    if bilinear_image is None:
        raise ValueError(
            f"No se pudo leer {bilinear_path}"
        )

    crop_region = CROP_REGIONS[
        image_name
    ]

    original_crop = original_image[
        crop_region["top"]:crop_region["bottom"],
        crop_region["left"]:crop_region["right"],
    ]

    top, bottom, left, right = (
        calculate_scaled_crop(
            crop_region,
            original_image.shape,
            nearest_image.shape,
        )
    )

    nearest_crop = nearest_image[
        top:bottom,
        left:right,
    ]

    bilinear_crop = bilinear_image[
        top:bottom,
        left:right,
    ]

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
        f"Original\n{original_crop.shape[1]} x "
        f"{original_crop.shape[0]}"
    )

    axes[1].imshow(
        nearest_crop_rgb,
        interpolation="nearest",
    )

    axes[1].set_title(
        f"Nearest, s = {scale_factor}\n"
        f"{nearest_crop.shape[1]} x "
        f"{nearest_crop.shape[0]}"
    )

    axes[2].imshow(
        bilinear_crop_rgb,
        interpolation="nearest",
    )

    axes[2].set_title(
        f"Bilinear, s = {scale_factor}\n"
        f"{bilinear_crop.shape[1]} x "
        f"{bilinear_crop.shape[0]}"
    )

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    output_path = (
        output_directory
        / (
            f"comparison_{image_name}_"
            f"s{scale_text}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"{image_name}, s = {scale_factor}: "
        f"{output_path}"
    )


def main():

    output_directory = Path(
        "results/pregunta3/comparison"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image_name in ORIGINAL_IMAGES:

        for scale_factor in SCALE_FACTORS:

            create_comparison(
                image_name,
                scale_factor,
                output_directory,
            )


if __name__ == "__main__":
    main()