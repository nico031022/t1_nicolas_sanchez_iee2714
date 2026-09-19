from pathlib import Path

import cv2
import matplotlib.pyplot as plt


REDUCTION_FACTORS = [
    0.6,
    0.8,
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


def load_crop(
    image_name,
    method,
    scale_factor,
    original_image,
):

    scale_text = str(
        scale_factor
    ).replace(".", "_")

    image_path = (
        Path("results/pregunta3/scaling")
        / image_name
        / (
            f"{image_name}_{method}_"
            f"s{scale_text}.png"
        )
    )

    resized_image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if resized_image is None:
        raise ValueError(
            f"No se pudo leer {image_path}"
        )

    crop_region = CROP_REGIONS[
        image_name
    ]

    top, bottom, left, right = (
        calculate_scaled_crop(
            crop_region,
            original_image.shape,
            resized_image.shape,
        )
    )

    crop = resized_image[
        top:bottom,
        left:right,
    ]

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )

    return crop_rgb


def create_reduction_comparison(
    image_name,
    output_directory,
):

    original_image = cv2.imread(
        str(ORIGINAL_IMAGES[image_name]),
        cv2.IMREAD_COLOR,
    )

    if original_image is None:
        raise ValueError(
            f"No se pudo leer la imagen {image_name}"
        )

    crop_region = CROP_REGIONS[
        image_name
    ]

    original_crop = original_image[
        crop_region["top"]:crop_region["bottom"],
        crop_region["left"]:crop_region["right"],
    ]

    original_crop_rgb = cv2.cvtColor(
        original_crop,
        cv2.COLOR_BGR2RGB,
    )

    nearest_06 = load_crop(
        image_name,
        "nearest",
        0.6,
        original_image,
    )

    bilinear_06 = load_crop(
        image_name,
        "bilinear",
        0.6,
        original_image,
    )

    nearest_08 = load_crop(
        image_name,
        "nearest",
        0.8,
        original_image,
    )

    bilinear_08 = load_crop(
        image_name,
        "bilinear",
        0.8,
        original_image,
    )

    images = [
        original_crop_rgb,
        nearest_06,
        bilinear_06,
        nearest_08,
        bilinear_08,
    ]

    titles = [
        "Original",
        "Nearest, s = 0.6",
        "Bilinear, s = 0.6",
        "Nearest, s = 0.8",
        "Bilinear, s = 0.8",
    ]

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5),
    )

    for axis, image, title in zip(
        axes,
        images,
        titles,
    ):

        axis.imshow(
            image,
            interpolation="nearest",
        )

        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()

    output_path = (
        output_directory
        / f"reduction_{image_name}.png"
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
        "results/pregunta3/reduction"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image_name in ORIGINAL_IMAGES:

        create_reduction_comparison(
            image_name,
            output_directory,
        )


if __name__ == "__main__":
    main()