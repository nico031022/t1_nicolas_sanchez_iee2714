from pathlib import Path

import cv2
import matplotlib.pyplot as plt


SCALE_FACTORS = [
    0.8,
    0.6,
]


CROP_REGION = {
    "top": 1220,
    "bottom": 1740,
    "left": 0,
    "right": 650,
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


def load_resized_crop(
    method,
    scale_factor,
    original_shape,
):

    scale_text = str(
        scale_factor
    ).replace(".", "_")

    image_path = (
        Path("results/pregunta3/scaling/profesor")
        / (
            f"profesor_{method}_"
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

    top, bottom, left, right = (
        calculate_scaled_crop(
            CROP_REGION,
            original_shape,
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


def create_aliasing_comparison():

    original_path = Path(
        "images/profesor/pregunta3.tif"
    )

    original_image = cv2.imread(
        str(original_path),
        cv2.IMREAD_COLOR,
    )

    if original_image is None:
        raise ValueError(
            f"No se pudo leer {original_path}"
        )

    original_crop = original_image[
        CROP_REGION["top"]:CROP_REGION["bottom"],
        CROP_REGION["left"]:CROP_REGION["right"],
    ]

    original_crop_rgb = cv2.cvtColor(
        original_crop,
        cv2.COLOR_BGR2RGB,
    )

    nearest_08 = load_resized_crop(
        "nearest",
        0.8,
        original_image.shape,
    )

    bilinear_08 = load_resized_crop(
        "bilinear",
        0.8,
        original_image.shape,
    )

    nearest_06 = load_resized_crop(
        "nearest",
        0.6,
        original_image.shape,
    )

    bilinear_06 = load_resized_crop(
        "bilinear",
        0.6,
        original_image.shape,
    )

    images = [
        original_crop_rgb,
        nearest_08,
        bilinear_08,
        nearest_06,
        bilinear_06,
    ]

    titles = [
        "Original",
        "Nearest, s = 0.8",
        "Bilinear, s = 0.8",
        "Nearest, s = 0.6",
        "Bilinear, s = 0.6",
    ]

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(22, 6),
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

        axis.set_title(
            title
        )

        axis.axis("off")

    figure.tight_layout()

    output_directory = Path(
        "results/pregunta3/aliasing"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "aliasing_profesor.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Crop original: {original_crop.shape}"
    )

    print(
        f"Guardado: {output_path}"
    )


def main():

    create_aliasing_comparison()


if __name__ == "__main__":
    main()