from pathlib import Path
import time

import cv2

from src.pregunta3 import resize_image


SCALE_FACTORS = [
    0.6,
    0.8,
    1.3,
    1.7,
]

METHODS = [
    "nearest",
    "bilinear",
]


def run_scaling_experiment(
    image_path,
    image_name,
    output_directory,
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise ValueError(
            f"No se pudo leer la imagen: {image_path}"
        )

    image_output_directory = (
        output_directory / image_name
    )

    image_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nImagen: {image_name}"
    )

    print(
        f"Shape original: {image.shape}"
    )

    for scale_factor in SCALE_FACTORS:

        print(
            f"\nFactor s = {scale_factor}"
        )

        for method in METHODS:

            start_time = time.perf_counter()

            resized_image = resize_image(
                image,
                scale_factor,
                method,
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            scale_text = str(
                scale_factor
            ).replace(".", "_")

            output_path = (
                image_output_directory
                / (
                    f"{image_name}_"
                    f"{method}_"
                    f"s{scale_text}.png"
                )
            )

            cv2.imwrite(
                str(output_path),
                resized_image,
            )

            print(
                f"  {method}: "
                f"{resized_image.shape}, "
                f"{elapsed_time:.2f} s"
            )


def main():

    output_directory = Path(
        "results/pregunta3/scaling"
    )

    images = [
        (
            Path("images/profesor/pregunta3.tif"),
            "profesor",
        ),
        (
            Path("images/propias/fruit.jpg"),
            "fruit",
        ),
    ]

    for image_path, image_name in images:

        run_scaling_experiment(
            image_path,
            image_name,
            output_directory,
        )

if __name__ == "__main__":
    main()