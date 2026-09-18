from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.pregunta2 import (
    local_histogram_equalization
)


def compute_histogram_and_cdf(
    image_region,
    num_bins=256
):
    histogram, bin_edges = np.histogram(
        image_region,
        bins=num_bins,
        range=(0, 256)
    )

    cdf = np.cumsum(
        histogram
    )

    return (
        histogram,
        bin_edges,
        cdf
    )


def create_clahe_reference(
    image_gray,
    approximate_tile_size=64,
    clip_limit=2.0
):
    image_height = image_gray.shape[0]
    image_width = image_gray.shape[1]

    tiles_x = max(
        1,
        round(
            image_width
            / approximate_tile_size
        )
    )

    tiles_y = max(
        1,
        round(
            image_height
            / approximate_tile_size
        )
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(
            tiles_x,
            tiles_y
        )
    )

    clahe_image = clahe.apply(
        image_gray
    )

    return clahe_image


def crop_image(
    image_gray,
    crop_config
):
    x_start = crop_config["x"]
    y_start = crop_config["y"]
    crop_width = crop_config["width"]
    crop_height = crop_config["height"]

    x_end = x_start + crop_width
    y_end = y_start + crop_height

    image_crop = image_gray[
        y_start:y_end,
        x_start:x_end
    ]

    return image_crop


def analyze_crop_statistics(
    image_crop
):
    crop_min = int(
        image_crop.min()
    )

    crop_max = int(
        image_crop.max()
    )

    crop_mean = float(
        image_crop.mean()
    )

    crop_std = float(
        image_crop.std()
    )

    percentile_05 = float(
        np.percentile(
            image_crop,
            5
        )
    )

    percentile_95 = float(
        np.percentile(
            image_crop,
            95
        )
    )

    return {
        "min": crop_min,
        "max": crop_max,
        "mean": crop_mean,
        "std": crop_std,
        "p05": percentile_05,
        "p95": percentile_95
    }


def run_bad_case_experiment(
    image_path,
    image_name,
    crop_config,
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

    unlimited_image = (
        local_histogram_equalization(
            image_gray,
            region_size=region_size,
            region_step=region_step,
            num_bins=num_bins,
            contrast_strength=1.0
        )
    )

    controlled_image = (
        local_histogram_equalization(
            image_gray,
            region_size=region_size,
            region_step=region_step,
            num_bins=num_bins,
            contrast_strength=0.50
        )
    )

    clahe_image = create_clahe_reference(
        image_gray,
        approximate_tile_size=64,
        clip_limit=2.0
    )

    original_crop = crop_image(
        image_gray,
        crop_config
    )

    unlimited_crop = crop_image(
        unlimited_image,
        crop_config
    )

    controlled_crop = crop_image(
        controlled_image,
        crop_config
    )

    clahe_crop = crop_image(
        clahe_image,
        crop_config
    )

    crop_statistics = analyze_crop_statistics(
        original_crop
    )

    histogram, bin_edges, cdf = (
        compute_histogram_and_cdf(
            original_crop,
            num_bins=256
        )
    )

    print()
    print("----------------------------------------")
    print(
        f"Bad case / homogeneous region: {image_name}"
    )
    print("----------------------------------------")

    print(
        "Crop:",
        crop_config
    )

    print(
        "Original crop statistics:"
    )

    print(
        f"min={crop_statistics['min']}, "
        f"max={crop_statistics['max']}, "
        f"mean={crop_statistics['mean']:.3f}, "
        f"std={crop_statistics['std']:.3f}"
    )

    print(
        f"p05={crop_statistics['p05']:.3f}, "
        f"p95={crop_statistics['p95']:.3f}"
    )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8)
    )

    axes[0, 0].imshow(
        image_gray,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    rectangle = Rectangle(
        (
            crop_config["x"],
            crop_config["y"]
        ),
        crop_config["width"],
        crop_config["height"],
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    axes[0, 0].add_patch(
        rectangle
    )

    axes[0, 0].set_title(
        "Original + crop"
    )

    axes[0, 1].imshow(
        original_crop,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 1].set_title(
        "Crop original"
    )

    axes[0, 2].imshow(
        unlimited_crop,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 2].set_title(
        "Crop unlimited"
    )

    axes[0, 3].imshow(
        controlled_crop,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 3].set_title(
        "Crop alpha=0.50"
    )

    axes[1, 0].imshow(
        clahe_crop,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 0].set_title(
        "Crop CLAHE"
    )

    axes[1, 1].bar(
        bin_edges[:-1],
        histogram,
        width=1.0
    )

    axes[1, 1].set_title(
        "Histograma del crop"
    )

    axes[1, 1].set_xlim(
        0,
        255
    )

    axes[1, 2].plot(
        cdf
    )

    axes[1, 2].set_title(
        "CDF del crop"
    )

    axes[1, 2].set_xlim(
        0,
        255
    )

    unlimited_difference = np.abs(
        unlimited_crop.astype(np.int16)
        - original_crop.astype(np.int16)
    )

    controlled_difference = np.abs(
        controlled_crop.astype(np.int16)
        - original_crop.astype(np.int16)
    )

    clahe_difference = np.abs(
        clahe_crop.astype(np.int16)
        - original_crop.astype(np.int16)
    )

    axes[1, 3].plot(
        unlimited_difference.mean(axis=0),
        label="Unlimited"
    )

    axes[1, 3].plot(
        controlled_difference.mean(axis=0),
        label="Alpha=0.50"
    )

    axes[1, 3].plot(
        clahe_difference.mean(axis=0),
        label="CLAHE"
    )

    axes[1, 3].set_title(
        "Cambio medio por columna"
    )

    axes[1, 3].legend()

    for axis_row in axes:
        for axis in axis_row:
            axis.axis("off")

    axes[1, 1].axis("on")
    axes[1, 2].axis("on")
    axes[1, 3].axis("on")

    figure.suptitle(
        image_name
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"bad_case_{image_name}.png"
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
        / "bad_case"
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
            ),
            "crop": {
                "x": 1040,
                "y": 120,
                "width": 320,
                "height": 220
            }
        },
        {
            "name": "fruit",
            "path": (
                project_directory
                / "images"
                / "propias"
                / "fruit.jpg"
            ),
            "crop": {
                "x": 1184,
                "y": 800,
                "width": 64,
                "height": 64
            }
        }
    ]

    for current_image in image_configurations:

        run_bad_case_experiment(
            image_path=current_image[
                "path"
            ],
            image_name=current_image[
                "name"
            ],
            crop_config=current_image[
                "crop"
            ],
            output_directory=output_directory
        )


if __name__ == "__main__":
    main()