from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.pregunta2 import (
    build_region_grid,
    local_histogram_equalization
)


def compute_average_coverage(
    image_shape,
    region_grid
):
    coverage_map = np.zeros(
        image_shape,
        dtype=np.int32
    )

    for region in region_grid:

        top = region["top"]
        bottom = region["bottom"]
        left = region["left"]
        right = region["right"]

        coverage_map[
            top:bottom,
            left:right
        ] += 1

    average_coverage = (
        coverage_map.mean()
    )

    return average_coverage


def measure_execution_time(
    image_gray,
    region_size,
    region_step,
    num_bins,
    num_repetitions
):
    execution_times = []

    for repetition in range(
        num_repetitions
    ):
        start_time = (
            time.perf_counter()
        )

        local_histogram_equalization(
            image_gray,
            region_size=region_size,
            region_step=region_step,
            num_bins=num_bins,
            contrast_strength=1.0
        )

        end_time = (
            time.perf_counter()
        )

        elapsed_time = (
            end_time
            - start_time
        )

        execution_times.append(
            elapsed_time
        )

    median_time = float(
        np.median(
            execution_times
        )
    )

    return (
        execution_times,
        median_time
    )


def main():
    project_directory = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    image_path = (
        project_directory
        / "images"
        / "profesor"
        / "pregunta2.tif"
    )

    output_directory = (
        project_directory
        / "results"
        / "pregunta2"
        / "runtime"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    image_gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image_gray is None:
        raise FileNotFoundError(
            f"No se pudo leer: {image_path}"
        )

    region_size = 128

    region_steps = [
        128,
        96,
        64,
        32
    ]

    num_bins = 256
    num_repetitions = 3

    region_counts = []
    average_coverages = []
    median_times = []

    print("----------------------------------------")
    print("Costo computacional")
    print("----------------------------------------")

    print(
        "Image shape:",
        image_gray.shape
    )

    print(
        "Region size:",
        region_size
    )

    print(
        "Repetitions:",
        num_repetitions
    )

    for region_step in region_steps:

        region_grid = build_region_grid(
            image_shape=image_gray.shape,
            region_size=region_size,
            region_step=region_step
        )

        num_regions = len(
            region_grid
        )

        average_coverage = (
            compute_average_coverage(
                image_gray.shape,
                region_grid
            )
        )

        (
            execution_times,
            median_time
        ) = measure_execution_time(
            image_gray=image_gray,
            region_size=region_size,
            region_step=region_step,
            num_bins=num_bins,
            num_repetitions=num_repetitions
        )

        overlap_percentage = (
            100.0
            * (
                region_size
                - region_step
            )
            / region_size
        )

        region_counts.append(
            num_regions
        )

        average_coverages.append(
            average_coverage
        )

        median_times.append(
            median_time
        )

        formatted_times = [
            f"{current_time:.3f}"
            for current_time in execution_times
        ]

        print()
        print(
            f"step={region_step}"
            f" | overlap="
            f"{overlap_percentage:.1f}%"
        )

        print(
            "regions:",
            num_regions
        )

        print(
            "average coverage:",
            f"{average_coverage:.2f}"
        )

        print(
            "times:",
            formatted_times
        )

        print(
            "median time:",
            f"{median_time:.3f} s"
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        region_counts,
        median_times,
        marker="o"
    )

    axes[0].set_xlabel(
        "Numero de regiones"
    )

    axes[0].set_ylabel(
        "Tiempo mediano [s]"
    )

    axes[0].set_title(
        "Costo vs numero de regiones"
    )

    axes[0].grid()

    axes[1].plot(
        average_coverages,
        median_times,
        marker="o"
    )

    axes[1].set_xlabel(
        "Cobertura promedio por pixel"
    )

    axes[1].set_ylabel(
        "Tiempo mediano [s]"
    )

    axes[1].set_title(
        "Costo vs overlap de la malla"
    )

    axes[1].grid()

    figure.suptitle(
        (
            "Pregunta 2 - costo computacional\n"
            f"region_size={region_size}, "
            f"num_bins={num_bins}"
        )
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / "runtime_mesh_density.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(figure)

    print()
    print(
        "Resultado guardado en:",
        output_path
    )


if __name__ == "__main__":
    main()