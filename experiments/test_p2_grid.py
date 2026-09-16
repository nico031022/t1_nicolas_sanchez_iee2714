from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.pregunta2 import build_region_grid


def compute_coverage_map(
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

    return coverage_map


def check_region_bounds(
    image_shape,
    region_grid
):
    image_height = image_shape[0]
    image_width = image_shape[1]

    for region in region_grid:

        top = region["top"]
        bottom = region["bottom"]
        left = region["left"]
        right = region["right"]

        assert 0 <= top < bottom <= image_height
        assert 0 <= left < right <= image_width


def plot_grid_test(
    image_shape,
    region_size,
    region_step,
    test_name,
    output_directory
):
    region_grid = build_region_grid(
        image_shape=image_shape,
        region_size=region_size,
        region_step=region_step
    )

    check_region_bounds(
        image_shape,
        region_grid
    )

    coverage_map = compute_coverage_map(
        image_shape,
        region_grid
    )

    minimum_coverage = coverage_map.min()
    maximum_coverage = coverage_map.max()

    covered_pixels = np.count_nonzero(
        coverage_map > 0
    )

    total_pixels = coverage_map.size

    coverage_percentage = (
        100.0
        * covered_pixels
        / total_pixels
    )

    row_positions = sorted(
        {
            region["top"]
            for region in region_grid
        }
    )

    column_positions = sorted(
        {
            region["left"]
            for region in region_grid
        }
    )

    print()
    print("----------------------------------------")
    print(f"Test: {test_name}")
    print(f"Image shape: {image_shape}")
    print(f"Region size: {region_size}")
    print(f"Region step: {region_step}")
    print(f"Number of regions: {len(region_grid)}")

    print(
        "Row positions:",
        row_positions
    )

    print(
        "Column positions:",
        column_positions
    )

    print(
        "Minimum coverage:",
        minimum_coverage
    )

    print(
        "Maximum coverage:",
        maximum_coverage
    )

    print(
        "Covered pixels:",
        f"{coverage_percentage:.2f}%"
    )

    # no quedar ningun pixel fuera
    assert minimum_coverage >= 1

    image_test = np.full(
        image_shape,
        128,
        dtype=np.uint8
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].imshow(
        image_test,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    for region in region_grid:

        rectangle = Rectangle(
            (
                region["left"],
                region["top"]
            ),
            region["right"] - region["left"],
            region["bottom"] - region["top"],
            fill=False,
            linewidth=1.0
        )

        axes[0].add_patch(rectangle)

        axes[0].plot(
            region["center_x"],
            region["center_y"],
            "."
        )

    axes[0].set_title(
        (
            f"{test_name}\n"
            f"size={region_size}, "
            f"step={region_step}"
        )
    )

    axes[0].set_xlim(
        0,
        image_shape[1]
    )

    axes[0].set_ylim(
        image_shape[0],
        0
    )

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    image_coverage = axes[1].imshow(
        coverage_map
    )

    axes[1].set_title(
        "Numero de regiones por pixel"
    )

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    figure.colorbar(
        image_coverage,
        ax=axes[1]
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"{test_name}.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(figure)


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
        / "grid"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    image_shape = (
        250,
        370
    )

    test_configurations = [
        {
            "name": "grid_step_equal_size",
            "region_size": 64,
            "region_step": 64
        },
        {
            "name": "grid_overlap_50",
            "region_size": 64,
            "region_step": 32
        },
        {
            "name": "grid_overlap_strong",
            "region_size": 64,
            "region_step": 16
        },
        {
            "name": "grid_border_shift",
            "region_size": 96,
            "region_step": 70
        },
        {
            "name": "grid_single_region",
            "region_size": 500,
            "region_step": 500
        }
    ]

    for current_config in test_configurations:

        plot_grid_test(
            image_shape=image_shape,
            region_size=current_config[
                "region_size"
            ],
            region_step=current_config[
                "region_step"
            ],
            test_name=current_config[
                "name"
            ],
            output_directory=output_directory
        )

    print()
    print(
        "Todos los tests de cobertura "
        "terminaron correctamente."
    )

    print(
        "Resultados guardados en:",
        output_directory
    )


if __name__ == "__main__":
    main()