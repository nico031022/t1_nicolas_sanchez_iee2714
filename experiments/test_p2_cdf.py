import numpy as np

from src.pregunta2 import (
    build_region_mapping,
    compute_region_cdf,
    compute_region_histogram,
    find_cdf_min,
    intensity_to_bin
)


def print_array(
    name,
    values
):
    print()
    print(name)
    print(values)


def main():
    region_test = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 2, 2],
            [2, 2, 3, 3],
            [3, 3, 3, 3]
        ],
        dtype=np.uint8
    )

    print("----------------------------------------")
    print("Test 1: histogram con 4 bins")
    print("----------------------------------------")

    histogram, bin_edges = (
        compute_region_histogram(
            region_test,
            num_bins=4
        )
    )

    cdf = compute_region_cdf(
        histogram
    )

    cdf_min = find_cdf_min(
        cdf
    )

    print_array(
        "Region:",
        region_test
    )

    print_array(
        "Histogram:",
        histogram
    )

    print_array(
        "Bin edges:",
        bin_edges
    )

    print_array(
        "CDF:",
        cdf
    )

    print()
    print(
        "CDF min:",
        cdf_min
    )

    print()
    print("----------------------------------------")
    print("Test 2: conversion intensidad -> bin")
    print("----------------------------------------")

    test_intensities = [
        0,
        1,
        63,
        64,
        127,
        128,
        191,
        192,
        255
    ]

    for intensity in test_intensities:

        bin_index = intensity_to_bin(
            intensity,
            num_bins=4
        )

        print(
            f"Intensity {intensity:3d}"
            f" -> bin {bin_index}"
        )

    print()
    print("----------------------------------------")
    print("Test 3: mapping con 256 bins")
    print("----------------------------------------")

    histogram_256, _ = (
        compute_region_histogram(
            region_test,
            num_bins=256
        )
    )

    mapping_256 = build_region_mapping(
        histogram_256,
        num_bins=256
    )

    used_intensities = [
        0,
        1,
        2,
        3
    ]

    for intensity in used_intensities:

        print(
            f"{intensity:3d}"
            f" -> "
            f"{mapping_256[intensity]:.3f}"
        )

    print()
    print("----------------------------------------")
    print("Test 4: region uniforme")
    print("----------------------------------------")

    uniform_region = np.full(
        (8, 8),
        120,
        dtype=np.uint8
    )

    histogram_uniform, _ = (
        compute_region_histogram(
            uniform_region,
            num_bins=256
        )
    )

    mapping_uniform = build_region_mapping(
        histogram_uniform,
        num_bins=256
    )

    print(
        "120 ->",
        mapping_uniform[120]
    )

    assert (
        mapping_uniform[120]
        == 120
    )

    print()
    print(
        "La region uniforme mantiene "
        "la identidad correctamente."
    )

    print()
    print("----------------------------------------")
    print("Test 5: propiedades basicas")
    print("----------------------------------------")

    assert len(
        mapping_256
    ) == 256

    assert np.all(
        mapping_256 >= 0
    )

    assert np.all(
        mapping_256 <= 255
    )

    assert np.all(
        np.diff(mapping_256) >= 0
    )

    print(
        "Mapping de longitud 256: OK"
    )

    print(
        "Mapping dentro de [0, 255]: OK"
    )

    print(
        "Mapping monotono: OK"
    )

    print()
    print(
        "Todos los tests terminaron "
        "correctamente."
    )


if __name__ == "__main__":
    main()