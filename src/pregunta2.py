import numpy as np


def build_region_positions(
    image_length,
    region_size,
    region_step
):
    if image_length <= 0:
        raise ValueError("image_length debe ser positivo")

    if region_size <= 0:
        raise ValueError("region_size debe ser positivo")

    if region_step <= 0:
        raise ValueError("region_step debe ser positivo")

    if region_step > region_size:
        raise ValueError(
            "region_step no puede ser mayor que region_size"
        )

    effective_region_size = min(
        region_size,
        image_length
    )

    # una sola region cubre todo el eje
    if effective_region_size == image_length:
        return [0]

    last_position = (
        image_length
        - effective_region_size
    )

    positions = list(
        range(
            0,
            last_position + 1,
            region_step
        )
    )

    # ajustar la ultima region al borde
    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


def build_region_grid(
    image_shape,
    region_size,
    region_step
):
    if len(image_shape) < 2:
        raise ValueError(
            "image_shape debe tener al menos dos dimensiones"
        )

    image_height = image_shape[0]
    image_width = image_shape[1]

    row_positions = build_region_positions(
        image_height,
        region_size,
        region_step
    )

    column_positions = build_region_positions(
        image_width,
        region_size,
        region_step
    )

    effective_region_height = min(
        region_size,
        image_height
    )

    effective_region_width = min(
        region_size,
        image_width
    )

    region_grid = []

    region_number = 0

    for region_row, top in enumerate(row_positions):

        bottom = (
            top
            + effective_region_height
        )

        for region_column, left in enumerate(
            column_positions
        ):

            right = (
                left
                + effective_region_width
            )

            center_y = (
                top
                + bottom
                - 1
            ) / 2.0

            center_x = (
                left
                + right
                - 1
            ) / 2.0

            region = {
                "number": region_number,
                "row": region_row,
                "column": region_column,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "center_y": center_y,
                "center_x": center_x
            }

            region_grid.append(region)

            region_number += 1

    return region_grid




def compute_region_histogram(
    region,
    num_bins=256
):
    if num_bins <= 0:
        raise ValueError(
            "num_bins debe ser positivo"
        )

    histogram, bin_edges = np.histogram(
        region,
        bins=num_bins,
        range=(0, 256)
    )

    return histogram, bin_edges


def compute_region_cdf(histogram):
    cdf = np.cumsum(
        histogram
    )

    return cdf


def find_cdf_min(cdf):
    nonzero_values = cdf[
        cdf > 0
    ]

    if len(nonzero_values) == 0:
        return 0

    cdf_min = nonzero_values[0]

    return cdf_min


def intensity_to_bin(
    intensity,
    num_bins
):
    bin_index = (
        int(intensity)
        * num_bins
        // 256
    )

    bin_index = min(
        bin_index,
        num_bins - 1
    )

    return bin_index


def build_region_mapping(
    histogram,
    num_bins
):
    cdf = compute_region_cdf(
        histogram
    )

    num_pixels = int(
        cdf[-1]
    )

    cdf_min = find_cdf_min(
        cdf
    )

    mapping = np.arange(
        256,
        dtype=np.float64
    )

    # vacia
    if num_pixels == 0:
        return mapping

    denominator = (
        num_pixels
        - cdf_min
    )

    # un bin ocupado no puede ecualizar
    if denominator == 0:
        return mapping

    for intensity in range(256):

        bin_index = intensity_to_bin(
            intensity,
            num_bins
        )

        cdf_value = cdf[
            bin_index
        ]

        mapped_value = (
            (
                cdf_value
                - cdf_min
            )
            / denominator
            * 255.0
        )

        mapped_value = np.clip(
            mapped_value,
            0.0,
            255.0
        )

        mapping[
            intensity
        ] = mapped_value

    return mapping


def apply_contrast_control(
    region_mapping,
    contrast_strength
):
    if not 0.0 <= contrast_strength <= 1.0:
        raise ValueError(
            "contrast_strength debe estar entre 0 y 1"
        )

    identity_mapping = np.arange(
        256,
        dtype=np.float64
    )

    if contrast_strength == 0.0:
        return identity_mapping

    if contrast_strength == 1.0:
        return region_mapping.copy()

    # acercar el mapping a la identidad
    controlled_mapping = (
        identity_mapping
        + contrast_strength
        * (
            region_mapping
            - identity_mapping
        )
    )

    return controlled_mapping



def apply_intensity_mapping(
    image_gray,
    mapping
):
    mapped_image = mapping[
        image_gray
    ]

    mapped_image = np.rint(
        mapped_image
    )

    mapped_image = np.clip(
        mapped_image,
        0,
        255
    )

    mapped_image = mapped_image.astype(
        np.uint8
    )

    return mapped_image


def global_equalization_own(
    image_gray,
    num_bins=256
):
    histogram, _ = compute_region_histogram(
        image_gray,
        num_bins=num_bins
    )

    mapping = build_region_mapping(
        histogram,
        num_bins=num_bins
    )

    equalized_image = apply_intensity_mapping(
        image_gray,
        mapping
    )

    return equalized_image


def compute_region_weights(region):
    region_height = (
        region["bottom"]
        - region["top"]
    )

    region_width = (
        region["right"]
        - region["left"]
    )

    local_y = np.arange(
        region_height,
        dtype=np.float64
    )

    local_x = np.arange(
        region_width,
        dtype=np.float64
    )

    center_y = (
        region_height
        - 1
    ) / 2.0

    center_x = (
        region_width
        - 1
    ) / 2.0

    half_height = (
        region_height
        / 2.0
    )

    half_width = (
        region_width
        / 2.0
    )

    # mas peso cerca del centro
    weight_y = (
        1.0
        - np.abs(
            local_y
            - center_y
        )
        / half_height
    )

    weight_x = (
        1.0
        - np.abs(
            local_x
            - center_x
        )
        / half_width
    )

    weight_y = np.clip(
        weight_y,
        0.0,
        1.0
    )

    weight_x = np.clip(
        weight_x,
        0.0,
        1.0
    )

    region_weights = (
        weight_y[:, None]
        * weight_x[None, :]
    )

    return region_weights


def local_histogram_equalization(
    image_gray,
    region_size,
    region_step,
    num_bins=256,
    contrast_strength=1.0
):
    if image_gray.ndim != 2:
        raise ValueError(
            "image_gray debe ser una imagen 2D"
        )

    if image_gray.dtype != np.uint8:
        raise ValueError(
            "image_gray debe tener dtype uint8"
        )

    if not 0.0 <= contrast_strength <= 1.0:
        raise ValueError(
            "contrast_strength debe estar entre 0 y 1"
        )

    region_grid = build_region_grid(
        image_shape=image_gray.shape,
        region_size=region_size,
        region_step=region_step
    )

    accumulated_values = np.zeros(
        image_gray.shape,
        dtype=np.float64
    )

    accumulated_weights = np.zeros(
        image_gray.shape,
        dtype=np.float64
    )

    for region in region_grid:

        top = region["top"]
        bottom = region["bottom"]
        left = region["left"]
        right = region["right"]

        region_image = image_gray[
            top:bottom,
            left:right
        ]

        histogram, _ = (
            compute_region_histogram(
                region_image,
                num_bins=num_bins
            )
        )

        region_mapping = build_region_mapping(
            histogram,
            num_bins=num_bins
        )

        controlled_mapping = (
            apply_contrast_control(
                region_mapping,
                contrast_strength
            )
        )

        mapped_region = controlled_mapping[
            region_image
        ]

        region_weights = (
            compute_region_weights(
                region
            )
        )

        # acumular las regiones que se solapan
        accumulated_values[
            top:bottom,
            left:right
        ] += (
            mapped_region
            * region_weights
        )

        accumulated_weights[
            top:bottom,
            left:right
        ] += region_weights

    if np.any(
        accumulated_weights <= 0
    ):
        raise RuntimeError(
            "hay pixels sin peso acumulado"
        )

    output_image = (
        accumulated_values
        / accumulated_weights
    )

    output_image = np.rint(
        output_image
    )

    output_image = np.clip(
        output_image,
        0,
        255
    )

    output_image = output_image.astype(
        np.uint8
    )

    return output_image