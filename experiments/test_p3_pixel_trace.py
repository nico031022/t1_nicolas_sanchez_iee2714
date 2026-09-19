import cv2
import numpy as np

from src.pregunta3 import (
    bilinear_value,
    calculate_output_size,
    map_output_to_input,
)


SCALE_FACTOR = 1.7

OUTPUT_ROW = 2380
OUTPUT_COL = 510


def main():

    image_bgr = cv2.imread(
        "images/profesor/pregunta3.tif",
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            "No se pudo leer la imagen"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    output_height, output_width = (
        calculate_output_size(
            image_rgb.shape,
            SCALE_FACTOR,
        )
    )

    output_shape = (
        output_height,
        output_width,
    )

    input_shape = image_rgb.shape[:2]

    input_row, input_col = (
        map_output_to_input(
            OUTPUT_ROW,
            OUTPUT_COL,
            input_shape,
            output_shape,
        )
    )

    # los cuatro vecinos
    top_row = int(
        np.floor(input_row)
    )

    left_col = int(
        np.floor(input_col)
    )

    bottom_row = min(
        top_row + 1,
        image_rgb.shape[0] - 1,
    )

    right_col = min(
        left_col + 1,
        image_rgb.shape[1] - 1,
    )

    row_distance = (
        input_row - top_row
    )

    col_distance = (
        input_col - left_col
    )

    # pesos
    weight_top_left = (
        (1.0 - row_distance)
        * (1.0 - col_distance)
    )

    weight_top_right = (
        (1.0 - row_distance)
        * col_distance
    )

    weight_bottom_left = (
        row_distance
        * (1.0 - col_distance)
    )

    weight_bottom_right = (
        row_distance
        * col_distance
    )

    top_left = image_rgb[
        top_row,
        left_col,
    ]

    top_right = image_rgb[
        top_row,
        right_col,
    ]

    bottom_left = image_rgb[
        bottom_row,
        left_col,
    ]

    bottom_right = image_rgb[
        bottom_row,
        right_col,
    ]

    interpolated_value = bilinear_value(
        image_rgb,
        input_row,
        input_col,
    )

    final_value = np.rint(
        interpolated_value
    )

    final_value = np.clip(
        final_value,
        0,
        255,
    ).astype(np.uint8)

    weight_sum = (
        weight_top_left
        + weight_top_right
        + weight_bottom_left
        + weight_bottom_right
    )

    print(
        f"Scale factor: {SCALE_FACTOR}"
    )

    print(
        f"Output shape: {output_shape}"
    )

    print(
        f"Output coordinate: "
        f"({OUTPUT_ROW}, {OUTPUT_COL})"
    )

    print(
        "\nInput coordinate:"
    )

    print(
        f"row = {input_row:.6f}"
    )

    print(
        f"col = {input_col:.6f}"
    )

    print(
        "\nNeighbors:"
    )

    print(
        f"top-left     "
        f"({top_row}, {left_col}) "
        f"RGB = {top_left}"
    )

    print(
        f"top-right    "
        f"({top_row}, {right_col}) "
        f"RGB = {top_right}"
    )

    print(
        f"bottom-left  "
        f"({bottom_row}, {left_col}) "
        f"RGB = {bottom_left}"
    )

    print(
        f"bottom-right "
        f"({bottom_row}, {right_col}) "
        f"RGB = {bottom_right}"
    )

    print(
        "\nWeights:"
    )

    print(
        f"w_top_left = "
        f"{weight_top_left:.6f}"
    )

    print(
        f"w_top_right = "
        f"{weight_top_right:.6f}"
    )

    print(
        f"w_bottom_left = "
        f"{weight_bottom_left:.6f}"
    )

    print(
        f"w_bottom_right = "
        f"{weight_bottom_right:.6f}"
    )

    print(
        f"sum = {weight_sum:.6f}"
    )

    print(
        "\nInterpolated RGB:"
    )

    print(
        interpolated_value
    )

    print(
        "\nFinal RGB:"
    )

    print(
        final_value
    )


if __name__ == "__main__":
    main()