import numpy as np


def calculate_output_size(image_shape, scale_factor):

    if scale_factor < 0.5 or scale_factor > 2.0:
        raise ValueError("scale_factor debe estar entre 0.5 y 2.0")

    input_height = image_shape[0]
    input_width = image_shape[1]

    output_height_float = input_height * scale_factor
    output_width_float = input_width * scale_factor

    output_height = int(np.floor(output_height_float + 0.5))
    output_width = int(np.floor(output_width_float + 0.5))

    return output_height, output_width


def map_output_to_input(
    output_row,
    output_col,
    input_shape,
    output_shape,
):

    input_height = input_shape[0]
    input_width = input_shape[1]

    output_height = output_shape[0]
    output_width = output_shape[1]

    scale_y = output_height / input_height
    scale_x = output_width / input_width

    input_row = (output_row + 0.5) / scale_y - 0.5
    input_col = (output_col + 0.5) / scale_x - 0.5

    # dejar la coordenada dentro de image
    input_row = np.clip(
        input_row,
        0.0,
        input_height - 1,
    )

    input_col = np.clip(
        input_col,
        0.0,
        input_width - 1,
    )

    return input_row, input_col