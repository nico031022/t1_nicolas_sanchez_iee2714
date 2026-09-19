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

def nearest_neighbor_value(
    image,
    input_row,
    input_col,
):

    input_height = image.shape[0]
    input_width = image.shape[1]

    # buscamos el vecino mas cercano
    nearest_row = int(np.floor(input_row + 0.5))
    nearest_col = int(np.floor(input_col + 0.5))

    # dejar indices dentro de image
    nearest_row = np.clip(
        nearest_row,
        0,
        input_height - 1,
    )

    nearest_col = np.clip(
        nearest_col,
        0,
        input_width - 1,
    )

    pixel_value = image[
        nearest_row,
        nearest_col,
    ]

    return pixel_value


def resize_image(
    image,
    scale_factor,
    method="nearest",
):

    if image.ndim == 2:
        is_grayscale = True

    elif image.ndim == 3 and image.shape[2] == 3:
        is_grayscale = False

    else:
        raise ValueError(
            "La imagen debe ser grayscale o RGB"
        )

    output_height, output_width = calculate_output_size(
        image.shape,
        scale_factor,
    )

    if is_grayscale:
        output_image = np.zeros(
            (output_height, output_width),
            dtype=image.dtype,
        )

    else:
        output_image = np.zeros(
            (output_height, output_width, 3),
            dtype=image.dtype,
        )

    input_shape = image.shape[:2]
    output_shape = (
        output_height,
        output_width,
    )

    for output_row in range(output_height):
        for output_col in range(output_width):

            input_row, input_col = map_output_to_input(
                output_row,
                output_col,
                input_shape,
                output_shape,
            )

            if method == "nearest":

                pixel_value = nearest_neighbor_value(
                    image,
                    input_row,
                    input_col,
                )

            else:
                raise ValueError(
                    "Metodo de interpolacion no valido"
                )

            output_image[
                output_row,
                output_col,
            ] = pixel_value

    return output_image