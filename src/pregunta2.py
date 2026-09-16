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