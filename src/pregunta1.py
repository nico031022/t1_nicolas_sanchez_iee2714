import numpy as np

# Conversiones de color
# con la ayuda de la capsula de la Semana 3


def prepare_rgb_image(image_rgb):
    image_array = np.asarray(image_rgb)

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("La imagen debe tener 3 canales RGB.")

    # uint8
    if np.issubdtype(image_array.dtype, np.floating):
        if image_array.max() <= 1.0:
            image_array = image_array * 255.0

    image_array = np.clip(image_array, 0, 255)

    return np.round(image_array).astype(np.uint8)


def rgb_to_hsv(image_rgb):
    image_float = image_rgb.astype(np.float64)
    image_float = image_float / 255.0

    red = image_float[:, :, 0]
    green = image_float[:, :, 1]
    blue = image_float[:, :, 2]

    epsilon = 1e-10

    colour_max = np.maximum(
        np.maximum(red, green),
        blue
    )

    colour_min = np.minimum(
        np.minimum(red, green),
        blue
    )

    delta = colour_max - colour_min

    value = colour_max

    saturation = np.where(
        colour_max > epsilon,
        delta / (value + epsilon),
        0.0
    )

    hue = np.zeros_like(value)

    # check cual canal domina
    mask_red = (colour_max == red) & (delta > epsilon)
    mask_green = (colour_max == green) & (delta > epsilon)
    mask_blue = (colour_max == blue) & (delta > epsilon)

    hue[mask_red] = 60.0 * (
        (
            (green[mask_red] - blue[mask_red])
            / delta[mask_red]
        ) % 6
    )

    hue[mask_green] = 60.0 * (
        (
            (blue[mask_green] - red[mask_green])
            / delta[mask_green]
        ) + 2
    )

    hue[mask_blue] = 60.0 * (
        (
            (red[mask_blue] - green[mask_blue])
            / delta[mask_blue]
        ) + 4
    )

    hue = np.where(
        hue < 0,
        hue + 360.0,
        hue
    )

    # gris no hay tono real
    hue = np.where(
        delta < epsilon,
        0.0,
        hue
    )

    hsv_image = np.stack(
        [hue, saturation, value],
        axis=-1
    )

    return hsv_image


def hsv_to_rgb(hsv_image):
    hue = hsv_image[:, :, 0] % 360.0
    saturation = np.clip(
        hsv_image[:, :, 1],
        0.0,
        1.0
    )

    value = np.clip(
        hsv_image[:, :, 2],
        0.0,
        1.0
    )

    chroma = value * saturation
    hue_section = hue / 60.0

    middle_value = chroma * (
        1 - np.abs(hue_section % 2 - 1)
    )

    min_value = value - chroma

    red_base = np.zeros_like(value)
    green_base = np.zeros_like(value)
    blue_base = np.zeros_like(value)

    mask_0 = (hue_section >= 0) & (hue_section < 1)
    mask_1 = (hue_section >= 1) & (hue_section < 2)
    mask_2 = (hue_section >= 2) & (hue_section < 3)
    mask_3 = (hue_section >= 3) & (hue_section < 4)
    mask_4 = (hue_section >= 4) & (hue_section < 5)
    mask_5 = (hue_section >= 5) & (hue_section <= 6)

    red_base[mask_0] = chroma[mask_0]
    green_base[mask_0] = middle_value[mask_0]

    red_base[mask_1] = middle_value[mask_1]
    green_base[mask_1] = chroma[mask_1]

    green_base[mask_2] = chroma[mask_2]
    blue_base[mask_2] = middle_value[mask_2]

    green_base[mask_3] = middle_value[mask_3]
    blue_base[mask_3] = chroma[mask_3]

    red_base[mask_4] = middle_value[mask_4]
    blue_base[mask_4] = chroma[mask_4]

    red_base[mask_5] = chroma[mask_5]
    blue_base[mask_5] = middle_value[mask_5]

    rgb_image = np.stack(
        [
            red_base + min_value,
            green_base + min_value,
            blue_base + min_value
        ],
        axis=-1
    )

    return np.clip(
        rgb_image,
        0.0,
        1.0
    )



# RGB <-> L*c*h*


def gamma_to_linear(channel, gamma=2.2):
    return channel ** gamma


def rgb_to_xyz(image_rgb):
    image_float = image_rgb.astype(np.float64)
    image_float = image_float / 255.0

    conversion_matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [0.000, 0.010, 0.990]
    ])

    rgb_linear = gamma_to_linear(
        image_float
    )

    xyz_image = (
        rgb_linear
        @ conversion_matrix.T
    )

    return xyz_image


def rgb_to_lab(image_rgb):
    white_x = 1.0
    white_y = 1.0
    white_z = 1.0

    xyz_image = rgb_to_xyz(
        image_rgb
    )

    x_relative = (
        xyz_image[:, :, 0]
        / white_x
    )

    y_relative = (
        xyz_image[:, :, 1]
        / white_y
    )

    z_relative = (
        xyz_image[:, :, 2]
        / white_z
    )

    delta = 6.0 / 29.0

    def lab_function(value):
        return np.where(
            value > delta ** 3,
            np.cbrt(value),
            value / (3 * delta ** 2)
            + 4.0 / 29.0
        )

    function_x = lab_function(
        x_relative
    )

    function_y = lab_function(
        y_relative
    )

    function_z = lab_function(
        z_relative
    )

    lightness = (
        116.0 * function_y
        - 16.0
    )

    component_a = (
        500.0
        * (function_x - function_y)
    )

    component_b = (
        200.0
        * (function_y - function_z)
    )

    lab_image = np.stack(
        [
            lightness,
            component_a,
            component_b
        ],
        axis=-1
    )

    return lab_image


def rgb_to_lch(image_rgb):
    lab_image = rgb_to_lab(
        image_rgb
    )

    lightness = lab_image[:, :, 0]
    component_a = lab_image[:, :, 1]
    component_b = lab_image[:, :, 2]

    chroma = np.sqrt(
        component_a ** 2
        + component_b ** 2
    )

    hue = np.degrees(
        np.arctan2(
            component_b,
            component_a
        )
    )

    hue = np.where(
        hue < 0,
        hue + 360.0,
        hue
    )

    lch_image = np.stack(
        [
            lightness,
            chroma,
            hue
        ],
        axis=-1
    )

    return lch_image


def xyz_to_rgb(xyz_image):
    conversion_matrix = np.array([
        [0.490, 0.310, 0.200],
        [0.177, 0.813, 0.011],
        [0.000, 0.010, 0.990]
    ])

    inverse_matrix = np.linalg.inv(
        conversion_matrix
    )

    rgb_linear = (
        xyz_image
        @ inverse_matrix.T
    )

    # cortar colores que no entran en RGB
    rgb_linear = np.clip(
        rgb_linear,
        0.0,
        1.0
    )

    rgb_image = (
        rgb_linear
        ** (1.0 / 2.2)
    )

    return np.clip(
        rgb_image,
        0.0,
        1.0
    )


def lab_to_xyz(lab_image):
    white_x = 1.0
    white_y = 1.0
    white_z = 1.0

    lightness = lab_image[:, :, 0]
    component_a = lab_image[:, :, 1]
    component_b = lab_image[:, :, 2]

    function_y = (
        lightness + 16.0
    ) / 116.0

    function_x = (
        function_y
        + component_a / 500.0
    )

    function_z = (
        function_y
        - component_b / 200.0
    )

    delta = 6.0 / 29.0

    def lab_inverse(value):
        return np.where(
            value > delta,
            value ** 3,
            3 * delta ** 2
            * (value - 4.0 / 29.0)
        )

    x_value = (
        lab_inverse(function_x)
        * white_x
    )

    y_value = (
        lab_inverse(function_y)
        * white_y
    )

    z_value = (
        lab_inverse(function_z)
        * white_z
    )

    xyz_image = np.stack(
        [
            x_value,
            y_value,
            z_value
        ],
        axis=-1
    )

    return xyz_image


def lab_to_rgb(lab_image):
    xyz_image = lab_to_xyz(
        lab_image
    )

    return xyz_to_rgb(
        xyz_image
    )


def lch_to_lab(lch_image):
    lightness = lch_image[:, :, 0]
    chroma = lch_image[:, :, 1]

    hue_radians = np.radians(
        lch_image[:, :, 2]
    )

    component_a = (
        chroma
        * np.cos(hue_radians)
    )

    component_b = (
        chroma
        * np.sin(hue_radians)
    )

    lab_image = np.stack(
        [
            lightness,
            component_a,
            component_b
        ],
        axis=-1
    )

    return lab_image


def lch_to_rgb(lch_image):
    lab_image = lch_to_lab(
        lch_image
    )

    return lab_to_rgb(
        lab_image
    )


# funcion extra

def prepare_control_points(control_points):
    if len(control_points) == 0:
        raise ValueError(
            "Se necesita al menos un punto de control."
        )

    points_clean = []

    # dejar todos los tonos entre 0 y 360
    for hue_value, m_value in control_points:
        hue_clean = (
            float(hue_value) % 360.0
        )

        m_clean = float(
            m_value
        )

        points_clean.append(
            (hue_clean, m_clean)
        )

    points_clean.sort(
        key=lambda point: point[0]
    )

    # revisar tonos repetidos
    for index in range(
        1,
        len(points_clean)
    ):
        hue_previous = (
            points_clean[index - 1][0]
        )

        hue_current = (
            points_clean[index][0]
        )

        if np.isclose(
            hue_previous,
            hue_current
        ):
            raise ValueError(
                "Hay dos puntos de control con el mismo tono."
            )

    return points_clean


def interpolate_hue_parameter(
    hue_values,
    control_points
):
    points_clean = prepare_control_points(
        control_points
    )

    hue_array = np.asarray(
        hue_values,
        dtype=np.float64
    )

    hue_array = (
        hue_array % 360.0
    )

    # con un punto m queda fijo
    if len(points_clean) == 1:
        m_value = points_clean[0][1]

        return np.full(
            hue_array.shape,
            m_value,
            dtype=np.float64
        )

    m_map = np.zeros_like(
        hue_array,
        dtype=np.float64
    )

    num_points = len(
        points_clean
    )

    # tramos normales
    for index in range(
        num_points - 1
    ):
        hue_left = (
            points_clean[index][0]
        )

        m_left = (
            points_clean[index][1]
        )

        hue_right = (
            points_clean[index + 1][0]
        )

        m_right = (
            points_clean[index + 1][1]
        )

        mask_segment = (
            (hue_array >= hue_left)
            & (hue_array < hue_right)
        )

        interpolation_weight = (
            (
                hue_array[mask_segment]
                - hue_left
            )
            /
            (
                hue_right
                - hue_left
            )
        )

        m_map[mask_segment] = (
            m_left
            + interpolation_weight
            * (m_right - m_left)
        )

    hue_last = points_clean[-1][0]
    m_last = points_clean[-1][1]

    hue_first = points_clean[0][0]
    m_first = points_clean[0][1]

    hue_first_extended = (
        hue_first + 360.0
    )

    mask_last_segment = (
        (hue_array >= hue_last)
        | (hue_array < hue_first)
    )

    hue_extended = (
        hue_array[mask_last_segment]
        .copy()
    )

    # los tonos chicos pasan al otro lado
    hue_extended[
        hue_extended < hue_first
    ] += 360.0

    interpolation_weight = (
        (
            hue_extended
            - hue_last
        )
        /
        (
            hue_first_extended
            - hue_last
        )
    )

    m_map[mask_last_segment] = (
        m_last
        + interpolation_weight
        * (m_first - m_last)
    )

    return m_map


def apply_gm(
    component_original,
    m_values,
    min_m=0.0,
    max_m=2.0
):
    # m = 1 no cambia nada
    m_limited = np.clip(
        m_values,
        min_m,
        max_m
    )

    component_new = (
        component_original
        * m_limited
    )

    return component_new


def color_saturation_hs(
    image_rgb,
    control_points,
    min_m=0.0,
    max_m=2.0
):
    image_uint8 = prepare_rgb_image(
        image_rgb
    )

    hsv_image = rgb_to_hsv(
        image_uint8
    )

    hue = hsv_image[:, :, 0]

    saturation_original = (
        hsv_image[:, :, 1]
    )

    m_map = interpolate_hue_parameter(
        hue,
        control_points
    )

    saturation_new = apply_gm(
        saturation_original,
        m_map,
        min_m,
        max_m
    )

    # S tiene rango entre 0 y 1
    saturation_new = np.clip(
        saturation_new,
        0.0,
        1.0
    )

    hsv_result = hsv_image.copy()

    # solo cambiar S
    hsv_result[:, :, 1] = (
        saturation_new
    )

    rgb_result = hsv_to_rgb(
        hsv_result
    )

    return rgb_result


def color_saturation_lch(
    image_rgb,
    control_points,
    min_m=0.0,
    max_m=2.0
):
    image_uint8 = prepare_rgb_image(
        image_rgb
    )

    lch_image = rgb_to_lch(
        image_uint8
    )

    lightness_original = (
        lch_image[:, :, 0]
    )

    chroma_original = (
        lch_image[:, :, 1]
    )

    hue_original = (
        lch_image[:, :, 2]
    )

    m_map = interpolate_hue_parameter(
        hue_original,
        control_points
    )

    chroma_new = apply_gm(
        chroma_original,
        m_map,
        min_m,
        max_m
    )

    # C* no negativo
    chroma_new = np.maximum(
        chroma_new,
        0.0
    )

    lch_result = lch_image.copy()

    # solo cambiar C*
    lch_result[:, :, 1] = (
        chroma_new
    )

    # L* y h* quedan iguales
    lch_result[:, :, 0] = (
        lightness_original
    )

    lch_result[:, :, 2] = (
        hue_original
    )

    rgb_result = lch_to_rgb(
        lch_result
    )

    return np.clip(
        rgb_result,
        0.0,
        1.0
    )


def color_saturation(
    image_rgb,
    control_points,
    mode="HS",
    min_m=0.0,
    max_m=2.0
):
    mode_clean = (
        mode.upper()
    )

    if mode_clean == "HS":
        return color_saturation_hs(
            image_rgb,
            control_points,
            min_m,
            max_m
        )

    if mode_clean in (
        "LCH",
        "L*C*H*"
    ):
        return color_saturation_lch(
            image_rgb,
            control_points,
            min_m,
            max_m
        )

    raise ValueError(
        "Modo no valido. Use 'HS' o 'LCH'."
    )