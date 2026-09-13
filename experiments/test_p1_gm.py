import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.pregunta1 import (
    rgb_to_hsv,
    hsv_to_rgb,
    interpolate_hue_parameter
)


# ---------------------------------------------------------
# gm lineal
# ---------------------------------------------------------

def apply_gm_linear(component_value, m_value):
    return component_value * m_value


# ---------------------------------------------------------
# mtf (de la capsula)
# ---------------------------------------------------------

def apply_mtf_transform(component_value, parameter_p):

    epsilon_value = 1e-12

    numerator_value = (
        (parameter_p - 1.0)
        * component_value
    )

    denominator_value = (
        (2.0 * parameter_p - 1.0)
        * component_value
        - parameter_p
    )

    denominator_value = np.where(
        np.abs(denominator_value) < epsilon_value,
        epsilon_value,
        denominator_value
    )

    result_value = (
        numerator_value
        / denominator_value
    )

    return result_value


# ---------------------------------------------------------
# m -> p
# neutral:
# m = 1  <->  p = 0.5
# ------------------------------------------------------

def convert_m_to_p(m_value):
    return 1.0 / (1.0 + m_value)


# ---------------------------------------------------------
# HS con gm lineal
# ---------------------------------------------------------

def color_saturation_hs_linear(
    image_rgb,
    control_points
):

    hsv_image = rgb_to_hsv(
        image_rgb
    )

    hue_channel = hsv_image[:, :, 0]
    saturation_channel = hsv_image[:, :, 1]

    m_map = interpolate_hue_parameter(
        hue_channel,
        control_points
    )

    saturation_new = apply_gm_linear(
        saturation_channel,
        m_map
    )

    saturation_new = np.clip(
        saturation_new,
        0.0,
        1.0
    )

    hsv_result = hsv_image.copy()
    hsv_result[:, :, 1] = saturation_new

    result_rgb = hsv_to_rgb(
        hsv_result
    )

    return result_rgb


# ---------------------------------------------------------
# HS con gm tipo MTF
# ---------------------------------------------------------

def color_saturation_hs_mtf(
    image_rgb,
    control_points
):

    hsv_image = rgb_to_hsv(
        image_rgb
    )

    hue_channel = hsv_image[:, :, 0]
    saturation_channel = hsv_image[:, :, 1]

    m_map = interpolate_hue_parameter(
        hue_channel,
        control_points
    )

    parameter_p = convert_m_to_p(
        m_map
    )

    saturation_new = apply_mtf_transform(
        saturation_channel,
        parameter_p
    )

    saturation_new = np.clip(
        saturation_new,
        0.0,
        1.0
    )

    hsv_result = hsv_image.copy()
    hsv_result[:, :, 1] = saturation_new

    result_rgb = hsv_to_rgb(
        hsv_result
    )

    return result_rgb


# ---------------------------------------------------------
# imagen
# ------------------------------------------------------

image_path = "images/propias/fruit.jpg"

image_bgr = cv2.imread(
    image_path,
    cv2.IMREAD_COLOR
)

if image_bgr is None:
    raise ValueError(
        "No se pudo leer la imagen."
    )

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

print("Shape:", image_rgb.shape)
print("Dtype:", image_rgb.dtype)
print(
    "Min / Max:",
    image_rgb.min(),
    image_rgb.max()
)


# ------------------------------------------------------
# carpeta
# ---------------------------------------------------------

results_folder = Path(
    "results/pregunta1/gm"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# usar la config 3
# porque tiene aumento y reduccion
# ---------------------------------------------------------

control_points = [
    (0, 1.0),
    (60, 0.7),
    (110, 1.0),
    (135, 1.2),
    (180, 1.0),
    (220, 1.5),
    (250, 1.2),
    (320, 1.0)
]


# ---------------------------------------------------------
# aplicar los dos gm
# ---------------------------------------------------------

result_linear = color_saturation_hs_linear(
    image_rgb,
    control_points
)

result_mtf = color_saturation_hs_mtf(
    image_rgb,
    control_points
)


# ---------------------------------------------------------
# comparacion visual
# ---------------------------------------------------------

figure, axes = plt.subplots(
    1,
    3,
    figsize=(16, 6)
)

axes[0].imshow(
    image_rgb
)
axes[0].set_title(
    "Original"
)
axes[0].axis(
    "off"
)

axes[1].imshow(
    result_linear
)
axes[1].set_title(
    "HS con g_m lineal"
)
axes[1].axis(
    "off"
)

axes[2].imshow(
    result_mtf
)
axes[2].set_title(
    "HS con g_m tipo MTF"
)
axes[2].axis(
    "off"
)

figure.suptitle(
    "Comparacion de disenos de g_m"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "gm_comparison_image.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# curvas g_m para varios m
# ---------------------------------------------------------

input_range = np.linspace(
    0.0,
    1.0,
    500
)

m_values_to_show = [
    0.7,
    1.0,
    1.5
]

figure, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

for current_m in m_values_to_show:

    output_linear = apply_gm_linear(
        input_range,
        current_m
    )

    output_linear = np.clip(
        output_linear,
        0.0,
        1.0
    )

    axes[0].plot(
        input_range,
        output_linear,
        label=f"m = {current_m}"
    )

axes[0].plot(
    input_range,
    input_range,
    linestyle="--",
    label="identidad"
)

axes[0].set_title(
    "g_m lineal"
)
axes[0].set_xlabel(
    "Entrada"
)
axes[0].set_ylabel(
    "Salida"
)
axes[0].grid(
    alpha=0.3
)
axes[0].legend()


for current_m in m_values_to_show:

    parameter_p = convert_m_to_p(
        current_m
    )

    output_mtf = apply_mtf_transform(
        input_range,
        parameter_p
    )

    output_mtf = np.clip(
        output_mtf,
        0.0,
        1.0
    )

    axes[1].plot(
        input_range,
        output_mtf,
        label=f"m = {current_m}"
    )

axes[1].plot(
    input_range,
    input_range,
    linestyle="--",
    label="identidad"
)

axes[1].set_title(
    "g_m tipo MTF"
)
axes[1].set_xlabel(
    "Entrada"
)
axes[1].set_ylabel(
    "Salida"
)
axes[1].grid(
    alpha=0.3
)
axes[1].legend()

plt.tight_layout()

plt.savefig(
    results_folder
    / "gm_curves.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# example numerico simple
# ------------------------------------------------------

sample_s_values = np.array([
    0.2,
    0.5,
    0.8
])

sample_m_values = np.array([
    0.7,
    1.0,
    1.5
])

print("\n--- Comparacion numerica simple ---")

for current_m in sample_m_values:

    parameter_p = convert_m_to_p(
        current_m
    )

    print(
        f"\nm = {current_m:.2f} | p = {parameter_p:.6f}"
    )

    for current_s in sample_s_values:

        result_linear = apply_gm_linear(
            current_s,
            current_m
        )

        result_linear = np.clip(
            result_linear,
            0.0,
            1.0
        )

        result_mtf = apply_mtf_transform(
            current_s,
            parameter_p
        )

        result_mtf = np.clip(
            result_mtf,
            0.0,
            1.0
        )

        print(
            f"S = {current_s:.2f} | "
            f"lineal = {result_linear:.6f} | "
            f"MTF = {result_mtf:.6f}"
        )


print(
    "\nResultados guardados en "
    "results/pregunta1/gm/"
)

print(
    " - gm_comparison_image.png"
)

print(
    " - gm_curves.png"
)