import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.pregunta1 import (
    interpolate_hue_parameter,
    color_saturation
)


# ---------------------------------------------------------
# imagen
# ---------------------------------------------------------

image_path = "images/profesor/pregunta1.tif"

image_bgr = cv2.imread(
    image_path,
    cv2.IMREAD_UNCHANGED
)

if image_bgr is None:
    raise ValueError(
        "No se pudo leer la imagen."
    )

image_bgr = image_bgr[:, :, :3]

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


# ---------------------------------------------------------
# carpeta
# ---------------------------------------------------------

results_folder = Path(
    "results/pregunta1/m_range"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# valores de m
# ---------------------------------------------------------

peak_values = [
    1.2,
    1.5,
    2.0,
    3.5
]


hue_range = np.linspace(
    0.0,
    360.0,
    721,
    endpoint=False
)


# ---------------------------------------------------------
# curvas
# ---------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)


all_control_points = []


for current_peak in peak_values:

    control_points = [
        (0, 1.0),
        (160, 1.0),
        (220, current_peak),
        (280, 1.0),
        (330, 1.0)
    ]

    all_control_points.append(
        control_points
    )

    m_values = interpolate_hue_parameter(
        hue_range,
        control_points
    )

    plt.plot(
        hue_range,
        m_values,
        label=f"m max = {current_peak}"
    )


plt.axhline(
    1.0,
    linestyle="--",
    label="m neutro"
)

plt.xlabel(
    "Hue [grados]"
)

plt.ylabel(
    "m(h)"
)

plt.title(
    "Efecto de la amplitud de m"
)

plt.xlim(
    0,
    360
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    results_folder
    / "m_amplitude_curves.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# resultados HS
# ---------------------------------------------------------

figure, axes = plt.subplots(
    1,
    5,
    figsize=(22, 6)
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


for index, current_peak in enumerate(
    peak_values
):

    result_hs = color_saturation(
        image_rgb,
        all_control_points[index],
        mode="HS",
        max_m=4.0
    )

    axes[index + 1].imshow(
        result_hs
    )

    axes[index + 1].set_title(
        f"m max = {current_peak}"
    )

    axes[index + 1].axis(
        "off"
    )


figure.suptitle(
    "Amplitud de m - modo HS"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "m_amplitude_hs.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# resultados LCh
# ---------------------------------------------------------

figure, axes = plt.subplots(
    1,
    5,
    figsize=(22, 6)
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


for index, current_peak in enumerate(
    peak_values
):

    result_lch = color_saturation(
        image_rgb,
        all_control_points[index],
        mode="LCH",
        max_m=4.0
    )

    axes[index + 1].imshow(
        result_lch
    )

    axes[index + 1].set_title(
        f"m max = {current_peak}"
    )

    axes[index + 1].axis(
        "off"
    )


figure.suptitle(
    "Amplitud de m - modo LCh"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "m_amplitude_lch.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# rango permitido de m
# ---------------------------------------------------------

control_points_strong = [
    (0, 1.0),
    (160, 1.0),
    (220, 3.5),
    (280, 1.0),
    (330, 1.0)
]


max_m_values = [
    1.5,
    2.0,
    4.0
]


figure, axes = plt.subplots(
    1,
    4,
    figsize=(18, 6)
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


for index, current_max_m in enumerate(
    max_m_values
):

    result_range = color_saturation(
        image_rgb,
        control_points_strong,
        mode="HS",
        max_m=current_max_m
    )

    axes[index + 1].imshow(
        result_range
    )

    axes[index + 1].set_title(
        f"max_m = {current_max_m}"
    )

    axes[index + 1].axis(
        "off"
    )


figure.suptitle(
    "Efecto del rango permitido de m"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "m_range_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "\nResultados guardados en "
    "results/pregunta1/m_range/"
)

print(
    " - m_amplitude_curves.png"
)

print(
    " - m_amplitude_hs.png"
)

print(
    " - m_amplitude_lch.png"
)

print(
    " - m_range_comparison.png"
)