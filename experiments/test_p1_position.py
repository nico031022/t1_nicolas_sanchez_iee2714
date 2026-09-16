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


# ---------------------------------------------------------
# carpeta
# ---------------------------------------------------------

results_folder = Path(
    "results/pregunta1/position"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# mismas forma y amplitud
# solo cambia la posicion
# ---------------------------------------------------------

position_configs = [
    {
        "name": "yellow",
        "title": "Centro en 60°",
        "points": [
            (0, 1.0),
            (30, 1.0),
            (60, 1.6),
            (90, 1.0),
            (330, 1.0)
        ]
    },
    {
        "name": "green",
        "title": "Centro en 120°",
        "points": [
            (0, 1.0),
            (90, 1.0),
            (120, 1.6),
            (150, 1.0),
            (330, 1.0)
        ]
    },
    {
        "name": "blue",
        "title": "Centro en 220°",
        "points": [
            (0, 1.0),
            (190, 1.0),
            (220, 1.6),
            (250, 1.0),
            (330, 1.0)
        ]
    }
]


# ---------------------------------------------------------
# curvas
# ---------------------------------------------------------

hue_range = np.linspace(
    0.0,
    360.0,
    721,
    endpoint=False
)

plt.figure(
    figsize=(9, 5)
)

for current_config in position_configs:

    m_values = interpolate_hue_parameter(
        hue_range,
        current_config["points"]
    )

    plt.plot(
        hue_range,
        m_values,
        label=current_config["title"]
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
    "Efecto de la posicion de los puntos"
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
    / "position_curves.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# resultados HS
# ---------------------------------------------------------

figure, axes = plt.subplots(
    1,
    4,
    figsize=(20, 6)
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


for index, current_config in enumerate(
    position_configs
):

    result_hs = color_saturation(
        image_rgb,
        current_config["points"],
        mode="HS"
    )

    axes[index + 1].imshow(
        result_hs
    )

    axes[index + 1].set_title(
        current_config["title"]
    )

    axes[index + 1].axis(
        "off"
    )


figure.suptitle(
    "Posicion de los puntos - modo HS"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "position_comparison_hs.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# resultados LCh
# ---------------------------------------------------------

figure, axes = plt.subplots(
    1,
    4,
    figsize=(20, 6)
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


for index, current_config in enumerate(
    position_configs
):

    result_lch = color_saturation(
        image_rgb,
        current_config["points"],
        mode="LCH"
    )

    axes[index + 1].imshow(
        result_lch
    )

    axes[index + 1].set_title(
        current_config["title"]
    )

    axes[index + 1].axis(
        "off"
    )


figure.suptitle(
    "Posicion de los puntos - modo LCh"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "position_comparison_lch.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "\nResultados guardados en "
    "results/pregunta1/position/"
)

print(
    " - position_curves.png"
)

print(
    " - position_comparison_hs.png"
)

print(
    " - position_comparison_lch.png"
)