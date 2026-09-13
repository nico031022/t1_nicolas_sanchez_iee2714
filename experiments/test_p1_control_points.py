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
    "results/pregunta1/control_points"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


hue_range = np.linspace(
    0.0,
    360.0,
    721,
    endpoint=False
)


# =========================================================
# EXPERIMENTO 1
# separacion de los puntos
# =========================================================

spacing_configs = [
    {
        "name": "wide",
        "title": "Rango amplio",
        "points": [
            (0, 1.0),
            (140, 1.0),
            (220, 1.6),
            (300, 1.0)
        ]
    },
    {
        "name": "medium",
        "title": "Rango medio",
        "points": [
            (0, 1.0),
            (180, 1.0),
            (220, 1.6),
            (260, 1.0)
        ]
    },
    {
        "name": "narrow",
        "title": "Rango estrecho",
        "points": [
            (0, 1.0),
            (205, 1.0),
            (220, 1.6),
            (235, 1.0)
        ]
    }
]


# curvas juntas
plt.figure(
    figsize=(9, 5)
)

for current_config in spacing_configs:

    points = current_config["points"]

    m_values = interpolate_hue_parameter(
        hue_range,
        points
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
    "Efecto de la separacion de los puntos"
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
    / "spacing_curves.png",
    dpi=150
)

plt.close()


# resultados de imagen
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
    spacing_configs
):

    result = color_saturation(
        image_rgb,
        current_config["points"],
        mode="HS"
    )

    axes[index + 1].imshow(
        result
    )

    axes[index + 1].set_title(
        current_config["title"]
    )

    axes[index + 1].axis(
        "off"
    )


figure.suptitle(
    "Separacion de puntos - modo HS"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "spacing_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# EXPERIMENTO 2
# numero de puntos de control
# =========================================================

number_configs = [
    {
        "name": "few",
        "title": "3 puntos",
        "points": [
            (170, 1.0),
            (220, 1.6),
            (270, 1.0)
        ]
    },
    {
        "name": "medium",
        "title": "5 puntos",
        "points": [
            (170, 1.0),
            (195, 1.3),
            (220, 1.6),
            (245, 1.3),
            (270, 1.0)
        ]
    },
    {
        "name": "many",
        "title": "8 puntos",
        "points": [
            (170, 1.0),
            (185, 1.18),
            (200, 1.36),
            (210, 1.48),
            (220, 1.6),
            (235, 1.42),
            (250, 1.24),
            (270, 1.0)
        ]
    }
]


plt.figure(
    figsize=(9, 5)
)

for current_config in number_configs:

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
    "Efecto del numero de puntos"
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
    / "number_curves.png",
    dpi=150
)

plt.close()


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
    number_configs
):

    result = color_saturation(
        image_rgb,
        current_config["points"],
        mode="HS"
    )

    axes[index + 1].imshow(
        result
    )

    axes[index + 1].set_title(
        current_config["title"]
    )

    axes[index + 1].axis(
        "off"
    )


figure.suptitle(
    "Numero de puntos - modo HS"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "number_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "\nResultados guardados en "
    "results/pregunta1/control_points/"
)

print(
    " - spacing_curves.png"
)

print(
    " - spacing_comparison.png"
)

print(
    " - number_curves.png"
)

print(
    " - number_comparison.png"
)