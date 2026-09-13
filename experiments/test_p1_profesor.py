import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.pregunta1 import (
    interpolate_hue_parameter,
    color_saturation
)


# --------------------------------------------
# leer la imagen
# ---------------------------------------------

image_path = "images/profesor/pregunta1.tif"


image_bgr = cv2.imread(
    image_path,
    cv2.IMREAD_UNCHANGED
)

if image_bgr is None:
    raise ValueError(
        "No se pudo leer la imagen."
    )

if image_bgr.ndim == 2:
    raise ValueError(
        "La imagen parece ser gris y no RGB."
    )

if image_bgr.shape[2] < 3:
    raise ValueError(
        "La imagen no tiene 3 canales."
    )

# por si viene con canal extra
image_bgr = image_bgr[:, :, :3]

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

print("Shape:", image_rgb.shape)
print("Dtype:", image_rgb.dtype)
print("Min / Max:", image_rgb.min(), image_rgb.max())


# ---------------------------------------------------------
# carpeta salida
# ---------------------------------------------------------

results_folder = Path(
    "results/pregunta1"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# configuraciones de control
# --------------------------------------------------------

configurations = [
    {
        "name": "config_1_blue_boost",
        "title": "Config 1 - aumentar azules",
        "control_points": [
            (0, 1.0),
            (150, 1.0),
            (190, 1.3),
            (220, 1.6),
            (250, 1.3),
            (330, 1.0)
        ]
    },
    {
        "name": "config_2_yellow_reduce",
        "title": "Config 2 - atenuar amarillos",
        "control_points": [
            (0, 1.0),
            (30, 1.0),
            (55, 0.8),
            (75, 0.5),
            (95, 0.8),
            (180, 1.0),
            (300, 1.0)
        ]
    },
    {
        "name": "config_3_mixed",
        "title": "Config 3 - mezcla",
        "control_points": [
            (0, 1.0),
            (60, 0.7),
            (110, 1.0),
            (135, 1.2),
            (180, 1.0),
            (220, 1.5),
            (250, 1.2),
            (320, 1.0)
        ]
    },
    {
        "name": "config_4_strong",
        "title": "Config 4 - caso fuerte",
        "control_points": [
            (0, 1.0),
            (160, 1.0),
            (200, 1.8),
            (220, 2.0),
            (240, 1.8),
            (330, 1.0)
        ]
    }
]


# ---------------------------------------------------------
# rango de tono para visua
# ---------------------------------------------------------

hue_range = np.linspace(
    0.0,
    360.0,
    721,
    endpoint=False
)


# ---------------------------------------------------------
# recorrer las configs
# ---------------------------------------------------------

for current_config in configurations:

    config_name = current_config["name"]
    config_title = current_config["title"]
    control_points = current_config["control_points"]

    print("\nTrabajando con:", config_name)

    # calcular curva
    m_values = interpolate_hue_parameter(
        hue_range,
        control_points
    )

    plt.figure(figsize=(8, 4.5))

    plt.plot(
        hue_range,
        m_values,
        label="m(h)"
    )

    hue_controls = [
        point[0]
        for point in control_points
    ]

    m_controls = [
        point[1]
        for point in control_points
    ]

    plt.scatter(
        hue_controls,
        m_controls,
        label="Puntos de control"
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="m neutro"
    )

    plt.title(config_title)
    plt.xlabel("Hue [grados]")
    plt.ylabel("m(h)")
    plt.xlim(0, 360)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_folder / f"{config_name}_curve.png",
        dpi=150
    )

    plt.close()

    # dos modos
    result_hs = color_saturation(
        image_rgb,
        control_points,
        mode="HS"
    )

    result_lch = color_saturation(
        image_rgb,
        control_points,
        mode="LCH"
    )

    #  | HS | LCh
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 6)
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(result_hs)
    axes[1].set_title("HS")
    axes[1].axis("off")

    axes[2].imshow(result_lch)
    axes[2].set_title("LCh")
    axes[2].axis("off")

    figure.suptitle(config_title)
    plt.tight_layout()

    plt.savefig(
        results_folder / f"{config_name}_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

print("\nListo. Se guardaron los resultados en results/pregunta1/")