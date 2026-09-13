import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.pregunta1 import (
    rgb_to_hsv,
    rgb_to_lch,
    lch_to_lab,
    lab_to_xyz,
    interpolate_hue_parameter,
    color_saturation
)


# ---------------------------------------------------------
# imagen
# -------------------------------------------------------

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


print("Shape:", image_rgb.shape)
print("Dtype:", image_rgb.dtype)
print("Min / Max:", image_rgb.min(), image_rgb.max())


# ---------------------------------------------------------
# carpeta
# ---------------------------------------------------------

results_folder = Path(
    "results/pregunta1"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------
# configuracion fuerte
# ---------------------------------------------------------

control_points_extreme = [
    (0, 1.0),
    (160, 1.0),
    (195, 2.5),
    (220, 3.5),
    (245, 2.5),
    (330, 1.0)
]

max_m_extreme = 4.0


# ---------------------------------------------------------
# HS: check clipping antes del np.clip
# ---------------------------------------------------------

hsv_image = rgb_to_hsv(
    image_rgb
)

hue_hs = hsv_image[:, :, 0]
saturation_original = hsv_image[:, :, 1]

m_map_hs = interpolate_hue_parameter(
    hue_hs,
    control_points_extreme
)

m_map_hs = np.clip(
    m_map_hs,
    0.0,
    max_m_extreme
)

saturation_raw = (
    saturation_original
    * m_map_hs
)

mask_clipping_hs = (
    saturation_raw > 1.0
)

num_pixels_hs = np.count_nonzero(
    mask_clipping_hs
)

num_total_pixels = (
    image_rgb.shape[0]
    * image_rgb.shape[1]
)

percentage_hs = (
    num_pixels_hs
    / num_total_pixels
    * 100.0
)


print("\n--- Clipping HS ---")

print(
    "Pixeles con S > 1:",
    num_pixels_hs
)

print(
    f"Porcentaje: {percentage_hs:.4f}%"
)

print(
    "S maxima antes del clip:",
    saturation_raw.max()
)


# ---------------------------------------------------------
# LCh: aumentar C* y mirar RGB antes del clip
# -----------------------------------------------------

lch_image = rgb_to_lch(
    image_rgb
)

chroma_original = lch_image[:, :, 1]
hue_lch = lch_image[:, :, 2]

m_map_lch = interpolate_hue_parameter(
    hue_lch,
    control_points_extreme
)

m_map_lch = np.clip(
    m_map_lch,
    0.0,
    max_m_extreme
)

chroma_raw = (
    chroma_original
    * m_map_lch
)

lch_modified = lch_image.copy()

lch_modified[:, :, 1] = (
    chroma_raw
)

lab_modified = lch_to_lab(
    lch_modified
)

xyz_modified = lab_to_xyz(
    lab_modified
)


# misma matriz usada en pregunta1.py
conversion_matrix = np.array([
    [0.490, 0.310, 0.200],
    [0.177, 0.813, 0.011],
    [0.000, 0.010, 0.990]
])

inverse_matrix = np.linalg.inv(
    conversion_matrix
)

rgb_linear_raw = (
    xyz_modified
    @ inverse_matrix.T
)


# si cualquier canal queda fuera, ese pixel no entra en RGB
mask_low_lch = np.any(
    rgb_linear_raw < 0.0,
    axis=2
)

mask_high_lch = np.any(
    rgb_linear_raw > 1.0,
    axis=2
)

mask_gamut_lch = (
    mask_low_lch
    | mask_high_lch
)

num_pixels_lch = np.count_nonzero(
    mask_gamut_lch
)

percentage_lch = (
    num_pixels_lch
    / num_total_pixels
    * 100.0
)


print("\n--- Fuera de gamut LCh ---")

print(
    "Pixeles fuera del rango RGB:",
    num_pixels_lch
)

print(
    f"Porcentaje: {percentage_lch:.4f}%"
)

print(
    "RGB lineal minimo antes del clip:",
    rgb_linear_raw.min()
)

print(
    "RGB lineal maximo antes del clip:",
    rgb_linear_raw.max()
)


# ---------------------------------------------------------
# resultados finales
# ---------------------------------------------------------

result_hs = color_saturation(
    image_rgb,
    control_points_extreme,
    mode="HS",
    max_m=max_m_extreme
)

result_lch = color_saturation(
    image_rgb,
    control_points_extreme,
    mode="LCH",
    max_m=max_m_extreme
)


# ---------------------------------------------------------
# comparacion
# ---------------------------------------------------------

figure, axes = plt.subplots(
    1,
    3,
    figsize=(15, 6)
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
    result_hs
)

axes[1].set_title(
    f"HS\nclipping: {percentage_hs:.2f}%"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    result_lch
)

axes[2].set_title(
    f"LCh\nfuera de gamut: {percentage_lch:.2f}%"
)

axes[2].axis(
    "off"
)


figure.suptitle(
    "Caso extremo de saturacion / croma"
)

plt.tight_layout()

plt.savefig(
    results_folder
    / "config_extreme_clipping_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# mapas de clipping
# ---------------------------------------------------------

figure, axes = plt.subplots(
    1,
    2,
    figsize=(11, 6)
)

axes[0].imshow(
    mask_clipping_hs,
    cmap="gray"
)

axes[0].set_title(
    "Pixeles con clipping en S"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    mask_gamut_lch,
    cmap="gray"
)

axes[1].set_title(
    "Pixeles fuera de gamut RGB"
)

axes[1].axis(
    "off"
)


plt.tight_layout()

plt.savefig(
    results_folder
    / "config_extreme_clipping_masks.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# curva m(h)
# ---------------------------------------------------------

hue_range = np.linspace(
    0.0,
    360.0,
    721,
    endpoint=False
)

m_curve = interpolate_hue_parameter(
    hue_range,
    control_points_extreme
)

plt.figure(
    figsize=(8, 4.5)
)

plt.plot(
    hue_range,
    m_curve,
    label="m(h)"
)

hue_controls = [
    point[0]
    for point in control_points_extreme
]

m_controls = [
    point[1]
    for point in control_points_extreme
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

plt.title(
    "Configuracion extrema"
)

plt.xlabel(
    "Hue [grados]"
)

plt.ylabel(
    "m(h)"
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
    / "config_extreme_curve.png",
    dpi=150
)

plt.close()


print(
    "\nResultados guardados en results/pregunta1/"
)