import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.pregunta1 import (
    rgb_to_hsv,
    hsv_to_rgb,
    rgb_to_lch,
    lch_to_rgb,
    interpolate_hue_parameter,
    apply_gm
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

image_height = image_rgb.shape[0]
image_width = image_rgb.shape[1]


# --------------------------------------------------------
# pixel elegido en el ala azul
# ---------------------------------------------------------

# posicion tomada desde la imagen de referencia
pixel_x_ratio = 939 / 1363
pixel_y_ratio = 1144 / 2048

pixel_x = int(
    pixel_x_ratio * image_width
)

pixel_y = int(
    pixel_y_ratio * image_height
)

pixel_rgb = image_rgb[
    pixel_y,
    pixel_x
]

print("\n--- Pixel seleccionado ---")
print(
    f"Coordenada (x, y): "
    f"({pixel_x}, {pixel_y})"
)
print(
    "RGB original:",
    pixel_rgb
)


# ---------------------------------------------------------
# confi mixta
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


# solo un pixel
pixel_image = np.array(
    [[pixel_rgb]],
    dtype=np.uint8
)


# ---------------------------------------------------------
# modo HS
# -------------------------------------------------------

hsv_pixel = rgb_to_hsv(
    pixel_image
)

hue_hs = hsv_pixel[0, 0, 0]
saturation_original = hsv_pixel[0, 0, 1]
value_original = hsv_pixel[0, 0, 2]

m_hs = interpolate_hue_parameter(
    np.array([hue_hs]),
    control_points
)[0]

saturation_raw = apply_gm(
    np.array([saturation_original]),
    np.array([m_hs])
)[0]

saturation_final = np.clip(
    saturation_raw,
    0.0,
    1.0
)

hsv_result = hsv_pixel.copy()

hsv_result[0, 0, 1] = (
    saturation_final
)

rgb_result_hs = hsv_to_rgb(
    hsv_result
)[0, 0]

rgb_result_hs_255 = np.round(
    rgb_result_hs * 255.0
).astype(np.uint8)


print("\n--- Modo HS ---")

print(
    f"H: {hue_hs:.4f} grados"
)

print(
    f"S original: "
    f"{saturation_original:.6f}"
)

print(
    f"V: "
    f"{value_original:.6f}"
)

print(
    f"m(h): "
    f"{m_hs:.6f}"
)

print(
    f"g_m(S) antes del clip: "
    f"{saturation_raw:.6f}"
)

print(
    f"S final: "
    f"{saturation_final:.6f}"
)

print(
    "RGB final:",
    rgb_result_hs_255
)


# ---------------------------------------------------------
# modo LCh
# ---------------------------------------------------------

lch_pixel = rgb_to_lch(
    pixel_image
)

lightness_original = (
    lch_pixel[0, 0, 0]
)

chroma_original = (
    lch_pixel[0, 0, 1]
)

hue_lch = (
    lch_pixel[0, 0, 2]
)

m_lch = interpolate_hue_parameter(
    np.array([hue_lch]),
    control_points
)[0]

chroma_final = apply_gm(
    np.array([chroma_original]),
    np.array([m_lch])
)[0]

lch_result = lch_pixel.copy()

lch_result[0, 0, 1] = (
    chroma_final
)

rgb_result_lch = lch_to_rgb(
    lch_result
)[0, 0]

rgb_result_lch_255 = np.round(
    rgb_result_lch * 255.0
).astype(np.uint8)


print("\n--- Modo LCh ---")

print(
    f"L*: "
    f"{lightness_original:.6f}"
)

print(
    f"c* original: "
    f"{chroma_original:.6f}"
)

print(
    f"h*: "
    f"{hue_lch:.4f} grados"
)

print(
    f"m(h*): "
    f"{m_lch:.6f}"
)

print(
    f"g_m(c*): "
    f"{chroma_final:.6f}"
)

print(
    "RGB final:",
    rgb_result_lch_255
)


# ----------------------------------------------------
# guardar los datos
# ---------------------------------------------------------

results_folder = Path(
    "results/pregunta1"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)

text_result = (
    "PIXEL TRACE - PREGUNTA 1\n"
    "\n"
    f"Coordenada (x, y): "
    f"({pixel_x}, {pixel_y})\n"
    f"RGB original: "
    f"{pixel_rgb.tolist()}\n"
    "\n"
    "MODO HS\n"
    f"H = {hue_hs:.6f} grados\n"
    f"S original = {saturation_original:.6f}\n"
    f"V = {value_original:.6f}\n"
    f"m(h) = {m_hs:.6f}\n"
    f"g_m(S) = {saturation_raw:.6f}\n"
    f"S final = {saturation_final:.6f}\n"
    f"RGB final = "
    f"{rgb_result_hs_255.tolist()}\n"
    "\n"
    "MODO LCh\n"
    f"L* = {lightness_original:.6f}\n"
    f"c* original = {chroma_original:.6f}\n"
    f"h* = {hue_lch:.6f} grados\n"
    f"m(h*) = {m_lch:.6f}\n"
    f"g_m(c*) = {chroma_final:.6f}\n"
    f"RGB final = "
    f"{rgb_result_lch_255.tolist()}\n"
)

with open(
    results_folder / "pixel_trace.txt",
    "w",
    encoding="utf-8"
) as file_result:
    file_result.write(
        text_result
    )


# ---------------------------------------------------------
# mostrar donde esta el pixel
# -------------------------------------------------------

crop_size = 170

x_min = max(
    pixel_x - crop_size,
    0
)

x_max = min(
    pixel_x + crop_size,
    image_width
)

y_min = max(
    pixel_y - crop_size,
    0
)

y_max = min(
    pixel_y + crop_size,
    image_height
)

image_crop = image_rgb[
    y_min:y_max,
    x_min:x_max
]


figure, axes = plt.subplots(
    1,
    2,
    figsize=(10, 6)
)

axes[0].imshow(
    image_rgb
)

axes[0].scatter(
    pixel_x,
    pixel_y,
    s=90,
    facecolors="none",
    edgecolors="red",
    linewidths=2
)

axes[0].set_title(
    "Pixel seleccionado"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    image_crop
)

axes[1].scatter(
    pixel_x - x_min,
    pixel_y - y_min,
    s=130,
    facecolors="none",
    edgecolors="red",
    linewidths=2
)

axes[1].set_title(
    "Detalle del ala"
)

axes[1].axis(
    "off"
)


plt.tight_layout()

plt.savefig(
    results_folder
    / "pixel_trace_location.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "\nDatos guardados en "
    "results/pregunta1/pixel_trace.txt"
)

print(
    "Imagen guardada en "
    "results/pregunta1/pixel_trace_location.png"
)