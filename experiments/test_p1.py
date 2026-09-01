import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.pregunta1 import (
    interpolate_hue_parameter,
    apply_gm,
    color_saturation
)



# Test 1: interpolacion de m(h)


control_points = [
    (30, 0.6),
    (120, 1.2),
    (220, 1.6),
    (310, 0.8)
]

hue_range = np.linspace(
    0.0,
    360.0,
    721,
    endpoint=False
)

m_values = interpolate_hue_parameter(
    hue_range,
    control_points
)


# Test 2: revisar los puntos de control

print("\n--- Test de puntos de control ---")

for hue_control, m_control in control_points:

    m_result = interpolate_hue_parameter(
        np.array([hue_control]),
        control_points
    )[0]

    print(
        f"h = {hue_control:6.1f}° | "
        f"m esperado = {m_control:.4f} | "
        f"m obtenido = {m_result:.4f}"
    )

    assert np.isclose(
        m_result,
        m_control
    )


print("OK: los puntos de control coinciden.")


# Test 3: revisar el cierre 360 -> 0

hue_red_test = np.array([
    358.0,
    359.0,
    0.0,
    1.0,
    2.0
])

m_red_test = interpolate_hue_parameter(
    hue_red_test,
    control_points
)

print("\n--- Test cerca del rojo ---")

for hue_value, m_value in zip(
    hue_red_test,
    m_red_test
):
    print(
        f"h = {hue_value:6.1f}° | "
        f"m = {m_value:.4f}"
    )


difference_359_0 = abs(
    m_red_test[1]
    - m_red_test[2]
)

difference_0_1 = abs(
    m_red_test[2]
    - m_red_test[3]
)

print(
    "\nDiferencia 359° -> 0°:",
    difference_359_0
)

print(
    "Diferencia 0° -> 1°:",
    difference_0_1
)

assert difference_359_0 < 0.01
assert difference_0_1 < 0.01

print("OK: no hay salto grande cerca de 0°.")


# Test 4: m = 1 debe ser neutro

component_original = np.array([
    0.0,
    0.25,
    0.50,
    0.75,
    1.0
])

m_neutral = np.ones_like(
    component_original
)

component_result = apply_gm(
    component_original,
    m_neutral
)

print("\n--- Test de g_m ---")

print(
    "Original:",
    component_original
)

print(
    "Con m = 1:",
    component_result
)

assert np.allclose(
    component_original,
    component_result
)

print("OK: m = 1 es neutro.")


# Test 5: imagen RGB pequena

image_test = np.array([
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0]
    ],
    [
        [0, 255, 255],
        [255, 0, 255],
        [128, 128, 128],
        [200, 100, 50]
    ]
], dtype=np.uint8)


neutral_points = [
    (0, 1.0),
    (90, 1.0),
    (180, 1.0),
    (270, 1.0)
]

image_original_float = (
    image_test.astype(np.float64)
    / 255.0
)

result_hs = color_saturation(
    image_test,
    neutral_points,
    mode="HS"
)

result_lch = color_saturation(
    image_test,
    neutral_points,
    mode="LCH"
)

error_hs = np.max(
    np.abs(
        image_original_float
        - result_hs
    )
)

error_lch = np.max(
    np.abs(
        image_original_float
        - result_lch
    )
)

print("\n--- Test completo con m = 1 ---")

print(
    f"Error maximo HS:  {error_hs:.8f}"
)

print(
    f"Error maximo LCh: {error_lch:.8f}"
)

assert error_hs < 1e-6
assert error_lch < 1e-6

print("OK: los dos modos dejan la imagen igual.")


# Grafico de la interpolacion

results_folder = Path(
    "results/pregunta1"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)

plt.figure(
    figsize=(9, 5)
)

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

plt.xlabel(
    "Hue [grados]"
)

plt.ylabel(
    "m(h)"
)

plt.title(
    "Interpolacion periodica de los puntos de control"
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
    / "test_interpolacion_periodica.png",
    dpi=150
)

plt.show()


print(
    "\nGrafico guardado en "
    "results/pregunta1/test_interpolacion_periodica.png"
)

print(
    "\nTodos los tests terminaron correctamente."
)