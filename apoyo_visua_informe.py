# este archivo contiene visualización extra de informe
# para correr hay que usar: python apoyo_visua_informe.py

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from experiments.test_p3_pixel_trace import (
    SCALE_FACTOR,
    OUTPUT_ROW,
    OUTPUT_COL,
)
from src.pregunta3 import (
    calculate_output_size,
    map_output_to_input,
)


def main():

    project_directory = Path(__file__).resolve().parent

    image_path = (
        project_directory
        / "images/profesor/pregunta3.tif"
    )

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError("No se pudo leer la imagen")

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # mismo pixel del experimento P3
    output_shape = calculate_output_size(
        image_rgb.shape,
        SCALE_FACTOR,
    )

    input_row, input_col = map_output_to_input(
        OUTPUT_ROW,
        OUTPUT_COL,
        image_rgb.shape[:2],
        output_shape,
    )

    top_row = int(np.floor(input_row))
    left_col = int(np.floor(input_col))

    bottom_row = min(
        top_row + 1,
        image_rgb.shape[0] - 1,
    )
    right_col = min(
        left_col + 1,
        image_rgb.shape[1] - 1,
    )

    # recorte
    image_crop = image_rgb[1396:1404, 296:304]

    figure, axis = plt.subplots(
        figsize=(5.2, 3.4)
    )

    axis.imshow(
        image_crop,
        extent=(295.5, 303.5, 1403.5, 1395.5),
        interpolation="nearest",
    )

    axis.scatter(
        [left_col, right_col, left_col, right_col],
        [top_row, top_row, bottom_row, bottom_row],
        s=65,
        facecolors="none",
        edgecolors="#ffb000",
        linewidths=1.5,
        label="Vecinos",
    )

    axis.scatter(
        input_col,
        input_row,
        marker="+",
        s=125,
        color="#c60020",
        linewidths=1.8,
        label="Posición de entrada",
    )

    axis.set_xlabel("Columna")
    axis.set_ylabel("Fila")
    axis.set_xticks([296, 298, 300, 302])
    axis.set_yticks([1396, 1398, 1400, 1402])
    axis.legend(loc="upper left", fontsize=9)

    figure.tight_layout()

    output_directory = (
        project_directory
        / "results/apoyo_informe"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / "p3_pixel.png"

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(
        f"Coordenada de entrada: "
        f"({input_row:.6f}, {input_col:.6f})"
    )
    print("Figura guardada en:", output_path)


if __name__ == "__main__":
    main()
