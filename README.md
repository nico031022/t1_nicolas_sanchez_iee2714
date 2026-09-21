# T1

Este repositorio contiene las preguntas 1, 2 y 3

## Instalación

El código está en Python (3.11). Las librerías necesarias son NumPy, Matplotlib y OpenCV.

Después de clonar el repositorio, tiene que abrir una terminal en la carpeta que contiene este README e instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Todos los comandos de abajo se ejecutan desde esa misma carpeta, usando `python -m`. 

## Organización

- `src/`: implementación de los algoritmos de cada pregunta.
- `experiments/`: scripts para hacer las pruebas y generar las figuras.
- `images/profesor/`: imágenes del profesor.
- `images/propias/`: imagen adicional(`fruit.jpg`).
- `results/`: resultados de cada pregunta y experimento.
- `informe/`: notas para el informe.

Cada pregunta usa la imagen del profesor y la imagen de frutas. Los parámetros están definidos dentro de cada script de `experiments/` y para reproducir los resultados se pueden dejar tal como están. Al ejecutar un experimento, sus archivos de salida se vuelven a guardar con los mismos nombres.

## P1

La implementación está en `src/pregunta1.py`. Se puede trabajar en HSV (modo `HS`) o en LCh. Los puntos de control definen una curva periódica de tono y el valor neutro de `m` es 1.

Para generar las comparaciones con ambas imágenes y los experimentos de parámetros:

```bash
python -m experiments.test_p1_profesor
python -m experiments.test_p1_fruit
python -m experiments.test_p1_m_range
python -m experiments.test_p1_position
python -m experiments.test_p1_control_points
python -m experiments.test_p1_gm
python -m experiments.test_p1_clipping
python -m experiments.test_p1_pixel
```

Estos scripts comparan los 2 espacios de color, la amplitud y el rango de `m`, la posición y cantidad de puntos de control, y dos formas de definir `g_m`. Y también incluyen un caso de clipping y el seguimiento de un píxel.

Las figuras quedan en `results/pregunta1/` y sus subcarpetas. Los valores del píxel se guardan además en `results/pregunta1/pixel_trace.txt`.

## P2

La implementación está en `src/pregunta2.py`. El tamaño y la separación de las regiones se controlan con `region_size` y `region_step`. Para controlar el contraste se mezcla la transformación local con la identidad: `contrast_strength=0` conserva la imagen y `contrast_strength=1` aplica la ecualización local sin limitar.

Para reproducir las comparaciones y los experimentos:

```bash
python -m experiments.test_p2_local
python -m experiments.test_p2_global_equivalence
python -m experiments.test_p2_clahe
python -m experiments.test_p2_clahe_configs
python -m experiments.test_p2_region_size
python -m experiments.test_p2_overlap
python -m experiments.test_p2_mesh_density
python -m experiments.test_p2_bins
python -m experiments.test_p2_contrast_control
python -m experiments.test_p2_bad_case
python -m experiments.test_p2_boundaries
python -m experiments.test_p2_runtime
```

La comparación principal muestra la imagen original, la ecualización global propia, la local sin limitar, la propuesta con control de contraste y CLAHE. El experimento `global_equivalence` comprueba que una sola región que cubre toda la imagen reproduce la ecualización global propia.

Los resultados quedan en `results/pregunta2/`. Y los tiempos de ejecución se muestran en la terminal y en la figura del experimento `runtime`; pueden variar según el pc.

## P3

La implementación está en `src/pregunta3.py`. La función `resize_image` permite elegir vecino más cercano (`nearest`) o interpolación bilineal (`bilinear`), con factores entre 0.5 y 2. Se usan los centros de los píxeles para relacionar las coordenadas de entrada y salida.

Primero hay que generar las imágenes reescaladas:

```bash
python -m experiments.test_p3_scaling
```

Este script aplica los factores 0.6, 0.8, 1.3 y 1.7 con ambos métodos y guarda las imágenes en `results/pregunta3/scaling/`. Puede tardar varios minutos porque el reescalado recorre los píxeles de salida.

Con esas imágenes listas, generar las comparaciones y recortes:

```bash
python -m experiments.test_p3_nearest_bilinear
python -m experiments.test_p3_reduction
python -m experiments.test_p3_enlargement
python -m experiments.test_p3_aliasing
```

Estos 4 scripts leen los archivos de `scaling/`. Si se usan los resultados que ya están incluidos en el repositorio, se pueden ejecutar directamente.

Los siguientes experimentos leen las imágenes originales y se pueden ejecutar por separado:

```bash
python -m experiments.test_p3_down_up
python -m experiments.test_p3_successive
python -m experiments.test_p3_coordinate_convention
python -m experiments.test_p3_pixel_trace
```

Se compara una reducción seguida de una ampliación, varios reescalados frente a uno equivalente, y dos convenciones de coordenadas. El último script imprime las coordenadas, los 4 vecinos, sus pesos y el valor final de un píxel. Las figuras quedan en las subcarpetas de `results/pregunta3/`.

## Pruebas básicas

También se pueden ejecutar estas tests de interpolación periódica, histogramas y cobertura de la malla:

```bash
python -m experiments.test_p1
python -m experiments.test_p2_cdf
python -m experiments.test_p2_grid
```

`test_p1` abre un gráfico de la interpolación; basta con cerrar la ventana para que termine.

## Referencia

Para las conversiones de color y la comparación con MTF me apoyé en las [cápsulas del curso](https://github.com/Kevaley/fundamentos-img), especialmente la de la Semana 3. En la pregunta 2, CLAHE de OpenCV se usa solamente como referencia para comparar con el método propio.
