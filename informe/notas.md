### Seguimiento de un píxel

Se seleccionó un píxel ubicado dentro de una pluma azul del ala, con valor RGB [78, 98, 115].

En HS se obtuvo H = 207.57°, S = 0.3217 y V = 0.4510. Para este tono, la interpolación entregó m(h) = 1.3446. Al aplicar g_m(S) = mS, la saturación aumentó a 0.4326 y el RGB final fue [65, 92, 115].

Para el mismo píxel, en LCh se obtuvo L* = 40.27, c* = 14.93 y h* = 254.71°. En este caso m(h*) = 1.1865, por lo que el nuevo croma fue 17.72 y el RGB final fue [74, 99, 119].

Esto muestra que un mismo píxel no tiene el mismo valor de tono en HSV y LCh. Por esta razón, incluso usando los mismos puntos de control, el valor interpolado de m puede ser diferente y el resultado visual también cambia.




### Comparación de g_m

La función principal usada en la herramienta fue g_m(S) = mS, ya que su efecto es directo y fácil de interpretar. Con m = 1 la saturación no cambia, con m > 1 aumenta y con m < 1 disminuye.

También se probó una alternativa basada en la MTF revisada en la cápsula. Para mantener m = 1 como valor neutro se utilizó p = 1/(1+m).

La diferencia principal aparece cerca de los extremos. Por ejemplo, con m = 1.5 y S = 0.8, la función lineal produce un valor mayor que 1 y debe limitarse, mientras que la MTF entrega 0.857 sin llegar al límite.

La transformación lineal genera cambios más directos y fuertes. La MTF modifica más suavemente los valores cercanos a 0 y 1, por lo que reduce la aparición de clipping. Visualmente la diferencia es más pequeña que en las curvas, pero se puede observar que el resultado de la MTF mantiene mejor algunos colores que ya estaban bastante saturados.