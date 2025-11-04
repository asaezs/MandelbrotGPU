# 🌌 Visualizador del Fractal de Mandelbrot en GPU (Python + GLSL)

Un explorador interactivo en tiempo real del fractal de Mandelbrot, calculado enteramente en la GPU usando Python, PyOpenGL y shaders GLSL. Este proyecto renderiza el fractal a altas velocidades de fotogramas, permitiendo un zoom "infinito" gracias al uso de precisión de 64 bits (double) en los shaders.

## 📸 Vistazo

[<img src="FractalCompleto.png">]

[<img src="ZoomALaIzq.png">]

[<img src="Espiral.png">]

---

## ✨ Características Principales

Este proyecto fue un ejercicio para aprender la computación moderna en GPU y la visualización en tiempo real, evitando las librerías de cálculo (como Numba/PyCUDA) y los motores de renderizado (como Pygame) que pueden causar conflictos de contexto.

* **Renderizado 100% en GPU:** El fractal se calcula y colorea en tiempo real para cada píxel usando un **Fragment Shader de GLSL**. Los datos nunca salen de la VRAM.
* **Zoom "Infinito" (Doble Precisión):** El shader utiliza la versión `#version 420 core` de GLSL para realizar todos los cálculos con `double` (64 bits) en lugar de `float` (32 bits). Esto evita la pixelación y la pérdida de definición que ocurre en zooms profundos con precisión simple.
* **Renderizado Progresivo:** Para mantener la interactividad y la eficiencia:
    * **Render Rápido:** Mientras se hace zoom (moviendo la rueda del ratón), el fractal se recalcula con un número bajo de iteraciones (`MAX_ITER_FAST`).
    * **Render de Calidad:** 0.5 segundos *después* de dejar de hacer zoom, el fractal se refina automáticamente con un número de iteraciones mucho mayor (`max_iter_high`).
* **Iteraciones Dinámicas:** El número de iteraciones para el render de alta calidad no es fijo. Aumenta logarítmicamente a medida que el zoom es más profundo (`new_max_iter = int(base_iter + 50.0 * abs(math.log(new_width)))`), revelando más detalle en zonas complejas.
* **Coloreado Suave (Smooth Coloring):** Utiliza una fórmula de `log(log(z_mag))` en el shader para calcular un valor de iteración fraccionario. Esto elimina las "bandas" de color y crea los gradientes suaves y detallados que se ven en las imágenes.
* **Uso Eficiente de la GPU (0% Inactivo):** El bucle principal utiliza `glfw.wait_events_timeout(0.01)`. Esto "duerme" la aplicación y reduce el uso de la GPU a casi 0% cuando no se está interactuando, evitando que el ventilador de la gráfica se dispare innecesariamente.

---

## 🛠️ Cómo Funciona

La aplicación se divide en dos partes:

1.  **Python (El Orquestador - CPU):**
    * Usa `glfw` para crear una ventana y un contexto de OpenGL 4.2.
    * Usa `PyOpenGL` para compilar los shaders GLSL y crear un rectángulo ("quad") que llena la pantalla.
    * Escucha los eventos de la rueda del ratón (`on_scroll`) para calcular las nuevas coordenadas de la vista.
    * Gestiona la lógica del renderizado progresivo (cuándo usar `MAX_ITER_FAST` vs. `MAX_ITER_HIGH`).
    * En cada fotograma, envía las variables de estado (coordenadas, iteraciones) a la GPU a través de `uniforms`.

2.  **GLSL (El Músculo - GPU):**
    * El `VERTEX_SHADER` es simple: solo dibuja el rectángulo en la pantalla.
    * El `FRAGMENT_SHADER` hace todo el trabajo pesado. Se ejecuta en paralelo para **cada píxel** de la pantalla:
        * Convierte la coordenada del píxel (ej. `[250, 400]`) a un número complejo (`c`) usando las coordenadas (`u_view`) y la precisión de `double`.
        * Ejecuta el algoritmo de "escape-time" ($z = z^2 + c$) para ese punto.
        * Calcula el `smooth_iter` para obtener un valor de color suave.
        * Pasa ese valor a la función `colormap` para obtener un color RGB vibrante.
        * Devuelve el color final (`FragColor`).

---

## ⌨️ Controles

* **Rueda del Ratón:** Hacer zoom (centrado en el cursor).
* **Cerrar Ventana:** Salir de la aplicación.

---

## ⚙️ Instalación y Ejecución

Este script requiere Python 3 y una tarjeta gráfica que soporte **OpenGL 4.2** o superior (necesario para la precisión de 64 bits `double` en los shaders).

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
    cd tu-repositorio
    ```

2.  **Crear un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (En Linux/macOS)
    .\venv\Scripts\activate   # (En Windows)
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install numpy glfw PyOpenGL PyOpenGL_accelerate
    ```

4.  **Ejecutar el script:**
    ```bash
    python mandelbrot.py
    ```
