import numpy as np
import pygame
from numba import cuda
import math

# Constantes
WIDTH, HEIGHT = 800, 600
MAX_ITER = 100  # A más iteraciones mayor detalle

@cuda.jit
def compute_mandelbrot(min_x, max_x, min_y, max_y, width, height, max_iter, output_array):
    """
    Este es el "kernel" que se ejecuta en la GPU.
    Cada hilo de la GPU (miles de ellos) ejecutará esta función
    para un píxel diferente (pixel_x, pixel_y) al mismo tiempo.
    """
    
    # Pixel correspondiente a este hilo
    pixel_x, pixel_y = cuda.grid(2)
    
    # Limitamos a nuestra imagen
    if pixel_x < width and pixel_y < height:
        
        # 3. Coordenada del píxel (ej. 200, 300) 
        #    a un número complejo 'c' en el plano matemático (ej. -0.5, 0.2i)
        real = min_x + (pixel_x / width) * (max_x - min_x)
        imag = min_y + (pixel_y / height) * (max_y - min_y)
        
        # c es el número complejo constante para este píxel
        c_real = real
        c_imag = imag
        
        # z es el número que iteramos
        z_real = 0
        z_imag = 0
        
        # "Escape-Time" z = z^2 + c
        iter_count = 0
        for i in range(max_iter):
            # z^2 = (z_real + z_imag*i)^2 = (z_real^2 - z_imag^2) + (2 * z_real * z_imag)i
            z_real_sq = z_real * z_real
            z_imag_sq = z_imag * z_imag
            
            # Si magnitud de z es > 2 entonces escapa
            # (z_real^2 + z_imag^2) > 4  (es lo mismo que sqrt(z_real^2 + z_imag^2) > 2, pero optimizado)
            if (z_real_sq + z_imag_sq) > 4:
                break
                
            # Calculamos la parte imaginaria de z^2 + c
            z_imag = 2 * z_real * z_imag + c_imag
            # Calculamos la parte real de z^2 + c
            z_real = z_real_sq - z_imag_sq + c_real
            
            iter_count += 1
            
        # Guardar el resultado en el array de salida
        output_array[pixel_y, pixel_x] = iter_count


# Función de Coloreado (CPU)

def create_colormap(iterations, max_iter):
    """
    Convierte el array de 'iteraciones' (blanco y negro) en una imagen a color.
    """
    # Creamos un array vacío para la imagen en color (RGB)
    color_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Normalizamos (log para transicion suave)
    log_iter = np.log(iterations + 1)
    log_max_iter = np.log(max_iter + 1)

    norm_iter = log_iter / log_max_iter
    
    # Mapeo simple a colores
    color_image[..., 0] = (np.sin(0.05 * log_iter) * 127 + 128).astype(np.uint8) # Rojo
    color_image[..., 1] = (np.cos(0.03 * log_iter) * 127 + 128).astype(np.uint8) # Verde
    color_image[..., 2] = (np.sin(0.01 * log_iter + 0.5) * 127 + 128).astype(np.uint8) # Azul
    
    # Si llega al máximo la iteración, de negro
    mask = (iterations == max_iter)
    color_image[mask, 0] = 0
    color_image[mask, 1] = 0
    color_image[mask, 2] = 0
    
    return color_image

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fractal de Mandelbrot en GPU (con Numba + Pygame)")
    
    # Creamos el array de la imagen en la memoria del "Host" (CPU/RAM)
    # Lo inicializamos con ceros. 'np.uint32' para el recuento de iteraciones.
    host_image_array = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    
    # Creamos el array de la imagen en la memoria del "Device" (GPU/VRAM)
    # 'cuda.to_device' copia el array de la CPU a la GPU
    print("Moviendo el array de imagen a la VRAM de la GPU...")
    device_image_array = cuda.to_device(host_image_array)
    
    # Definimos los límites del plano complejo (la vista inicial)
    min_x, max_x = -2.0, 1.0
    min_y, max_y = -1.0, 1.0

    # Configuración de los bloques e hilos de CUDA
    # Queremos un hilo por cada píxel, en una cuadrícula 2D
    threads_per_block = (16, 16) # 16*16 = 256 hilos por "bloque"
    
    # Calculamos cuántos "bloques" necesitamos para cubrir toda la imagen
    blocks_x = math.ceil(WIDTH / threads_per_block[0])
    blocks_y = math.ceil(HEIGHT / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)
    
    print(f"Configuración de la GPU: {blocks_per_grid[0]*blocks_per_grid[1]} bloques, {threads_per_block[0]*threads_per_block[1]} hilos/bloque")
    print("Calculando el fractal en la GPU...")

    # La CPU NO espera, sigue adelante.
    compute_mandelbrot[blocks_per_grid, threads_per_block](
        min_x, max_x, min_y, max_y, WIDTH, HEIGHT, MAX_ITER, device_image_array
    )
    
    # Le decimos a la CPU: "Espera aquí hasta que la GPU haya terminado su cálculo".
    cuda.synchronize()
    print("¡Cálculo de la GPU completado!")
    
    # Copia el array de la GPU (VRAM) de vuelta a la CPU (RAM)
    host_image_array = device_image_array.copy_to_host()
    
    print("Generando colores...")
    # Convertimos los recuentos de iteraciones en una imagen de color (en la CPU)
    color_image = create_colormap(host_image_array, MAX_ITER)
    
    # Convertimos el array de NumPy en una superficie de Pygame
    # Pygame usa (ancho, alto) y NumPy usa (alto, ancho), así que giramos y volteamos
    surf = pygame.surfarray.make_surface(np.rot90(color_image))
    surf = pygame.transform.flip(surf, True, False) # Voltear horizontalmente

    print("Mostrando fractal. Cierra la ventana para salir.")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Dibujamos
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()