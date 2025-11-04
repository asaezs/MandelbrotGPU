import os
os.environ['SDL_VIDEODRIVER'] = 'windib'
import numpy as np
import pygame
from numba import cuda
import math

WIDTH, HEIGHT = 800, 600
MAX_ITER = 100

# CUDA
@cuda.jit
def compute_mandelbrot_kernel(min_x, max_x, min_y, max_y, width, height, max_iter, output_array):
    pixel_x, pixel_y = cuda.grid(2)
    if pixel_x < width and pixel_y < height:
        real = min_x + (pixel_x / width) * (max_x - min_x)
        imag = min_y + (pixel_y / height) * (max_y - min_y)
        c_real, c_imag = real, imag
        z_real, z_imag = 0, 0
        iter_count = 0
        for i in range(max_iter):
            z_real_sq = z_real * z_real
            z_imag_sq = z_imag * z_imag
            if (z_real_sq + z_imag_sq) > 4:
                iter_count = i
                break
            z_imag = 2 * z_real * z_imag + c_imag
            z_real = z_real_sq - z_imag_sq + c_real
            iter_count = i + 1
        output_array[pixel_y, pixel_x] = iter_count

# Coloreado en la CPU
def create_colormap(iterations, max_iter):
    color_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    log_iter = np.log(iterations + 1)
    
    color_image[..., 0] = (np.sin(0.05 * log_iter) * 127 + 128).astype(np.uint8)
    color_image[..., 1] = (np.cos(0.03 * log_iter) * 127 + 128).astype(np.uint8)
    color_image[..., 2] = (np.sin(0.01 * log_iter + 0.5) * 127 + 128).astype(np.uint8)
    
    mask = (iterations == max_iter)
    color_image[mask, 0] = 0
    color_image[mask, 1] = 0
    color_image[mask, 2] = 0
    
    return color_image

def calculate_fractal_once():
    print("Moviendo arrays a la GPU...")
    host_image_array = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    device_image_array = cuda.to_device(host_image_array)

    threads_per_block = (16, 16)
    blocks_x = math.ceil(WIDTH / threads_per_block[0])
    blocks_y = math.ceil(HEIGHT / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)

    print("Calculando fractal en la GPU...")
    compute_mandelbrot_kernel[blocks_per_grid, threads_per_block](
        -2.0, 1.0, -1.0, 1.0, WIDTH, HEIGHT, MAX_ITER, device_image_array
    )
    
    print("Esperando a la GPU...")
    # Sincronizamos cpu-gpu
    cuda.synchronize()
    
    print("Copiando datos de vuelta a la CPU...")
    host_image_array = device_image_array.copy_to_host()
    
    cuda.close() 
    
    print("Coloreando imagen...")
    color_image = create_colormap(host_image_array, MAX_ITER)
    
    surf = pygame.surfarray.make_surface(np.rot90(color_image))
    surf = pygame.transform.flip(surf, True, False)
    return surf

def main():
    fractal_surface = calculate_fractal_once()
    
    print("Iniciando Pygame...")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fractal (Numba + Pygame) - ¡NO INTERACTIVO!")

    screen.blit(fractal_surface, (0, 0))
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    pygame.quit()

if __name__ == "__main__":
    main()