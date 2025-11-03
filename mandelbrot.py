import glfw
from OpenGL.GL import *
import numpy as np
import math

# --- 1. CÓDIGO DE LOS SHADERS (GLSL) ---

# VERTEX SHADER (Sin cambios)
VERTEX_SHADER = """
#version 420 core
layout (location = 0) in vec2 aPos;
out vec2 v_coord;
void main()
{
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    v_coord = aPos * 0.5 + 0.5;
}
"""

# --- MODIFICADO: Fragment Shader (Precisión Doble + Mejor Color) ---
FRAGMENT_SHADER = """
#version 420 core // <-- Versión 4.20 para soportar 'double' (dvec)
out vec4 FragColor;
in vec2 v_coord;

// "Uniformes"
// --- ¡CAMBIO A DOUBLE! ---
// Usamos dvec4 (vector de 4 doubles) para la vista
uniform dvec4 u_view; // (min_x, max_x, min_y, max_y)
uniform int u_max_iter;
uniform vec3 u_thetas; // Para el nuevo algoritmo de color

// --- NUEVA FUNCIÓN DE COLOR (basada en tu idea) ---
// Traducimos tu función de colormap a GLSL
vec3 colormap(float x, vec3 thetas) {
    float pi = 3.1415926535;
    // (x + thetas[0]) * 2 * math.pi
    float r = 0.5 + 0.5 * sin((x + thetas.x) * 2.0 * pi);
    float g = 0.5 + 0.5 * sin((x + thetas.y) * 2.0 * pi);
    float b = 0.5 + 0.5 * sin((x + thetas.z) * 2.0 * pi);
    return vec3(r, g, b);
}
// --- FIN DE LA FUNCIÓN ---

void main()
{
    // 1. Mapeo al plano complejo usando precisión DOBLE
    // --- ¡CAMBIO A DOUBLE! ---
    double real = u_view.x + v_coord.x * (u_view.y - u_view.x);
    double imag = u_view.z + v_coord.y * (u_view.w - u_view.z);
    
    dvec2 c = dvec2(real, imag);
    dvec2 z = dvec2(0.0, 0.0);
    // --- FIN DEL CAMBIO ---
    
    // 2. Algoritmo Escape-Time
    int iter_count = 0;
    for(int i = 0; i < u_max_iter; i++)
    {
        // --- ¡CAMBIO A DOUBLE! ---
        double z_real_sq = z.x * z.x;
        double z_imag_sq = z.y * z.y;
        
        // Usamos 4.0 (double) en lugar de 4.0f (float)
        if((z_real_sq + z_imag_sq) > 4.0) 
        {
            break;
        }
        
        z = dvec2(
            z_real_sq - z_imag_sq + c.x,
            2.0 * z.x * z.y + c.y
        );
        // --- FIN DEL CAMBIO ---
        iter_count++;
    }
    
    // 3. Coloreado
    vec3 color;
    if (iter_count == u_max_iter) {
        color = vec3(0.0, 0.0, 0.0); // Negro
    } else {
        // --- ¡NUEVO COLOREADO! ---
        // Normalizamos la iteración (0.0 a 1.0)
        float iter_norm = float(iter_count) / float(u_max_iter);
        color = colormap(iter_norm, u_thetas);
        // --- FIN DEL CAMBIO ---
    }
    
    FragColor = vec4(color, 1.0);
}
"""

# --- 2. Variables Globales de Estado ---
# --- ¡CAMBIO A DOUBLE! ---
# Usamos np.float64 (double de 64 bits) para guardar el estado en Python
view_state = {
    "min_x": np.float64(-2.0),
    "max_x": np.float64(1.0),
    "min_y": np.float64(-1.0),
    "max_y": np.float64(1.0),
    "thetas": [0.3, 0.4, 0.5] # Fases de color (¡juega con estos valores!)
}
WIDTH, HEIGHT = 800, 600

# --- ¡NUEVO! Constantes para Renderizado Progresivo ---
MAX_ITER_FAST = 15   # Iteraciones para la previsualización (mientras se hace zoom)
MAX_ITER_HIGH = 200  # Iteraciones para la imagen final (cuando se para)
last_zoom_time = 0.0
needs_high_quality_render = True
current_iter_setting = MAX_ITER_HIGH

# --- 3. Funciones de Ayuda de OpenGL (Sin Cambios) ---

def create_shader_program(vertex_source, fragment_source):
    
    def compile_shader(source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(shader).decode()
            raise Exception(f"Error compilando shader: {log}")
        return shader

    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode()
        raise Exception(f"Error enlazando programa: {log}")
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return program

def create_quad_buffers():
    vertices = np.array([
        -1.0, -1.0, 1.0, -1.0, 1.0,  1.0, -1.0,  1.0
    ], dtype=np.float32)
    
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), None)
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao

# --- 4. Callback de Eventos (Zoom) (¡MODIFICADO!) ---

def on_scroll(window, xoffset, yoffset):
    """
    Callback de la rueda del ratón.
    Esto NO recalcula, solo actualiza las coordenadas y pide un recalculo.
    """
    global view_state, last_zoom_time, needs_high_quality_render, current_iter_setting
    
    mouse_x, mouse_y = glfw.get_cursor_pos(window)
    mouse_y = HEIGHT - mouse_y
    
    # --- ¡CAMBIO A DOUBLE! ---
    current_width = view_state["max_x"] - view_state["min_x"]
    current_height = view_state["max_y"] - view_state["min_y"]
    
    target_real = view_state["min_x"] + (mouse_x / WIDTH) * current_width
    target_imag = view_state["min_y"] + (mouse_y / HEIGHT) * current_height
    
    zoom_factor = np.float64(0.9) if yoffset > 0 else np.float64(1.10)
    new_width = current_width * zoom_factor
    new_height = current_height * zoom_factor
    
    x_percent = np.float64(mouse_x / WIDTH)
    y_percent = np.float64(mouse_y / HEIGHT)
    
    view_state["min_x"] = target_real - new_width * x_percent
    view_state["max_x"] = target_real + new_width * (1 - x_percent)
    view_state["min_y"] = target_imag - new_height * y_percent
    view_state["max_y"] = target_imag + new_height * (1 - y_percent)
    # --- FIN DEL CAMBIO ---

    # --- ¡NUEVA LÓGICA DE RENDER PROGRESIVO! ---
    last_zoom_time = glfw.get_time()        # Guardamos la hora del último zoom
    needs_high_quality_render = True        # Pedimos un render de alta calidad (para *después*)
    current_iter_setting = MAX_ITER_FAST    # Pero POR AHORA, renderizamos rápido
    # --- FIN DE LA LÓGICA ---

# --- 5. Función de Renderizado (¡NUEVA!) ---
def render_frame(shader_program, vao, loc_view, loc_max_iter, loc_thetas, iter_count):
    """
    Dibuja un solo fotograma con la configuración actual.
    """
    # 1. Limpiar y activar el shader
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader_program)
    
    # 2. Enviar los "uniforms" (coordenadas, iteraciones, colores) a la GPU
    # --- ¡CAMBIO A DOUBLE! ---
    glUniform4d(loc_view, view_state["min_x"], view_state["max_x"], view_state["min_y"], view_state["max_y"])
    
    glUniform1i(loc_max_iter, iter_count)
    
    thetas = view_state["thetas"]
    glUniform3f(loc_thetas, thetas[0], thetas[1], thetas[2])

    # 3. Dibujar el cuadrado (que ejecuta el shader para cada píxel)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

# --- 6. Función Principal (¡MODIFICADA!) ---
def main():
    global WIDTH, HEIGHT, last_zoom_time, needs_high_quality_render, current_iter_setting
    
    if not glfw.init():
        raise Exception("GLFW no se pudo inicializar")

    # Pedimos un contexto 4.2 para asegurar el soporte de 'double'
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(WIDTH, HEIGHT, "Fractal GLSL (Doble Precisión + Render Progresivo)", None, None)
    if not window:
        glfw.terminate()
        raise Exception("La ventana de GLFW no se pudo crear. ¿Tu GPU soporta OpenGL 4.2?")

    glfw.make_context_current(window)
    
    # ¡IMPORTANTE! Desactivamos V-Sync
    # (Ya no lo necesitamos, controlaremos el refresco manualmente)
    glfw.swap_interval(0)
    
    print("Compilando shaders...")
    shader_program = create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
    quad_vao = create_quad_buffers()
    
    glfw.set_scroll_callback(window, on_scroll)

    # Localizamos los "uniforms" UNA SOLA VEZ
    glUseProgram(shader_program)
    loc_view = glGetUniformLocation(shader_program, "u_view")
    loc_max_iter = glGetUniformLocation(shader_program, "u_max_iter")
    loc_thetas = glGetUniformLocation(shader_program, "u_thetas")
    
    print("¡Iniciado! Usa la rueda del ratón para hacer zoom.")

    # --- BUCLE PRINCIPAL (LÓGICA PROGRESIVA) ---
    needs_recalculate = True
    last_zoom_time = glfw.get_time()

    while not glfw.window_should_close(window):
        
        current_time = glfw.get_time()
        
        # --- Lógica de Render Progresivo ---
        # ¿El usuario ha parado de hacer zoom?
        if needs_high_quality_render and (current_time - last_zoom_time > 0.5): # 0.5 segundos de espera
            current_iter_setting = MAX_ITER_HIGH
            needs_recalculate = True
            needs_high_quality_render = False # Ya no lo necesitamos (hasta el prox. zoom)
            print("--- Refinando (Alta Calidad) ---")

        # ¿Hay algo que (re)dibujar?
        if needs_recalculate:
            # Renderizamos el fotograma con la configuración actual
            render_frame(shader_program, quad_vao, loc_view, loc_max_iter, loc_thetas, current_iter_setting)
            
            # Mostramos el resultado
            glfw.swap_buffers(window)
            
            # Marcamos como "limpio"
            needs_recalculate = False
        else:
            # --- ¡ARREGLO DE GPU AL 100%! ---
            # Si no hay nada que hacer, le decimos al bucle que "duerma"
            # y espere eventos (como el zoom).
            # Esto reduce el uso de CPU/GPU a casi 0% cuando está inactivo.
            glfw.wait_events_timeout(0.01) # Espera 10ms o hasta un evento
        
        # Procesar eventos (siempre)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()