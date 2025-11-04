import glfw
from OpenGL.GL import *
import numpy as np
import math

# SHADERS GLSL

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

FRAGMENT_SHADER = """
#version 420 core
out vec4 FragColor;
in vec2 v_coord;

uniform dvec4 u_view;
uniform int u_max_iter;
uniform vec3 u_thetas;

vec3 colormap(float x, vec3 thetas) {
    float pi = 3.14159265359;
    float r = 0.5 + 0.5 * sin((x + thetas.x) * 2.0 * pi);
    float g = 0.5 + 0.5 * sin((x + thetas.y) * 2.0 * pi);
    float b = 0.5 + 0.5 * sin((x + thetas.z) * 2.0 * pi);
    return vec3(r, g, b);
}

void main()
{
    double real = u_view.x + v_coord.x * (u_view.y - u_view.x);
    double imag = u_view.z + v_coord.y * (u_view.w - u_view.z);
    
    dvec2 c = dvec2(real, imag);
    dvec2 z = dvec2(0.0, 0.0);
    
    int iter_count = 0;
    
    for(int i = 0; i < u_max_iter; i++)
    {
        double z_real_sq = z.x * z.x;
        double z_imag_sq = z.y * z.y;
        
        if((z_real_sq + z_imag_sq) > 4.0) 
        {
            iter_count = i;
            break; 
        }
        
        z = dvec2(
            z_real_sq - z_imag_sq + c.x,
            2.0 * z.x * z.y + c.y
        );
        
        iter_count = i + 1;
    }
    
    // Coloreado
    vec3 color;
    if (iter_count == u_max_iter) {
        color = vec3(0.0, 0.0, 0.0); // Negro
    } else {
        
        double z_mag = length(z); 
        
        // float para usar log()
        float z_mag_float = float(z_mag);
        
        float smooth_iter = float(iter_count) + 1.0 - log(log(z_mag_float)) / log(2.0f);

        // Normalizamos
        float iter_norm = mod(smooth_iter / 50.0, 1.0);
        
        color = colormap(iter_norm, u_thetas);
    }
    
    FragColor = vec4(color, 1.0);
}
"""

view_state = {
    "min_x": np.float64(-2.0),
    "max_x": np.float64(1.0),
    "min_y": np.float64(-1.0),
    "max_y": np.float64(1.0),
    "thetas": [0.0, 0.15, 0.3],
    "max_iter_high": np.int32(200)
}
WIDTH, HEIGHT = 800, 600
# Iteraciones para la render de baja calidad
MAX_ITER_FAST = 20
last_zoom_time = 0.0
needs_high_quality_render = True
current_iter_setting = view_state["max_iter_high"]

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

# Eventos de zoom dinámico y render progresivo

def on_scroll(window, xoffset, yoffset):
    """
    Callback de la rueda del ratón.
    Actualiza coordenadas Y recalcula el MAX_ITER necesario.
    """
    global view_state, last_zoom_time, needs_high_quality_render, current_iter_setting, needs_recalculate
    
    mouse_x, mouse_y = glfw.get_cursor_pos(window)
    mouse_y = HEIGHT - mouse_y
    
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

    # Aumentamos las iteraciones a medida que el ancho de la vista disminuye
    base_iter = 100
    # Aumento logarítmico de iteraciones por zoom
    new_max_iter = int(base_iter + 50.0 * abs(math.log(new_width))) 
    
    # Limitamos para que nunca baje de 100, pero pueda subir hasta 5000 MAXIMO DE ITERACIONES
    view_state["max_iter_high"] = np.int32(max(100, min(new_max_iter, 5000)))

    last_zoom_time = glfw.get_time()
    needs_high_quality_render = True
    current_iter_setting = MAX_ITER_FAST
    needs_recalculate = True

# Renderizado
def render_frame(shader_program, vao, loc_view, loc_max_iter, loc_thetas, iter_count):
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader_program)
    
    glUniform4d(loc_view, view_state["min_x"], view_state["max_x"], view_state["min_y"], view_state["max_y"])
    glUniform1i(loc_max_iter, iter_count)
    
    thetas = view_state["thetas"]
    glUniform3f(loc_thetas, thetas[0], thetas[1], thetas[2])

    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

def main():
    global WIDTH, HEIGHT, last_zoom_time, needs_high_quality_render, current_iter_setting, needs_recalculate
    
    if not glfw.init():
        raise Exception("GLFW no se pudo inicializar")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(WIDTH, HEIGHT, "Fractal GLSL", None, None)
    if not window:
        glfw.terminate()
        raise Exception("La ventana de GLFW no se pudo crear. ¿Tu GPU soporta OpenGL 4.2?")

    glfw.make_context_current(window)
    glfw.swap_interval(0)
    
    print("Compilando shaders...")
    shader_program = create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
    quad_vao = create_quad_buffers()
    
    glfw.set_scroll_callback(window, on_scroll)

    glUseProgram(shader_program)
    loc_view = glGetUniformLocation(shader_program, "u_view")
    loc_max_iter = glGetUniformLocation(shader_program, "u_max_iter")
    loc_thetas = glGetUniformLocation(shader_program, "u_thetas")
    
    print("Todo listo, usa la rueda del ratón para hacer zoom.")

    needs_recalculate = True
    current_iter_setting = view_state["max_iter_high"]
    last_zoom_time = glfw.get_time()

    while not glfw.window_should_close(window):
        
        current_time = glfw.get_time()
        
        # Lógica de Render Progresivo
        if needs_high_quality_render and (current_time - last_zoom_time > 0.5):
            current_iter_setting = view_state["max_iter_high"]            
            needs_recalculate = True
            needs_high_quality_render = False
            print(f"Refinando (Iter: {current_iter_setting})")

        if needs_recalculate:
            render_frame(shader_program, quad_vao, loc_view, loc_max_iter, loc_thetas, current_iter_setting)
            glfw.swap_buffers(window)
            needs_recalculate = False
        else:
            glfw.wait_events_timeout(0.01)
        
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()