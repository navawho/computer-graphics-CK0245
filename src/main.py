import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
from PIL import Image

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

GLOBAL_X = np.array([1,0,0], dtype=np.float32)
GLOBAL_Y = np.array([0,1,0], dtype=np.float32) 
GLOBAL_Z = np.array([0,0,1], dtype=np.float32)

ENTITY_TYPE = {
    "BAR_COUNTER": 0,
    "BEER": 1,
    "WHISKY": 2,
    "FAN": 3,
    "WALLS": 4,
    "FLOORS": 5,
}

UNIFORM_TYPE = {
    "MODEL": 0,
    "VIEW": 1,
    "PROJECTION": 2,
    "CAMERA_POS": 3,
    "LIGHT_COLOR": 4,
    "LIGHT_POS": 5,
    "LIGHT_STRENGTH": 6,
    "TINT": 7,
}

PIPELINE_TYPE = {
    "STANDARD": 0,
    "EMISSIVE": 1,
}

def create_shader(vertex_filepath, fragment_filepath):
    with open(vertex_filepath,'r') as f:
        vertex_src = f.readlines()

    with open(fragment_filepath,'r') as f:
        fragment_src = f.readlines()
    
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))
    
    return shader

def read_obj(filename):
    v = []
    vt = []
    vn = []
    vertices = []

    with open(filename, "r") as file:
        line = file.readline()

        while line:
            words = line.split(" ")

            # coordenadas dos vertices
            if words[0] == "v":
                vertex = [
                    float(words[1]),
                    float(words[2]),
                    float(words[3])
                ]
                v.append(vertex)

            # coordenadas de textura
            if words[0] == "vt":
                vertex_t = [
                    float(words[1]),
                    float(words[2])
                ]
                vt.append(vertex_t)

            # normais do vertice
            if words[0] == "vn":
                vertex_n = [
                    float(words[1]),
                    float(words[2]),
                    float(words[3])
                ]
                vn.append(vertex_n)

            # faces do objeto
            if words[0] == "f":
                for i in range(len(words) - 3):
                    read_face(words[1], v, vt, vn, vertices)
                    read_face(words[2 + i], v, vt, vn, vertices)
                    read_face(words[3 + i], v, vt, vn, vertices)

            line = file.readline()

    return vertices

def read_face(face_description, v, vt, vn, vertices):
    v_vt_vn = face_description.split("/")
    
    for element in v[int(v_vt_vn[0]) - 1]:
        vertices.append(element)
    for element in vt[int(v_vt_vn[1]) - 1]:
        vertices.append(element)
    for element in vn[int(v_vt_vn[2]) - 1]:
        vertices.append(element)

def perspective_projection(fovy, aspect, near, far):
    """
    E 0 A 0
    0 F B 0
    0 0 C D
    0 0-1 0

    A = (right+left)/(right-left)
    B = (top+bottom)/(top-bottom)
    C = -(far+near)/(far-near)
    D = -2*far*near/(far-near)
    E = 2*near/(right-left)
    F = 2*near/(top-bottom)
    """

    ymax = near * np.tan(fovy * np.pi / 360.0)
    xmax = ymax * aspect

    left = -xmax
    right = xmax
    bottom = -ymax
    top = ymax

    A = (right + left) / (right - left)
    B = (top + bottom) / (top - bottom)
    C = -(far + near) / (far - near)
    D = -2. * far * near / (far - near)
    E = 2. * near / (right - left)
    F = 2. * near / (top - bottom)

    return np.array((
        (  E, 0., 0., 0.),
        ( 0.,  F, 0., 0.),
        (  A,  B,  C,-1.),
        ( 0., 0.,  D, 0.),
    ), dtype=np.float32)

# classe base para as entidades do jogo
class Entity:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

    def update(self, dt, camera_pos):
        pass

    def get_model_transform(self):
        mt = np.identity(4, dtype=np.float32)

        # rotacao no eixo y
        angle_rad = np.radians(self.eulers[1])
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0.0, np.sin(angle_rad), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angle_rad), 0.0, np.cos(angle_rad), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        mt = np.dot(mt, rotation_matrix)

        # rotacao no eixo z
        angle_rad = np.radians(self.eulers[2])
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0, 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        mt = np.dot(mt, rotation_matrix)


        # translacao 
        translation = np.identity(4, dtype=np.float32)
        translation[3, 0:3] = self.position[:3]

        return np.dot(mt, translation)

# camera em primeira pessoa
class Camera(Entity):
    def __init__(self, position):
        super().__init__(position, eulers = [0,0,0])
        self.update(0)
    
    def update(self, dt):
        theta = self.eulers[2]
        phi = self.eulers[1]

        self.forwards = np.array(
            [
                np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(phi))
            ],
            dtype = np.float32
        )

        self.right = np.cross(self.forwards, GLOBAL_Z)

        self.up = np.cross(self.right, self.forwards)

    def get_view_transform(self):
        eye = np.array(self.position, dtype=np.float32)
        target = np.array(self.position + self.forwards, dtype=np.float32)
        up = np.array(self.up, dtype=np.float32)

        f = target - eye
        f = f / np.linalg.norm(f)

        r = np.cross(f, up)
        r = r / np.linalg.norm(r)

        u = np.cross(r, f)
        u = u / np.linalg.norm(u)


        return np.array((
            (r[0], u[0], -f[0], 0.),
            (r[1], u[1], -f[1], 0.),
            (r[2], u[2], -f[2], 0.),
            (-np.dot(r, eye), -np.dot(u, eye), np.dot(f, eye), 1.0)
        ), dtype=np.float32)
    
    # move a camera
    def move(self, d_pos):
        self.position += d_pos[0] * self.forwards \
                        + d_pos[1] * self.right \
                        + d_pos[2] * self.up
    
        # z nao muda (personagem nao pula/voa)
        self.position[2] = 2
    
    # rotaciona a camera
    def spin(self, d_eulers):
        self.eulers += d_eulers

        self.eulers[0] %= 360
        # limite de 90 graus ao olhar pra cima ou pra baixo
        self.eulers[1] = min(89, max(-89, self.eulers[1]))
        self.eulers[2] %= 360

class BarCounter(Entity):
    def __init__(self, position, eulers):
        super().__init__(position, eulers)
    
    def update(self, dt, camera_pos):
        pass

class Beer(Entity):
    def __init__(self, position, eulers):
        super().__init__(position, eulers)
    
    def update(self, dt, camera_pos):
        pass

class Whisky(Entity):
    def __init__(self, position, eulers):
        super().__init__(position, eulers)
    
    def update(self, dt, camera_pos):
        pass

class Fan(Entity):
    def __init__(self, position, eulers):
        super().__init__(position, eulers)
        self.rotation = 'left'
    
    def update(self, dt, camera_pos):
        if self.rotation == 'right':
            self.eulers[2] += 0.25 * dt

            if self.eulers[2] > 30:
                self.rotation = 'left'
        else:
            self.eulers[2] -= 0.25 * dt

            if self.eulers[2] < -30:
                self.rotation = 'right'


class Walls(Entity):
    def __init__(self, position, eulers):
        super().__init__(position, eulers)
    
    def update(self, dt, camera_pos):
        pass

class Floors(Entity):
    def __init__(self, position, eulers):
        super().__init__(position, eulers)
    
    def update(self, dt, camera_pos):
        pass

# lida com as entidades
class Scene:
    def __init__(self):

        self.entities = {
            ENTITY_TYPE["BAR_COUNTER"]: [
                BarCounter(position = [6,0,0], eulers = [0,0,0]),
            ],
            ENTITY_TYPE["BEER"]: [
                Beer(position = [6,0,0], eulers = [0,0,0]),
            ],
            ENTITY_TYPE["WHISKY"]: [
                Whisky(position = [6,0,0], eulers = [0,0,0]),
            ],
            ENTITY_TYPE["FAN"]: [
                Fan(position = [6,0,1.93], eulers = [0,0,0]),
            ],  
            ENTITY_TYPE["WALLS"]: [
                Walls(position = [6,0,0], eulers = [0,0,0]),
            ],
            ENTITY_TYPE["FLOORS"]: [
                Floors(position = [6,0,0], eulers = [0,0,0]),
            ],
        }

        self.lights = []

        self.player = Camera(
            position = [0,0,2]
        )

    def update(self, dt):
        for entities in self.entities.values():
            for entity in entities:
                entity.update(dt, self.player.position)
        
        for light in self.lights:
            light.update(dt, self.player.position)

        self.player.update(dt)

    def move_player(self, d_pos):
        self.player.move(d_pos)
    
    def spin_player(self, d_eulers):
        self.player.spin(d_eulers)

# classe main
class App:
    def __init__(self):
        self._set_up_glfw()

        self._set_up_timer()

        self._set_up_input_systems()

        self._create_assets()

    def _set_up_glfw(self) -> None:
        glfw.init()
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR,3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR,3)
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, 
            GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER,GL_FALSE) 
        self.window = glfw.create_window(
            SCREEN_WIDTH, SCREEN_HEIGHT, "Title", None, None)
        glfw.make_context_current(self.window)
    
    def _set_up_timer(self) -> None:
        self.last_time = glfw.get_time()
        self.current_time = 0
        self.frames_rendered = 0
        self.frametime = 0.0
    
    def _set_up_input_systems(self) -> None:
        glfw.set_input_mode(
            self.window, 
            GLFW_CONSTANTS.GLFW_CURSOR, 
            GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN
        )

        self._keys = {}
        glfw.set_key_callback(self.window, self._key_callback)
    
    def _key_callback(self, window, key, scancode, action, mods) -> None:
        state = False
        if action == GLFW_CONSTANTS.GLFW_PRESS:
            state = True
        elif action == GLFW_CONSTANTS.GLFW_RELEASE:
            state = False
        else:
            return

        self._keys[key] = state
    
    def _create_assets(self) -> None:
        self.renderer = GraphicsEngine()

        self.scene = Scene()
    
    def run(self) -> None:
        running = True
        while (running):
            if glfw.window_should_close(self.window) \
                or self._keys.get(GLFW_CONSTANTS.GLFW_KEY_ESCAPE, False):
                running = False
            
            self._handle_keys()
            self._handle_mouse()

            glfw.poll_events()

            self.scene.update(self.frametime / 16.67)
            
            self.renderer.render(
                self.scene.player, self.scene.entities, self.scene.lights)

            #timing
            self._calculate_framerate()

    def _handle_keys(self) -> None:
        rate = 0.005*self.frametime
        d_pos = np.zeros(3, dtype=np.float32)

        if self._keys.get(GLFW_CONSTANTS.GLFW_KEY_W, False):
            d_pos += GLOBAL_X
        if self._keys.get(GLFW_CONSTANTS.GLFW_KEY_A, False):
            d_pos -= GLOBAL_Y
        if self._keys.get(GLFW_CONSTANTS.GLFW_KEY_S, False):
            d_pos -= GLOBAL_X
        if self._keys.get(GLFW_CONSTANTS.GLFW_KEY_D, False):
            d_pos += GLOBAL_Y

        length = np.linalg.norm(d_pos)

        if abs(length) < 0.00001:
            return

        d_pos = rate * d_pos / length

        self.scene.move_player(d_pos)

    def _handle_mouse(self) -> None:

        (x,y) = glfw.get_cursor_pos(self.window)
        d_eulers = 0.02 * ((SCREEN_WIDTH / 2) - x) * GLOBAL_Z
        d_eulers += 0.02 * ((SCREEN_HEIGHT / 2) - y) * GLOBAL_Y
        self.scene.spin_player(d_eulers)
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    def _calculate_framerate(self) -> None:
        self.current_time = glfw.get_time()
        delta = self.current_time - self.last_time
        if (delta >= 1):
            framerate = max(1,int(self.frames_rendered/delta))
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.last_time = self.current_time
            self.frames_rendered = -1
            self.frametime = float(1000.0 / max(1,framerate))
        self.frames_rendered += 1

    def quit(self):
        
        self.renderer.destroy()

# classe responsavel por desenhar
class GraphicsEngine:
    def __init__(self):
        self._set_up_opengl()

        self._create_assets()

        self._set_onetime_uniforms()

        self._get_uniform_locations()
    
    def _set_up_opengl(self):
        glClearColor(0.1, 0.1, 0.1, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def _create_assets(self):
        self.meshes = {
            ENTITY_TYPE["BAR_COUNTER"]: Mesh("models/bar_counter.obj"),
            ENTITY_TYPE["BEER"]: Mesh("models/beer.obj"),
            ENTITY_TYPE["WHISKY"]: Mesh("models/whisky.obj"),
            ENTITY_TYPE["FAN"]: Mesh("models/fan.obj"),
            ENTITY_TYPE["WALLS"]: Mesh("models/walls.obj"),
            ENTITY_TYPE["FLOORS"]: Mesh("models/floors.obj"),
        }

        self.materials = {
            ENTITY_TYPE["BAR_COUNTER"]: Material("gfx/bar_counter.png"),
            ENTITY_TYPE["BEER"]: Material("gfx/beer.jpg"),
            ENTITY_TYPE["WHISKY"]: Material("gfx/whisky.jpg"),
            ENTITY_TYPE["FAN"]: Material("gfx/fan.jpg"),
            ENTITY_TYPE["WALLS"]: Material("gfx/wall.jpg"),
            ENTITY_TYPE["FLOORS"]: Material("gfx/floor.jpg"),
        }
        
        self.shaders = {
            PIPELINE_TYPE["STANDARD"]: Shader(
                "shaders/vertex.txt", "shaders/fragment.txt"),
            PIPELINE_TYPE["EMISSIVE"]: Shader(
                "shaders/vertex_light.txt", "shaders/fragment_light.txt"),
        }
    
    def _set_onetime_uniforms(self):
        projection_transform = perspective_projection(
            fovy = 45, aspect = 640/480, 
            near = 0.1, far = 50
        )

        for shader in self.shaders.values():
            shader.use()
            glUniform1i(glGetUniformLocation(shader.program, "imageTexture"), 0)

            glUniformMatrix4fv(
                glGetUniformLocation(shader.program,"projection"),
                1, GL_FALSE, projection_transform
            )

    def _get_uniform_locations(self) -> None:
        shader = self.shaders[PIPELINE_TYPE["STANDARD"]]
        shader.use()

        shader.cache_single_location(
            UNIFORM_TYPE["CAMERA_POS"], "cameraPosition")
        shader.cache_single_location(UNIFORM_TYPE["MODEL"], "model")
        shader.cache_single_location(UNIFORM_TYPE["VIEW"], "view")

        for i in range(8):
            shader.cache_multi_location(
                UNIFORM_TYPE["LIGHT_COLOR"], f"Lights[{i}].color")
            shader.cache_multi_location(
                UNIFORM_TYPE["LIGHT_POS"], f"Lights[{i}].position")
            shader.cache_multi_location(
                UNIFORM_TYPE["LIGHT_STRENGTH"], f"Lights[{i}].strength")
        
        shader = self.shaders[PIPELINE_TYPE["EMISSIVE"]]
        shader.use()

        shader.cache_single_location(UNIFORM_TYPE["MODEL"], "model")
        shader.cache_single_location(UNIFORM_TYPE["VIEW"], "view")
        shader.cache_single_location(UNIFORM_TYPE["TINT"], "tint")
    
    def render(self, camera, renderables, lights):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = camera.get_view_transform()
        
        shader = self.shaders[PIPELINE_TYPE["STANDARD"]]
        shader.use()

        glUniformMatrix4fv(
            shader.fetch_single_location(UNIFORM_TYPE["VIEW"]),
            1, GL_FALSE, view)

        glUniform3fv(
            shader.fetch_single_location(UNIFORM_TYPE["CAMERA_POS"]),
            1, camera.position)

        for i,light in enumerate(lights):

            glUniform3fv(
                shader.fetch_multi_location(UNIFORM_TYPE["LIGHT_POS"], i),
                1, light.position)
            glUniform3fv(
                shader.fetch_multi_location(UNIFORM_TYPE["LIGHT_COLOR"], i),
                1, light.color)
            glUniform1f(
                shader.fetch_multi_location(UNIFORM_TYPE["LIGHT_STRENGTH"], i),
                light.strength)
        
        for entity_type, entities in renderables.items():

            if entity_type not in self.materials:
                continue

            material = self.materials[entity_type]
            material.use()
            mesh = self.meshes[entity_type]
            mesh.arm_for_drawing()
            
            for entity in entities:

                glUniformMatrix4fv(
                    shader.fetch_single_location(UNIFORM_TYPE["MODEL"]),
                    1, GL_FALSE, entity.get_model_transform())

                mesh.draw()
        
        shader = self.shaders[PIPELINE_TYPE["EMISSIVE"]]
        shader.use()

        glUniformMatrix4fv(
            shader.fetch_single_location(UNIFORM_TYPE["VIEW"]),
            1, GL_FALSE, view)
        
        glFlush()

    def destroy(self):
        for mesh in self.meshes.values():
            mesh.destroy()
        for material in self.materials.values():
            material.destroy()
        for shader in self.shaders.values():
            shader.destroy()

class Shader:
    def __init__(self, vertex_filepath, fragment_filepath):
        self.program = create_shader(vertex_filepath, fragment_filepath)

        self.single_uniforms = {}
        self.multi_uniforms = {}
    
    def cache_single_location(self, uniform_type, uniform_name):

        self.single_uniforms[uniform_type] = glGetUniformLocation(
            self.program, uniform_name)
    
    def cache_multi_location(self, uniform_type, uniform_name):

        if uniform_type not in self.multi_uniforms:
            self.multi_uniforms[uniform_type] = []
        
        self.multi_uniforms[uniform_type].append(
            glGetUniformLocation(self.program, uniform_name)
        )
    
    def fetch_single_location(self, uniform_type):
        return self.single_uniforms[uniform_type]
    
    def fetch_multi_location(self, uniform_type, index):
        return self.multi_uniforms[uniform_type][index]

    def use(self) -> None:
        glUseProgram(self.program)
    
    def destroy(self) -> None:
        glDeleteProgram(self.program)

class Mesh:
    def __init__(self, filename):

        # x, y, z, s, t, nx, ny, nz
        vertices = read_obj(filename)
        self.vertex_count = len(vertices)//8
        vertices = np.array(vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        #Vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def arm_for_drawing(self):
        glBindVertexArray(self.vao)
    
    def draw(self):
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class Material:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(filepath, mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))



my_app = App()
my_app.run()
my_app.quit()