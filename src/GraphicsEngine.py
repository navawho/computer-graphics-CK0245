from OpenGL.GL import *

from Texture import Texture
from Mesh import Mesh
from Shader import Shader

from constants import *

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
            ENTITY_TYPE["LAMP"]: Mesh("models/lamp.obj"),
            ENTITY_TYPE["LAMP_2"]: Mesh("models/lamp.obj"),
        }

        self.Textures = {
            ENTITY_TYPE["BAR_COUNTER"]: Texture("textures/bar_counter.png"),
            ENTITY_TYPE["BEER"]: Texture("textures/beer.jpg"),
            ENTITY_TYPE["WHISKY"]: Texture("textures/whisky.jpg"),
            ENTITY_TYPE["FAN"]: Texture("textures/fan.jpg"),
            ENTITY_TYPE["WALLS"]: Texture("textures/wall.jpg"),
            ENTITY_TYPE["FLOORS"]: Texture("textures/floor.jpg"),
            ENTITY_TYPE["LAMP"]: Texture("textures/lamp.jpg"),
            ENTITY_TYPE["LAMP_2"]: Texture("textures/lamp.jpg"),
        }
        
        self.shaders = {
            PIPELINE_TYPE["STANDARD"]: Shader(
                "shaders/vertex.txt", "shaders/fragment.txt"),
            PIPELINE_TYPE["EMISSIVE"]: Shader(
                "shaders/vertex_light.txt", "shaders/fragment_light.txt"),
        }
    
    def _set_onetime_uniforms(self):
        projection_transform = self._perspective_projection(
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

    def _get_uniform_locations(self):
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
    
    def _perspective_projection(self, fovy, aspect, near, far):
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

            if entity_type not in self.Textures:
                continue

            Texture = self.Textures[entity_type]
            Texture.use()
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
        for Texture in self.Textures.values():
            Texture.destroy()
        for shader in self.shaders.values():
            shader.destroy()
