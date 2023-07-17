from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader

class Shader:
    def __init__(self, vertex_filepath, fragment_filepath):
        self.program = self.create_shader(vertex_filepath, fragment_filepath)

        self.single_uniforms = {}
        self.multi_uniforms = {}

    def create_shader(self, vertex_filepath, fragment_filepath):
        with open(vertex_filepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragment_filepath,'r') as f:
            fragment_src = f.readlines()
        
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
        
        return shader
    
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

    def use(self):
        glUseProgram(self.program)
    
    def destroy(self):
        glDeleteProgram(self.program)
