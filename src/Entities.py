import numpy as np

from constants import *

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

class Lamp(Entity):
    def __init__(self, position, eulers):
        super().__init__(position, eulers)
    
    def update(self, dt, camera_pos):
        self.eulers[2] += 0.2 * dt
        
        if self.eulers[2] > 360:
            self.eulers[2] -= 360

class Light(Entity):
    def __init__(self, position, color, strength):
        super().__init__(position, eulers = [0,0,0])
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength