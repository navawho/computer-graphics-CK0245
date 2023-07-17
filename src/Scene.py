from Entities import BarCounter, Beer, Camera, Fan, Floors, Lamp, Light, Walls, Whisky

from constants import ENTITY_TYPE

class Scene:
    def __init__(self, state):
        self.state = state

        self.entities = {
            ENTITY_TYPE["BAR_COUNTER"]: [
                BarCounter(position = [6,0,0], eulers = [0,0,0])
            ],
            ENTITY_TYPE["BEER"]: [
                Beer(position = [6,0,0], eulers = [0,0,0], state=self.state),
            ],
            ENTITY_TYPE["WHISKY"]: [
                Whisky(position = [2.3,0,1.7], eulers = [0,0,0], state=self.state),
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
            ENTITY_TYPE["LAMP"]: [
                Lamp(position = [6,0,5], eulers = [0,0,0]),
            ],
            ENTITY_TYPE["LAMP_2"]: [
                Lamp(position = [6,10,5], eulers = [0,0,0]),
            ],
        }

        self.lights = [
            Light(position=[6,0,4], color = [0.9,0.9,0.9], strength=5),
            Light(position=[6,10,4], color = [0.9,0.9,0.9], strength=3)
        ]

        self.player = Camera(
            position = [0,0,2]
        )

    def update(self, dt):
        for entities in self.entities.values():
            for entity in entities:
                entity.update(dt, self.player.position, self.state)
        
        for light in self.lights:
            light.update(dt, self.player.position, self.state)

        self.player.update(dt)

    def move_player(self, d_pos):
        self.player.move(d_pos)
    
    def spin_player(self, d_eulers):
        self.player.spin(d_eulers)
