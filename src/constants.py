import numpy as np

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768

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
    "LAMP": 6,
    "LAMP_2": 7,
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