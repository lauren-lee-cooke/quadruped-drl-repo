import pybullet as p
import numpy as np
import random
from opensimplex import noise2, random_seed, seed
import numpy as np

class unevenPLANE:
    def __init__(self, client, friction_coeff: int = 0 , terrain_size = [20, 80], height_range = 0.04):
        # terrain size should be specified in [row, column] format 
        
        grid_height = terrain_size[0]
        grid_width = terrain_size[1]
        amplitude = height_range
        
        num_octaves = 2
        lacunarity = 2.0
        gain = 0.25 # aka persistence 
        frequency = 10
        
        height_data, foot_heights = self.createFractalTerrain(grid_height, grid_width, num_octaves, lacunarity, gain, frequency, amplitude)
        height_data_reshaped = height_data.reshape(-1)
        self.foot_heights = foot_heights

        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, heightfieldData=height_data_reshaped, meshScale=[0.1, 0.1, 1],
                                                numHeightfieldRows=grid_width, numHeightfieldColumns=grid_height, physicsClientId = client)

        self.terrain = p.createMultiBody(0, terrainShape, physicsClientId = client)
        
        p.changeVisualShape(self.terrain, -1, rgbaColor=[0.738, 0.762, 0.777, 1], physicsClientId = client)
        p.resetBasePositionAndOrientation(self.terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId = client)
        p.changeDynamics(self.terrain, -1, lateralFriction = friction_coeff, physicsClientId = client)
    
    def changeFriction(self, friction_coeff, client):
        p.resetBasePositionAndOrientation(self.terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId = client)
        p.changeDynamics(self.terrain, -1, lateralFriction = friction_coeff)
    
    def createFractalTerrain(self, grid_height, grid_width, num_octaves, lacunarity, gain, frequency, amplitude):
        
        grid = np.zeros(shape=[grid_height, grid_width], dtype=np.float64)
    
        random_seed()
        
        for i in range(grid_height):
            for j in range(grid_width):
                
                elevation = 0
                t_frequency = frequency
                t_amplitude = amplitude
                
                for k in range(num_octaves):
                    
                    sample_x = j*t_frequency
                    sample_y = i*t_frequency
                    elevation = elevation + noise2(sample_x, sample_y)*t_amplitude
                    t_frequency *= lacunarity
                    t_amplitude *= gain
                
                grid[i, j] = elevation
        mid_height = grid_height//2
        mid_width = grid_width//2
        
        foot_heights = [(grid[(mid_height + 2), (mid_width-1)] + grid[(mid_height + 2), (mid_width)])/2, 
                        (grid[(mid_height + 2), (mid_width+1)]+ grid[(mid_height + 2), (mid_width)])/2,
                        (grid[(mid_height - 2), (mid_width-1)] + grid[(mid_height - 2), (mid_width)])/2, 
                        (grid[(mid_height - 2), (mid_width+1)]+ grid[(mid_height - 2), (mid_width)])/2]
        
        return grid, foot_heights
    
    def removeTerrain(self, client):
        p.removeBody(self.terrain, physicsClientId = client)
        
    def getFootHeights(self): 
        for i in range(len(self.foot_heights)):
            self.foot_heights[i] = -(0.28 - (self.foot_heights[i] + 0.03)) # adjust for foot height
                
        return self.foot_heights