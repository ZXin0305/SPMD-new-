"""
dataset:cmu
'0': Neck
'1': Nose
'2': BodyCenter (center of hips)
'3': lShoulder
'4': lElbow
'5': lWrist,
'6': lHip
'7': lKnee
'8': lAnkle
'9': rShoulder
'10': rElbow
'11': rWrist
'12': rHip
'13': rKnee
'14': rAnkle
"""
import numpy as np
from IPython import embed 

class Pose():
    def __init__(self,dataset = 'cmu') -> None:
        self.dataset = dataset

    def get_level(self,dataset = None):
        """
        """
        level = {'cmu':[0,1,3,4,5,6,7,8,9,10,11,12,13,14]}
        return level[dataset]
