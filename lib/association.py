import numpy as np
from IPython import embed
import math

import numpy as np
from IPython import embed
import math

class Association():
    def __init__(self, cnf):
        self.Z = math.sqrt(cnf.outh ** 2 + cnf.outw ** 2)
        self.factor_x = cnf.oriw / cnf.outw  # 1920 / 228
        self.factor_y = cnf.orih / cnf.outh  # 1080 / 128
        self.level = [[0, 1],
                      [3, 4, 5],
                      [6, 7, 8],
                      [9, 10, 11],
                      [12, 13, 14]]
        self.should_change_joint_order = True
        self.change_order = [1,2,0,3,4,5,6,7,8,9,10,11,12,13,14]


    def _assocition(self, centers, offset_maps, depth_maps):
        """
        :param centers: (y, x)
        :param offset_maps:(h, w, c)
        :param depth_maps:(h, w, c)
        :return:
        """
        poses = []

        # 遍历所有的人
        for center in centers:
            pose = []
            center_x, center_y = center[1], center[0]
            root_depth = depth_maps[center_y, center_x, 0] * self.Z
            # start_depth = root_depth
            root_joint = [center_x * self.factor_x, center_y * self.factor_y, root_depth]
            pose.append(root_joint)
            for single_path in self.level:  # 遍历每一条通道
                start_joint = root_joint    # 每条支路都是从center joint 开始的
                for i, index in enumerate(single_path):
                    offset_x = offset_maps[center_y, center_x, 2*index + 1] * self.Z * self.factor_x
                    offset_y = offset_maps[center_y, center_x, 2*index] * self.Z * self.factor_y

                    if index == 0 or index == 1:
                        index += 1
                    relative_depth = depth_maps[center_y, center_x, index]

                    end_joint = [int(start_joint[0] + offset_x + 0.5), int(start_joint[1] + offset_y + 0.5), start_joint[2] + relative_depth]
                    pose.append(end_joint)
                    start_joint = end_joint
            poses.append(pose)
        poses = np.array(poses)

        if self.should_change_joint_order:
            poses[:, :, :] = poses[:, self.change_order, :]

        return poses