data format

xxxxx.json
according to different img, the name is different

every json include
{
    "img_path": the image's toor path;
    "cam_coors":keypoints in camera coordinate(cm);
    "pixel_coors":keypoints in img coordinate(pixel);
    "skel_with_conf":keypoints in world coordinate with cofidence;
    "cam":camera nei can
    "img_width":img's width
    "img_height":img's height
}

"""
CMU:original keypoints num is 19, we just use the 0-14
"""
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
'15': lEye
'16': lEar
'17': rEye
'18': rEar
"""