import numpy as np
from IPython import embed

def change_pose(points_pre, points_true):
    """
    the function is to change the format to (X,Y,Z) , (X,Y,Z) ....
    """
    predict = []
    gt = []
    if len(points_pre) > 0:
        for pose in points_pre:
            for i in range(19):
                joint = []
                X = pose[0][i]
                Y = pose[1][i]
                Z = pose[2][i]
                joint.append(i)   #joint type
                joint.append(X)
                joint.append(Y)
                joint.append(Z)
                predict.append(joint)
                
    if len(points_true) > 0:
        for pose in points_true:
            for i in range(19):
                joint = []
                X = pose[0][i]
                Y = pose[1][i]
                Z = pose[2][i]
                joint.append(i)
                joint.append(X)
                joint.append(Y)
                joint.append(Z)
                gt.append(joint)

    return predict , gt

def dist(p1, p2, th):
    """
    type: (Seq, Seq, float) -> float
    3D Point Distance
    p1:predict point
    p2:GT point
    th:the max acceptable distance
    return:euclidean distance between the positions of the two joints
    """
    if p1[0] != p2[0]:
        return np.nan
    d = np.linalg.norm(np.array(p1[1:]) - np.array(p2[1:]))
    return d if d <= th else np.nan

def non_minima_suppression(x):
    """
    return:non-minima suppressed version of the input array
    supressed values become np.nan
    """
    min = np.nanmin(x)
    x[x != min] = np.nan
    if len(x[x == min]) > 1:
        ok = True
        for i in range(len(x)):
            if x[i] == min and ok:
                ok = False
            else:
                x[i] = np.nan
    return x

def not_nan_count(x):
    """
    :return: number of not np.nan elements of the array
    返回的是一个数
    """
    return len(x[~np.isnan(x)])



def joint_det_metrics(points_pre, points_true, th=7.0):
    """
    points_pre : the predict poses in camera coordinate
    points_true: the gt-truth poses in camera coordinate
    th:distance threshold; all distances > th will be considered 'np.nan'.
    return :  a dictionary of metrics, 'met', related to joint detection;
              the the available metrics are:
              (1) met['tp'] = number of True Positives
              (2) met['fn'] = number of False Negatives
              (3) met['fp'] = number of False Positives
              (4) met['pr'] = PRecision
              (5) met['re'] = REcall
              (6) met['f1'] = F1-score
    """
    predict, gt = change_pose(points_pre=points_pre, points_true=points_true)
    if len(predict) > 0 and len(gt) > 0:
        mat = []
        for p_true in gt:
            row = np.array([dist(p_pred, p_true, th=th) for p_pred in predict])
            mat.append(row)
        mat = np.array(mat)
        mat = np.apply_along_axis(non_minima_suppression, 1, mat)
        mat = np.apply_along_axis(non_minima_suppression, 0, mat)

        # calculate joint detection metrics
        nr = np.apply_along_axis(not_nan_count, 1, mat)
        tp = len(nr[nr != 0])   #number of true positives  / 预测出来并且存在于真值中
        fn = len(nr[nr == 0])   #number of false negatives / 没有预测出来
        fp = len(predict) - tp  #预测出来但是并不对
        pr = tp / (tp+fp)
        re = tp / (tp+fn)
        f1 = (2 * tp) / (2 * tp + fn + fp)

    elif len(predict) == 0 and len(gt) == 0:
        tp = 0    #number of true positives
        fn = 0    #number of false negatives
        fp = 0    #number of false positive
        pr = 1.0
        re = 1.0
        f1 = 1.0
    elif len(predict) == 0:
        tp = 0
        fn = len(gt)
        fp = 0
        pr = 0.0
        re = 0.0
        f1 = 0.0
    else:
        tp = 0
        fn = 0
        fp = len(predict)
        pr = 0.0
        re = 0.0
        f1 = 0.0

    metrics = {
        'tp':tp, 'fn':fn, 'fp':fp,
        'pr':pr, 're':re, 'f1':f1,
    }

    return metrics