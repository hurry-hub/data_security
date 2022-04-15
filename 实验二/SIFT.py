# coding: utf-8
import warnings

warnings.filterwarnings("ignore")  # 忽略警告
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image


def convolve(filter, mat, padding, strides):
    """
    卷积，使图片更平滑
    :param filter: 滤波器
    :param mat: 图层
    :param padding: 填充
    :param strides: 步长
    :return: 平滑后的矩阵
    """
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:   # 卷积核为二维
        if len(mat_size) == 3:  # 图像为三维
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0, mat_size[0], strides[1]):
                    temp.append([])
                    for k in range(0, mat_size[1], strides[0]):
                        val = (filter * pad_mat[j:j + filter_size[0], k:k + filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:    # 图像为两维
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j:j + filter_size[0], k:k + filter_size[1]]).sum()
                    channel[-1].append(val)

            result = np.array(channel)

    return result


def downsample(img, step=2):
    """
    压缩图片
    :param img:     原始图像
    :param step:    图层压缩倍数
    :return:    压缩后的图像
    """
    return img[::step, ::step]


def GuassianKernel(sigma, dim):
    """
    计算高斯核
    :param sigma: Standard deviation            尺度空间因子
    :param dim: dimension(must be positive and also an odd number)  维数
    :return: return the required Gaussian kernel.   高斯核result
    """
    temp = [t - (dim // 2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant) # 公式中的x，y是其转置
    temp = 2 * sigma * sigma
    # 高斯和函数计算公式
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + (assistant.T) ** 2) / temp)
    return result


def getDoG(img, n, sigma0, S=None, O=None):
    """
    获得DoG金字塔
    :param img: the original img.       原始图像
    :param sigma0: sigma of the first stack of the first octave. default 1.52 for complicate reasons.   第一组第一层的初始σ
    :param n: how many stacks of feature that you wanna extract.                提取的特征层数
    :param S: how many stacks does every octave have. S must bigger than 3.     每组金字塔对应的层数
    :param O: how many octaves do we have.          金字塔组数
    :return: the DoG Pyramid                    DoG金字塔
    """
    if S is None:           # 每组的层数
        S = n + 3
    if O is None:           # 计算得到金字塔组数
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3

    k = 2 ** (1.0 / n)  # 相邻两层之间的尺度相差的比例因子
    # o为所在的组，s为所在的层，σ0为初始的尺度，计算得出每层的高斯模糊因子
    sigma = [[(k ** s) * sigma0 * (1 << o) for s in range(S)] for o in range(O)]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]

    GuassianPyramid = []
    for i in range(O):              # 每组
        GuassianPyramid.append([])
        for j in range(S):          # 每层
            dim = int(6 * sigma[i][j] + 1)  # 维度
            if dim % 2 == 0:
                dim += 1
            GuassianPyramid[-1].append(
                convolve(GuassianKernel(sigma[i][j], dim), samplePyramid[i], [dim // 2, dim // 2, dim // 2, dim // 2],
                         [1, 1]))       # 计算高斯核
    DoG = [[GuassianPyramid[o][s + 1] - GuassianPyramid[o][s] for s in range(S - 1)] for o in range(O)]     # 高斯金字塔相减得到DoG金字塔

    return DoG, GuassianPyramid

# 删除不稳定的边缘特征点
def adjustLocalExtrema(DoG, o, s, x, y, contrastThreshold, edgeThreshold, sigma, n, SIFT_FIXPT_SCALE):
    SIFT_MAX_INTERP_STEPS = 5
    SIFT_IMG_BORDER = 5

    point = []

    img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE)
    deriv_scale = img_scale * 0.5
    second_deriv_scale = img_scale
    cross_deriv_scale = img_scale * 0.25

    img = DoG[o][s]
    i = 0
    while i < SIFT_MAX_INTERP_STEPS:
        if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
                img.shape[0] - SIFT_IMG_BORDER:
            return None, None, None, None

        img = DoG[o][s]
        prev = DoG[o][s - 1]
        next = DoG[o][s + 1]

        dD = [(img[x, y + 1] - img[x, y - 1]) * deriv_scale,
              (img[x + 1, y] - img[x - 1, y]) * deriv_scale,
              (next[x, y] - prev[x, y]) * deriv_scale]

        v2 = img[x, y] * 2
        dxx = (img[x, y + 1] + img[x, y - 1] - v2) * second_deriv_scale
        dyy = (img[x + 1, y] + img[x - 1, y] - v2) * second_deriv_scale
        dss = (next[x, y] + prev[x, y] - v2) * second_deriv_scale
        dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * cross_deriv_scale
        dxs = (next[x, y + 1] - next[x, y - 1] - prev[x, y + 1] + prev[x, y - 1]) * cross_deriv_scale
        dys = (next[x + 1, y] - next[x - 1, y] - prev[x + 1, y] + prev[x - 1, y]) * cross_deriv_scale

        H = [[dxx, dxy, dxs],
             [dxy, dyy, dys],
             [dxs, dys, dss]]

        X = np.matmul(np.linalg.pinv(np.array(H)), np.array(dD))

        xi = -X[2]
        xr = -X[1]
        xc = -X[0]

        if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
            break

        y += int(np.round(xc))
        x += int(np.round(xr))
        s += int(np.round(xi))

        i += 1

    if i >= SIFT_MAX_INTERP_STEPS:
        return None, x, y, s
    if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
            img.shape[0] - SIFT_IMG_BORDER:
        return None, None, None, None

    t = (np.array(dD)).dot(np.array([xc, xr, xi]))

    contr = img[x, y] * img_scale + t * 0.5
    if np.abs(contr) * n < contrastThreshold:
        return None, x, y, s

    # 利用Hessian矩阵的迹和行列式计算主曲率的比值
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    if det <= 0 or tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det:
        return None, x, y, s

    point.append((x + xr) * (1 << o))
    point.append((y + xc) * (1 << o))
    point.append(o + (s << 8) + (int(np.round((xi + 0.5)) * 255) << 16))
    point.append(sigma * np.power(2.0, (s + xi) / n) * (1 << o) * 2)

    return point, x, y, s

# 求取特征点的主方向
def GetMainDirection(img, r, c, radius, sigma, BinNum):
    expf_scale = -1.0 / (2.0 * sigma * sigma)

    X = []
    Y = []
    W = []
    temphist = []

    for i in range(BinNum):
        temphist.append(0.0)

    # 图像梯度直方图统计的像素范围
    k = 0
    for i in range(-radius, radius + 1):
        y = r + i
        if y <= 0 or y >= img.shape[0] - 1:
            continue
        for j in range(-radius, radius + 1):
            x = c + j
            if x <= 0 or x >= img.shape[1] - 1:
                continue

            dx = (img[y, x + 1] - img[y, x - 1])
            dy = (img[y - 1, x] - img[y + 1, x])

            X.append(dx)
            Y.append(dy)
            W.append((i * i + j * j) * expf_scale)
            k += 1

    length = k

    W = np.exp(np.array(W))
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi
    Mag = (X ** 2 + Y ** 2) ** 0.5

    # 计算直方图的每个bin
    for k in range(length):
        bin = int(np.round((BinNum / 360.0) * Ori[k]))
        if bin >= BinNum:
            bin -= BinNum
        if bin < 0:
            bin += BinNum
        temphist[bin] += W[k] * Mag[k]

    # smooth the histogram
    # 高斯平滑
    temp = [temphist[BinNum - 1], temphist[BinNum - 2], temphist[0], temphist[1]]
    temphist.insert(0, temp[0])
    temphist.insert(0, temp[1])
    temphist.insert(len(temphist), temp[2])
    temphist.insert(len(temphist), temp[3])  # padding

    hist = []
    for i in range(BinNum):
        hist.append(
            (temphist[i] + temphist[i + 4]) * (1.0 / 16.0) + (temphist[i + 1] + temphist[i + 3]) * (4.0 / 16.0) +
            temphist[i + 2] * (6.0 / 16.0))

    # 得到主方向
    maxval = max(hist)

    return maxval, hist


def LocateKeyPoint(DoG, sigma, GuassianPyramid, n, BinNum=36, contrastThreshold=0.04, edgeThreshold=10.0):
    """
    定位特征点
    :param DoG: 获得的DoG图像
    :param sigma:   高斯核函数中的尺度空间因子
    :param GuassianPyramid: 高斯金字塔
    :param n: 高斯金字塔特提取特征层数
    :param BinNum:  直方图柱的个数
    :param contrastThreshold: 过滤掉较差的特征点对的阈值
    :param edgeThreshold: 过滤掉边缘效应的阈值
    :return: 特征点KeyPoints
    """
    SIFT_ORI_SIG_FCTR = 1.52
    SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR
    SIFT_ORI_PEAK_RATIO = 0.8

    SIFT_INT_DESCR_FCTR = 512.0
    SIFT_FIXPT_SCALE = 1

    KeyPoints = []
    O = len(DoG)
    S = len(DoG[0])
    for o in range(O):
        for s in range(1, S - 1):
            threshold = 0.5 * contrastThreshold / (n * 255 * SIFT_FIXPT_SCALE)
            img_prev = DoG[o][s - 1]
            img = DoG[o][s]
            img_next = DoG[o][s + 1]
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    val = img[i, j]
                    eight_neiborhood_prev = img_prev[max(0, i - 1):min(i + 2, img_prev.shape[0]),
                                            max(0, j - 1):min(j + 2, img_prev.shape[1])]
                    eight_neiborhood = img[max(0, i - 1):min(i + 2, img.shape[0]),
                                       max(0, j - 1):min(j + 2, img.shape[1])]
                    eight_neiborhood_next = img_next[max(0, i - 1):min(i + 2, img_next.shape[0]),
                                            max(0, j - 1):min(j + 2, img_next.shape[1])]
                    # 筛掉低对比度的特征点
                    if np.abs(val) > threshold and \
                            ((val > 0 and (val >= eight_neiborhood_prev).all() and (val >= eight_neiborhood).all() and (
                                    val >= eight_neiborhood_next).all())
                             or (val < 0 and (val <= eight_neiborhood_prev).all() and (
                                            val <= eight_neiborhood).all() and (val <= eight_neiborhood_next).all())):

                        # 筛掉不稳定的边缘响应点
                        point, x, y, layer = adjustLocalExtrema(DoG, o, s, i, j, contrastThreshold, edgeThreshold,
                                                                sigma, n, SIFT_FIXPT_SCALE)
                        if point == None:
                            continue

                        scl_octv = point[-1] * 0.5 / (1 << o)
                        # 求取特征点的主方向
                        omax, hist = GetMainDirection(GuassianPyramid[o][layer], x, y,
                                                      int(np.round(SIFT_ORI_RADIUS * scl_octv)),
                                                      SIFT_ORI_SIG_FCTR * scl_octv, BinNum)
                        mag_thr = omax * SIFT_ORI_PEAK_RATIO
                        for k in range(BinNum):
                            if k > 0:
                                l = k - 1
                            else:
                                l = BinNum - 1
                            if k < BinNum - 1:
                                r2 = k + 1
                            else:
                                r2 = 0
                            if hist[k] > hist[l] and hist[k] > hist[r2] and hist[k] >= mag_thr:
                                bin = k + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[k] + hist[r2])
                                if bin < 0:
                                    bin = BinNum + bin
                                else:
                                    if bin >= BinNum:
                                        bin = bin - BinNum
                                temp = point[:]
                                temp.append((360.0 / BinNum) * bin)
                                KeyPoints.append(temp)

    return KeyPoints

# 计算SIFT特征描述符
def calcSIFTDescriptor(img, ptf, ori, scl, d, n, SIFT_DESCR_SCL_FCTR=3.0, SIFT_DESCR_MAG_THR=0.2,
                       SIFT_INT_DESCR_FCTR=512.0, FLT_EPSILON=1.19209290E-07):
    dst = []
    pt = [int(np.round(ptf[0])), int(np.round(ptf[1]))]  # 坐标点取整
    cos_t = np.cos(ori * (np.pi / 180))  # 余弦值
    sin_t = np.sin(ori * (np.pi / 180))  # 正弦值
    bins_per_rad = n / 360.0
    exp_scale = -1.0 / (d * d * 0.5)
    hist_width = SIFT_DESCR_SCL_FCTR * scl
    radius = int(np.round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))
    cos_t /= hist_width
    sin_t /= hist_width

    rows = img.shape[0]
    cols = img.shape[1]

    hist = [0.0] * ((d + 2) * (d + 2) * (n + 2))
    X = []
    Y = []
    RBin = []
    CBin = []
    W = []

    k = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):

            c_rot = j * cos_t - i * sin_t
            r_rot = j * sin_t + i * cos_t
            rbin = r_rot + d // 2 - 0.5
            cbin = c_rot + d // 2 - 0.5
            r = pt[1] + i
            c = pt[0] + j

            if -1 < rbin < d and -1 < cbin < d and 0 < r < rows - 1 and 0 < c < cols - 1:
                dx = (img[r, c + 1] - img[r, c - 1])
                dy = (img[r - 1, c] - img[r + 1, c])
                X.append(dx)
                Y.append(dy)
                RBin.append(rbin)
                CBin.append(cbin)
                W.append((c_rot * c_rot + r_rot * r_rot) * exp_scale)
                k += 1

    length = k
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi
    Mag = (X ** 2 + Y ** 2) ** 0.5
    W = np.exp(np.array(W))

    for k in range(length):
        rbin = RBin[k]
        cbin = CBin[k]
        obin = (Ori[k] - ori) * bins_per_rad
        mag = Mag[k] * W[k]

        r0 = int(rbin)
        c0 = int(cbin)
        o0 = int(obin)
        rbin -= r0
        cbin -= c0
        obin -= o0

        if o0 < 0:
            o0 += n
        if o0 >= n:
            o0 -= n

        # 使用三线性插值法更新直方图
        v_r1 = mag * rbin
        v_r0 = mag - v_r1

        v_rc11 = v_r1 * cbin
        v_rc10 = v_r1 - v_rc11

        v_rc01 = v_r0 * cbin
        v_rc00 = v_r0 - v_rc01

        v_rco111 = v_rc11 * obin
        v_rco110 = v_rc11 - v_rco111

        v_rco101 = v_rc10 * obin
        v_rco100 = v_rc10 - v_rco101

        v_rco011 = v_rc01 * obin
        v_rco010 = v_rc01 - v_rco011

        v_rco001 = v_rc00 * obin
        v_rco000 = v_rc00 - v_rco001

        idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0
        hist[idx] += v_rco000
        hist[idx + 1] += v_rco001
        hist[idx + (n + 2)] += v_rco010
        hist[idx + (n + 3)] += v_rco011
        hist[idx + (d + 2) * (n + 2)] += v_rco100
        hist[idx + (d + 2) * (n + 2) + 1] += v_rco101
        hist[idx + (d + 3) * (n + 2)] += v_rco110
        hist[idx + (d + 3) * (n + 2) + 1] += v_rco111

    # 根据方向直方图是环形，补完直方图
    for i in range(d):
        for j in range(d):
            idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
            hist[idx] += hist[idx + n]
            hist[idx + 1] += hist[idx + n + 1]
            for k in range(n):
                dst.append(hist[idx + k])

    # 将直方图复制到描述符中
    # 使用滞后阈值
    # 规范结果使其便于处理
    # 转为字节数组
    nrm2 = 0
    length = d * d * n
    for k in range(length):
        nrm2 += dst[k] * dst[k]
    thr = np.sqrt(nrm2) * SIFT_DESCR_MAG_THR

    nrm2 = 0
    for i in range(length):
        val = min(dst[i], thr)
        dst[i] = val
        nrm2 += val * val
    nrm2 = SIFT_INT_DESCR_FCTR / max(np.sqrt(nrm2), FLT_EPSILON)
    for k in range(length):
        dst[k] = min(max(dst[k] * nrm2, 0), 255)

    return dst


def calcDescriptors(gpyr, keypoints, SIFT_DESCR_WIDTH=4, SIFT_DESCR_HIST_BINS=8):
    """
    生成特征描述
    :param gpyr: 高斯金字塔
    :param keypoints: 关键点
    :param SIFT_DESCR_WIDTH: 描述直方图的宽度
    :param SIFT_DESCR_HIST_BINS: 梯度直方图的方向
    :return:
    """
    d = SIFT_DESCR_WIDTH
    n = SIFT_DESCR_HIST_BINS
    descriptors = []

    for i in range(len(keypoints)):
        kpt = keypoints[i]
        o = kpt[2] & 255
        s = (kpt[2] >> 8) & 255  # 该特征点所在的组序号和层序号
        scale = 1.0 / (1 << o)  # 缩放倍数
        size = kpt[3] * scale  # 该特征点所在组的图像尺寸
        ptf = [kpt[1] * scale, kpt[0] * scale]  # 该特征点在金字塔组中的坐标
        img = gpyr[o][s]  # 该点所在的金字塔图像
        # 计算SIFT特征描述符
        descriptors.append(calcSIFTDescriptor(img, ptf, kpt[-1], size * 0.5, d, n))
    return descriptors


def SIFT(img, showDoGimgs=False):
    """
    获得图像的SIFT特征
    :param img: 待计算图片
    :param showDoGimgs: 是否计算Dog图像
    :return: 关键点keypoints和描述符discriptors
    """
    SIFT_SIGMA = 1.6  # 高斯核函数中的σ
    SIFT_INIT_SIGMA = 0.5  # 假设的摄像头的尺度
    sigma0 = np.sqrt(SIFT_SIGMA ** 2 - SIFT_INIT_SIGMA ** 2)  # 第一层的高斯模糊参数

    n = 3  # 高斯金字塔特提取特征层数

    DoG, GuassianPyramid = getDoG(img, n, sigma0)
    if showDoGimgs:     # 选择了计算DoG图
        for i in DoG:
            for j in i:
                plt.imshow(j.astype(np.uint8), cmap='gray')
                plt.axis('off')
                plt.show()

    KeyPoints = LocateKeyPoint(DoG, SIFT_SIGMA, GuassianPyramid, n)     # 定位特征点
    discriptors = calcDescriptors(GuassianPyramid, KeyPoints)           # 计算描述符

    return KeyPoints, discriptors

# 在合成图上画线
def Lines(img, info, color=(255, 0, 0), err=700):
    if len(img.shape) == 2:
        result = np.dstack((img, img, img))
    else:
        result = img
    k = 0
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            temp = (info[:, 1] - info[:, 0])
            A = (j - info[:, 0]) * (info[:, 3] - info[:, 2])
            B = (i - info[:, 2]) * (info[:, 1] - info[:, 0])
            temp[temp == 0] = 1e-9
            t = (j - info[:, 0]) / temp
            e = np.abs(A - B)
            temp = e < err
            if (temp * (t >= 0) * (t <= 1)).any():
                result[i, j] = color
                k += 1
    print(k)

    return result


def drawLines(X1, X2, Y1, Y2, dis, img, num=10):
    """
    画出两图片的对应关键点
    :param X1: 图像1关键点的横坐标
    :param X2: 图像2关键点的横坐标
    :param Y1: 图像1关键点的纵坐标
    :param Y2: 图像2关键点的纵坐标
    :param dis: knn分类器得到的结果
    :param img: 组合图片
    :param num: 画出的关键点数量
    :return: None
    """
    info = list(np.dstack((X1, X2, Y1, Y2, dis))[0])
    info = sorted(info, key=lambda x: x[-1])
    info = np.array(info)
    info = info[:min(num, info.shape[0]), :]
    img = Lines(img, info)      # 画线

    if len(img.shape) == 2:
        plt.imshow(img.astype(np.uint8), cmap='gray')
    else:
        plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    origimg = plt.imread('./SIFTimg/3.jpeg')  # 读取图像信息
    if len(origimg.shape) == 3:  # RGB图像
        img = origimg.mean(axis=-1)  # 按最后一维合并数组
    else:  # 灰度图
        img = origimg
    keyPoints, discriptors = SIFT(img)  # 获得SIFT特征

    origimg2 = plt.imread('./SIFTimg/03.jpeg')  # 读取图像信息
    if len(origimg.shape) == 3:  # RGB图像
        img2 = origimg2.mean(axis=-1)  # 按最后一维合并数组
    else:  # 灰度图
        img2 = origimg2
    ScaleRatio = img.shape[0] * 1.0 / img2.shape[0]

    img2 = np.array(Image.fromarray(img2).resize((int(round(ScaleRatio * img2.shape[1])), img.shape[0]), Image.BICUBIC))
    keyPoints2, discriptors2 = SIFT(img2, True)

    knn = KNeighborsClassifier(n_neighbors=1)   # K近邻分类器
    knn.fit(discriptors, [0] * len(discriptors))
    match = knn.kneighbors(discriptors2, n_neighbors=1, return_distance=True)

    # 处理特征点
    keyPoints = np.array(keyPoints)[:, :2]
    keyPoints2 = np.array(keyPoints2)[:, :2]

    keyPoints2[:, 1] = img.shape[1] + keyPoints2[:, 1]
    origimg2 = np.array(Image.fromarray(origimg2).resize((img2.shape[1], img2.shape[0]), Image.BICUBIC))
    result = np.hstack((origimg, origimg2))     # 将两图片水平贴合

    keyPoints = keyPoints[match[1][:, 0]]
    X1 = keyPoints[:, 1]
    X2 = keyPoints2[:, 1]
    Y1 = keyPoints[:, 0]
    Y2 = keyPoints2[:, 0]

    drawLines(X1, X2, Y1, Y2, match[0][:, 0], result)   # 关键点匹配连线
