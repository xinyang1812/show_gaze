from numpy.linalg import eig, inv
import numpy as np


def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    x = np.array(x)
    y = np.array(y)
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))


def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 +
                                                 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 +
                                                 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation2(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi / 2
    else:
        if a > c:
            return np.arctan(2 * b / (a - c)) / 2
        else:
            return np.pi / 2 + np.arctan(2 * b / (a - c)) / 2


def Ellipse(x,y):
    # TODO: check accuracy
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation2(a)
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)

    arc = 1.8
    R = np.arange(0, arc * np.pi, 0.01)
    a, b = axes
    xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)

    (ell_x0, ell_y0), (ell_w, ell_h) = (
        center[0], center[1]), (axes[0], axes[1])

    # get coeff -- A*x^2+B*x*y+C*y^2+D*x+E*y+F=0
    # Initialise ellipse conic equation constants
    ell_A = ell_w * ell_w * np.sin(phi) * np.sin(phi) + \
        ell_h * ell_h * np.cos(phi) * np.cos(phi)
    ell_B = 2 * (ell_h * ell_h - ell_w * ell_w) * np.sin(phi) * np.cos(phi)
    ell_C = ell_w * ell_w * np.cos(phi) * np.cos(phi) + \
        ell_h * ell_h * np.sin(phi) * np.sin(phi)
    ell_D = -2 * ell_A * ell_x0 - ell_B * ell_y0
    ell_E = -ell_B * ell_x0 - 2 * ell_C * ell_y0
    ell_F = ell_A * ell_x0 * ell_x0 + ell_B * ell_x0 * ell_y0 + \
        ell_C * ell_y0 * ell_y0 - ell_w * ell_w * ell_h * ell_h

    zz = np.copy(yy)
    for i in range(xx.shape[0]):
        zz[i] = (-(ell_E + ell_B * xx[i]) + \
                 np.sqrt((ell_E + ell_B * xx[i]) * (ell_E + ell_B * \
                 xx[i]) - 4 * ell_C * (ell_A * xx[i] * xx[i] + ell_D * xx[i] + ell_F))) / (2. * ell_C)
    # (aa,bb) = (xx[i],yy[i])
    return ell_A, ell_B, ell_C, ell_D, ell_E, ell_F, [xx, yy, zz]
