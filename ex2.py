import numpy as np
import gradDescent as grad

LOW = 0
HIGH = 1

w_star = np.array([0.6, -1])


def sample(m, epsilon):
    s = np.zeros((m, 3))
    x = np.random.uniform(LOW, HIGH, size=(m, 2))
    s[:, :2] = x
    w_star_times_x = x.dot(w_star.T)
    temp = np.sign(w_star_times_x)
    y_label = temp + np.random.normal(loc=0.0, scale=epsilon, size=m)

    s[:, 2] = y_label

    return s


def calcLS(s):
    w_low = -3
    w_up = 3
    mesh_range = np.arange(w_low, w_up, (w_up - w_low) / 200)
    ws = np.meshgrid(mesh_range, mesh_range)

    ws_0 = ws[0].flatten()
    ws_1 = ws[1].flatten()

    ws_flat = np.vstack((ws_0, ws_1)).T

    calc_loss = 0.5 * np.square(np.dot(ws_flat, s[:, :2].T) - s[:, 2].T)

    l_s = (np.sum(calc_loss, 1) / s.shape[0]).reshape(ws[0].shape)

    return l_s


def calcGradient(s, w):
    w = np.array(w)
    calc_loss = np.dot(w, s[:, :2].T) - s[:, 2].T
    tmp = np.multiply(calc_loss.reshape((s.shape[0], 1)), s[:, :2])
    grad_l_w = np.sum(tmp, 0) / s.shape[0]
    return grad_l_w


def main():
    s_5_1 = sample(5, 1)

    s_1000_0001 = sample(1000, 0.001)
    l_s_5_1 = calcLS(s_5_1)
    l_s_1000_0001 = calcLS(s_1000_0001)

    grad.gradDescent(s_5_1, l_s_5_1)
    grad.gradDescent(s_1000_0001, l_s_1000_0001)


if __name__ == '__main__':
    main()
