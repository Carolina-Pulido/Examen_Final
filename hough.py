import numpy as np
"""
Hough --> Identificar las rectas
y = kx + m
Nos vamos al plano de la pendiente(plano m-k): m = -xk + y
r es la distancia más corta del origen de la imagen a la resta
"""



class hough():
    def __init__(self, bw_edges):
        [self.rows, self.cols] = bw_edges.shape[:2]
        self.center_x = self.cols // 2
        self.center_y = self.rows // 2
        self.theta = np.arange(0, 360, 0.5)
        self.bw_edges = bw_edges

    def standard_HT(self):

        rmax = int(round(0.5 * np.sqrt(self.rows ** 2 + self.cols ** 2)))
        y, x = np.where(self.bw_edges >= 1)     #Recuperar valores en x y y de la imagen de bordes

        accumulator = np.zeros((rmax, len(self.theta))) #Acumulador en ceros

        for idx, th in enumerate(self.theta):   #Para cada valor de tetha se calcula r
            r = np.around(
                (x - self.center_x) * np.cos((th * np.pi) / 180) + (y - self.center_y) * np.sin((th * np.pi) / 180))   #Se coloa el plano centrado en toda la mitad de la imagen
            r = r.astype(int)
            r_idx = np.where(np.logical_and(r >= 0, r < rmax))  #Pequeño filtrado, que r sea mayor a 0 y menos a rmax
            np.add.at(accumulator[:, idx], r[r_idx[0]], 1)
        return accumulator

    def direct_HT(self, theta_data):

        rmax = int(round(0.5 * np.sqrt(self.rows ** 2 + self.cols ** 2)))
        # y , x = np.where(M >= 0.1)
        y, x = np.where(self.bw_edges >= 1)

        x_ = x - self.center_x
        y_ = y - self.center_y

        th = theta_data[y, x] + np.pi / 2

        hist_val, bin_edges = np.histogram(th, bins=32)
        print('Histogram', hist_val)

        print(np.amin(th), np.amax(th))
        th[y_ < 0] = th[y_ < 0] + np.pi
        print(np.amin(th), np.amax(th))
        accumulator = np.zeros((rmax, len(self.theta)))

        r = np.around(x_ * np.cos(th) + y_ * np.sin(th))
        r = r.astype(int)
        th = np.around(360 * th / np.pi)
        th = th.astype(int)
        th[th == 720] = 0
        print(np.amin(th), np.amax(th))
        r_idx = np.where(np.logical_and(r >= 0, r < rmax))
        np.add.at(accumulator, (r[r_idx[0]], th[r_idx[0]]), 1)
        return accumulator

    def find_peaks(self, accumulator, nhood, accumulator_threshold, N_peaks):
        done = False
        acc_copy = accumulator
        nhood_center = [(nhood[0] - 1) / 2, (nhood[1] - 1) / 2]
        peaks = []
        while not done:
            [p, q] = np.unravel_index(acc_copy.argmax(), acc_copy.shape)
            if acc_copy[p, q] >= accumulator_threshold:
                peaks.append([p, q])
                p1 = p - nhood_center[0]
                p2 = p + nhood_center[0]
                q1 = q - nhood_center[1]
                q2 = q + nhood_center[1]

                [qq, pp] = np.meshgrid(np.arange(np.max([q1, 0]), np.min([q2, acc_copy.shape[1] - 1]) + 1, 1), \
                                       np.arange(np.max([p1, 0]), np.min([p2, acc_copy.shape[0] - 1]) + 1, 1))
                pp = np.array(pp.flatten(), dtype=np.intp)
                qq = np.array(qq.flatten(), dtype=np.intp)

                acc_copy[pp, qq] = 0
                done = np.array(peaks).shape[0] == N_peaks
            else:
                done = True

        return peaks
