import numpy as np
import data


class Classification:
    def __init__(self, classes, gate=400):
        self.classes = classes
        self.mid = []
        self.get_mid()
        self.cov = []
        self.covariance()
        self.gate = gate

        print("kernels in order:")
        print(*self.mid, sep='\n\n')

    def get_mid(self):
        for ind, i in enumerate(self.classes):
            self.mid.append(np.sum(np.array(i), axis=0) / len(i))
            data.toimg(self.mid[-1], ind)

    def covariance(self):
        for ind, z in enumerate(self.classes):
            s = np.fromfunction(
                lambda i, j: np.sum(
                    [np.dot(z[k][i] - self.mid[ind][i], z[k][j] - self.mid[ind][j]) / (len(z) - 1) for k in
                     range(len(z))], axis=0),
                (len(z[0]), len(z[0])), dtype='uint8')
            self.cov.append(s)

    def metric(self, x, y, ind):
        inv = np.linalg.inv(self.cov[ind] + np.eye(len(self.cov[ind])))
        return np.sqrt((x - y).transpose().dot(inv).dot(x - y))

    def find(self, x):
        res = []
        for ind, m in enumerate(self.mid):
            res.append(self.metric(x, m, ind))
        return res

    def process(self, samples):
        for ind, i in enumerate(samples):
            print()
            print("sample {}: \n {}".format(ind, i))
            res = self.find(i)
            print("distance to classes: ", res)
            m = min(res)
            print("belongs to", res.index(m) if m < self.gate else -1)