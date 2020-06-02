import numpy as np


class Differentiator:
    """Time domain differentiation using the alpha transform"""

    def __init__(self, fs, alpha=1.0):
        self.T = 1.0 / fs
        self.alpha = alpha
        self.x_1 = 0.0
        self.xD_1 = 0.0

    def differentiate(self, x):
        xD = (((1 + self.alpha) / self.T) * (x - self.x_1)) - \
            self.alpha * self.xD_1
        self.x_1 = x
        self.xD_1 = xD
        return xD


class Hysteresis:
    """Class to implement hysteresis processing"""

    def __init__(self, drive, sat, width, fs, dAlpha=1.0, mode='RK2'):
        # def __init__(self, M_s, a, alpha, k, c, fs, dAlpha=1.0, mode='RK2'):
        """
        Parameters
        ----------
        drive : float
            Hysteresis drive parameter
        sat : float
            Saturation parameter
        width : float
            Hysteresis width parameter
        fs : float
            Sample rate
        dAlpha : float
            Alpha value used for the alpha transform
        """
        self.deriv = Differentiator(fs, dAlpha)
        self.T = 1.0 / fs

        self.M_s = 0.5 + 1.5*(1-sat)  # saturation
        self.a = self.M_s / (0.01 + 6*drive)  # adjustable parameter
        self.alpha = 1.6e-3
        self.k = 30 * (1-0.5)**6 + 0.01  # Coercivity
        self.c = (1-width)**0.5 - 0.01  # Changes slope

        assert mode == 'RK2' or mode == 'RK4', "Invalid mode!"
        self.mode = mode

    def langevin(self, x):
        """Langevin function: coth(x) - (1/x)"""
        if (abs(x) > 10 ** -4):
            return (1 / np.tanh(x)) - (1/x)
        else:
            return (x / 3)

    def langevin_deriv(self, x):
        """Derivative of the Langevin function: (1/x^2) - coth(x)^2 + 1"""
        if (abs(x) > 10 ** -4):
            return (1 / x ** 2) - (1 / np.tanh(x)) ** 2 + 1
        else:
            return (1 / 3)

    def dMdt(self, M, H, H_d):
        """Jiles-Atherton differential equation

        Parameters
        ----------
        M : float
            Magnetisation
        H : float
            Magnetic field
        H_d : float
            Time derivative of magnetic field

        Returns
        -------
        dMdt : float
            Derivative of magnetisation w.r.t time
        """
        Q = (H + self.alpha * M) / self.a
        M_diff = self.M_s * self.langevin(Q) - M
        delta = 1 if H_d > 0 else -1
        delta_M = 1 if np.sign(delta) == np.sign(M_diff) else 0
        L_prime = self.langevin_deriv(Q)

        denominator = 1 - self.c * self.alpha * (self.M_s / self.a) * L_prime

        t1_num = (1 - self.c) * delta_M * M_diff
        t1_den = (1 - self.c) * delta * self.k - self.alpha * M_diff
        t1 = (t1_num / t1_den) * H_d

        t2 = self.c * (self.M_s / self.a) * H_d * L_prime

        return (t1 + t2) / denominator

    def RK2(self, M_n1, H, H_n1, H_d, H_d_n1):
        """Compute hysteresis function with Runge-Kutta 2nd order

        Parameters
        ----------
        M_n1 : float
            Previous magnetisation
        H : float
            Magnetic field
        H_n1 : float
            Previous magnetic field
        H_d : float
            Magnetic field derivative
        H_d_n1 : float
            Previous magnetic field derivative

        Returns
        -------
        M : float
            Current magnetisation
        """
        k1 = self.T * self.dMdt(M_n1, H_n1, H_d_n1)
        k2 = self.T * self.dMdt(M_n1 + k1/2, (H + H_n1) /
                                2, (H_d + H_d_n1) / 2)
        return M_n1 + k2

    def RK4(self, M_n1, H, H_n1, H_d, H_d_n1):
        """Compute hysteresis function with Runge-Kutta 2nd order

        Parameters
        ----------
        M_n1 : float
            Previous magnetisation
        H : float
            Magnetic field
        H_n1 : float
            Previous magnetic field
        H_d : float
            Magnetic field derivative
        H_d_n1 : float
            Previous magnetic field derivative

        Returns
        -------
        M : float
            Current magnetisation
        """
        k1 = self.T * self.dMdt(M_n1, H_n1, H_d_n1)
        k2 = self.T * self.dMdt(M_n1 + k1/2, (H + H_n1) /
                                2, (H_d + H_d_n1) / 2)
        k3 = self.T * self.dMdt(M_n1 + k2/2, (H + H_n1) /
                                2, (H_d + H_d_n1) / 2)
        k4 = self.T * self.dMdt(M_n1 + k3, H, H_d)
        return M_n1 + (k1 / 6) + (k2 / 3) + (k3 / 3) + (k4 / 6)

    def process_block(self, x):
        """Process block of samples"""
        M_out = np.zeros(len(x))
        M_n1 = 0
        H_n1 = 0
        H_d_n1 = 0

        n = 0
        for H in x:
            H_d = self.deriv.differentiate(H)

            if self.mode == 'RK2':
                M = self.RK2(M_n1, H, H_n1, H_d, H_d_n1)
            elif self.mode == 'RK4':
                M = self.RK4(M_n1, H, H_n1, H_d, H_d_n1)

            M_n1 = M
            H_n1 = H
            H_d_n1 = H_d

            M_out[n] = M
            n += 1

        return M_out
