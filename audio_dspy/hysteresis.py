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
        mode : {'RK2', 'RK4', 'NR*'}, optional
            'RK2'
              By default, this class will use the 2nd-order
              Runge-Kutta method for solving the Jiles-Atherton
              equation

            'RK4'
              Mode uses the 4th-order Runge-Kutta method

            'NR{X}'
              Mode uses Newton-Raphson iterations, with 'X'
              maximum iterations, i.e. 'NR10' corresponds to
              a Newton-Raphson iteration that times out after
              10 iterations.
        """
        self.deriv = Differentiator(fs, dAlpha)
        self.T = 1.0 / fs

        self.M_s = 0.5 + 1.5*(1-sat)  # saturation
        self.a = self.M_s / (0.01 + 6*drive)  # adjustable parameter
        self.alpha = 1.6e-3
        self.k = 30 * (1-0.5)**6 + 0.01  # Coercivity
        self.c = (1-width)**0.5 - 0.01  # Changes slope

        assert mode == 'RK2' or mode == 'RK4' or mode[:2] == 'NR', "Invalid mode!"
        self.mode = mode

    @staticmethod
    def langevin(x):
        """Langevin function: coth(x) - (1/x)"""
        if (abs(x) > 10 ** -4):
            return (1 / np.tanh(x)) - (1/x)
        else:
            return (x / 3)

    @staticmethod
    def langevin_deriv(x):
        """Derivative of the Langevin function: (1/x^2) - coth(x)^2 + 1"""
        if (abs(x) > 10 ** -4):
            return (1 / x ** 2) - (1 / np.tanh(x)) ** 2 + 1
        else:
            return (1 / 3)

    @staticmethod
    def langevin_deriv2(x):
        """2nd derivative of the Langevin function: 2 coth(x) (coth(x)^2 - 1) - 2/x^3"""
        if (abs(x) > 10 ** -3):
            return 2 * (1 / np.tanh(x)) * ((1 / np.tanh(x)) ** 2 - 1) - (2 / x ** 3)
        else:
            return -2 * x / 15

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
        """Compute hysteresis function with Runge-Kutta 4th order

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

    def dMdt_prime(self, M, H, H_d):
        """Jiles-Atherton differential equation Jacobian

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
            Derivative of dMdt w.r.t. M
        """
        Q = (H + self.alpha * M) / self.a
        M_diff = self.M_s * self.langevin(Q) - M
        delta = 1 if H_d > 0 else -1
        delta_M = 1 if np.sign(delta) == np.sign(M_diff) else 0
        L_prime = self.langevin_deriv(Q)
        L_prime2 = self.langevin_deriv2(Q)
        M_diff2 = self.alpha * self.M_s * L_prime / self.a - 1

        k1 = (1 - self.c) * delta_M
        k2 = (1 - self.c) * delta * self.k

        f1_denom = k2 - self.alpha * M_diff
        f1 = H_d * k1 * M_diff / f1_denom
        f2 = self.c * self.M_s * H_d * L_prime / self.a
        f3 = 1 - self.c * self.alpha * self.M_s * L_prime / self.a

        f1_p = k1 * H_d * ((M_diff2 / f1_denom) +
                           (M_diff * self.alpha * M_diff2 / f1_denom ** 2))
        f2_p = self.c * self.alpha * self.M_s * H_d * L_prime2 / self.a ** 2
        f3_p = -self.c * self.alpha ** 2 * self.M_s * L_prime2 / self.a ** 2

        return ((f1_p + f2_p) * f3 - (f1 + f2) * f3_p) / f3 ** 2

    def NR(self, M_n1, H, H_n1, H_d, H_d_n1):
        """Compute hysteresis function with Newton-Raphson iteration

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
        N_iter = int(self.mode[2:])

        M = M_n1
        T_2 = self.T / 2.0
        last_dMdt = self.dMdt(M_n1, H_n1, H_d_n1)
        for _ in range(N_iter):
            delta = (M - M_n1 - T_2 * (self.dMdt(M, H, H_d) + last_dMdt)
                     ) / (1 - T_2 * self.dMdt_prime(M, H, H_d))
            M -= delta

        return M

    def process_block(self, x):
        """Process block of samples"""
        M_out = np.zeros(len(x))
        M_n1 = 0
        H_n1 = 0
        H_d_n1 = 0

        if self.mode == 'RK2':
            solver = self.RK2
        elif self.mode == 'RK4':
            solver = self.RK4
        elif self.mode[:2] == 'NR':
            solver = self.NR

        n = 0
        for H in x:
            H_d = self.deriv.differentiate(H)
            M = solver(M_n1, H, H_n1, H_d, H_d_n1)

            M_n1 = M
            H_n1 = H
            H_d_n1 = H_d

            M_out[n] = M
            n += 1

        return M_out
