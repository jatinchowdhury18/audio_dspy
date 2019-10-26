import audio_dspy as adsp
import numpy as np
import scipy.signal as signal


class Filter:
    """
    A filter that was created with a function from the eq_design
    module. Includes useful methods for processing, reseting,
    and plotting.
    """

    def __init__(self, order, fs, type='Other'):
        self.fs = fs
        self.order = order
        self.type = type
        self.b_coefs = np.zeros(order + 1)
        self.a_coefs = np.zeros(order + 1)
        self.__z = np.zeros(order + 1)

    def reset(self):
        """Resets the state of the filter
        """
        self.__z = np.zeros(self.order + 1)

    def has_been_reset(self):
        """Returns true if the filter state has been cleared
        """
        return np.sum(self.__z) == 0

    def set_coefs(self, b, a):
        """
        Set the coefficients of the filter

        Parameters
        ----------
        b : array-like
            Feed-forward coefficients. Must of the length order + 1
        a : array-like
            Feed-back coefficients. Must of the length order + 1
        """
        assert np.size(b) == (
            self.order + 1), 'b coefficients size is not the same as filter order'
        assert np.size(a) == (
            self.order + 1), 'a coefficients size is not the same as filter order'

        self.b_coefs = np.copy(b)
        self.a_coefs = np.copy(a)

    def process_sample(self, x):
        """
        Processes a sample through the filter, using the Transposed
        Direct Form II filter form (https://ccrma.stanford.edu/~jos/filters/Transposed_Direct_Forms.html)

        Parameters
        ----------
        x : float
            Input sample

        Returns
        -------
        y : float
            Output sample
        """
        y = self.__z[1] + x * self.b_coefs[0]

        for n in range(self.order):
            self.__z[n] = self.__z[n+1] + x * \
                self.b_coefs[n] - y * self.a_coefs[n]

        self.__z[self.order] = x * self.b_coefs[self.order] - \
            y * self.a_coefs[self.order]
        return y

    def process_block(self, block):
        """
        Process a block of samples.

        Parameters
        ----------
        block : array-like
            The block of samples to process

        Returns
        -------
        output : array-like
            Block of output samples
        """
        out = np.copy(block)
        for n, _ in enumerate(block):
            out[n] = self.process_sample(block[n])

        return out


class EQ:
    """
    An audio equalizer object. Functionally, this this object holds several filters
    all of which can be created with the eq_design module, and provides several useful
    functions for interacting with them, including processing, reseting, and plotting.
    """

    def __init__(self, fs):
        self.fs = fs
        self.filters = []

    def add_filter(self, filter):
        """
        Add a filter to the EQ

        Parameters
        ----------
        filter : Filter
            The filter to add
        """
        assert isinstance(filter, Filter), 'filter must be of adsp.Filter type'
        self.filters.append(filter)

    def add_LPF(self, fc, Q):
        """
        Add a lowpass filter to the EQ

        Parameters
        ----------
        fc : float
            Cutoff frequency
        Q : float
            Q factor
        """
        string = 'LPF, Freq: {}, Q: {}'.format(fc, Q)
        filter = adsp.Filter(2, self.fs, type=string)
        b, a = adsp.design_LPF2(fc, Q, self.fs)
        filter.set_coefs(b, a)
        self.add_filter(filter)

    def add_HPF(self, fc, Q):
        """
        Add a highpass filter to the EQ

        Parameters
        ----------
        fc : float
            Cutoff frequency
        Q : float
            Q factor
        """
        string = 'HPF, Freq: {}, Q: {}'.format(fc, Q)
        filter = adsp.Filter(2, self.fs, type=string)
        b, a = adsp.design_HPF2(fc, Q, self.fs)
        filter.set_coefs(b, a)
        self.add_filter(filter)

    def add_bell(self, fc, Q, gain):
        """
        Add a bell filter to the EQ

        Parameters
        ----------
        fc : float
            Cutoff frequency
        Q : float
            Q factor
        gain : float
            gain in linear units
        """
        string = 'Bell, Freq: {}, Q: {}, gain: {}'.format(fc, Q, gain)
        filter = adsp.Filter(2, self.fs, type=string)
        b, a = adsp.design_bell(fc, Q, gain, self.fs)
        filter.set_coefs(b, a)
        self.add_filter(filter)

    def add_notch(self, fc, Q):
        """
        Add a notch filter to the EQ

        Parameters
        ----------
        fc : float
            Cutoff frequency
        Q : float
            Q factor
        """
        string = 'Notch, Freq: {}, Q: {}'.format(fc, Q)
        filter = adsp.Filter(2, self.fs, type=string)
        b, a = adsp.design_notch(fc, Q, self.fs)
        filter.set_coefs(b, a)
        self.add_filter(filter)

    def add_highshelf(self, fc, Q, gain):
        """
        Add a highshelf filter to the EQ

        Parameters
        ----------
        fc : float
            Cutoff frequency
        Q : float
            Q factor
        gain : float
            gain in linear units
        """
        string = 'High Shelf, Freq: {}, Q: {}, gain: {}'.format(fc, Q, gain)
        filter = adsp.Filter(2, self.fs, type=string)
        b, a = adsp.design_highshelf(fc, Q, gain, self.fs)
        filter.set_coefs(b, a)
        self.add_filter(filter)

    def add_lowshelf(self, fc, Q, gain):
        """
        Add a lowshelf filter to the EQ

        Parameters
        ----------
        fc : float
            Cutoff frequency
        Q : float
            Q factor
        gain : float
            gain in linear units
        """
        string = 'Low Shelf, Freq: {}, Q: {}, gain: {}'.format(fc, Q, gain)
        filter = adsp.Filter(2, self.fs, type=string)
        b, a = adsp.design_lowshelf(fc, Q, gain, self.fs)
        filter.set_coefs(b, a)
        self.add_filter(filter)

    def reset(self):
        """
        Resets the state of the EQ
        """
        for filter in self.filters:
            filter.reset()

    def process_block(self, block):
        """
        Process a block of samples.

        Parameters
        ----------
        block : array-like
            The block of samples to process

        Returns
        -------
        output : array-like
            Block of output samples
        """
        out = np.copy(block)
        for filter in self.filters:
            out = filter.process_block(out)

        return out

    def plot_eq_curve(self, worN=512):
        """
        Plots the magnitude response of the EQ

        worN: {None, int, array_like}, optional
            If a single integer, then compute at that many frequencies (default is N=512).
            If an array_like, compute the response at the frequencies given. These are in the same units as fs.
        """
        assert len(self.filters) > 0, 'Trying to plot an empty EQ!'

        w, H = signal.freqz(
            self.filters[0].b_coefs, self.filters[0].a_coefs, worN=worN, fs=self.fs)

        H_sum = np.zeros(len(H))
        for filter in self.filters:
            w, H = signal.freqz(
                filter.b_coefs, filter.a_coefs, worN=worN, fs=self.fs)
            H_sum += np.abs(H)

        adsp.plot_freqz_mag(w, H_sum / len(self.filters))

    def print_eq_info(self):
        """
        Print the specs of the EQ
        """
        for filter in self.filters:
            if filter.type == 'Other':
                print('Filter: b_coefs: {}, a_coefs: {}'.format(
                    filter.b_coefs, filter.a_coefs))
            else:
                print(filter.type)
