
import numpy as np

class Simulator(object):

    def __init__(self, dt=1, N=1024):
        """
        Methods to simulate and visualize light curves.

        Parameters:
        ----------
        dt: int
            time resolution of simulated light curve
        N: int
            bins of simulated light curve
        """
        self.dt = dt
        self.N = N

    def simulate(self, *args):
        """
        Simulate light curve generation using power spectrum or
        impulse response.

        Examples:
        --------
        - x = simulate(2)
            For generating a light curve using power law spectrum.

            Parameters:
            -----------
            Beta: int
                Defines the shape of spectrum
            N: int
                Number of samples

            Returns:
            --------
            lightCurve: `LightCurve` object

        - x = simulate(s,h)
            For generating a light curve using impulse response.

            Parameters:
            -----------
            s: array-like
                Underlying variability signal
            h: array-like
                Impulse response

            Returns:
            -------
            lightCurve: `LightCurve` object
        """

        if type(args[0]) == int:
            return  self._simulate_power_law(args[0])

        elif len(args) == 2:
            return self._simulate_impulse_response(args[0], args[1])

        else:
            raise AssertionError("Length of arguments must be 1 or 2.")


    def _simulate_power_law(self, B):

        """
        Generate LightCurve from a power law spectrum.

        Parameters:
        ----------
        B: int
            Defines the shape of power law spectrum.

        Returns
        -------
        lightCurve: array-like
        """

        # Define frequencies from 0 to 2*pi
        w = np.linspace(0.001,2*np.pi,self.N)

        # Draw two set of 'N' guassian distributed numbers
        a1 = np.random.normal(size=self.N)
        a2 = np.random.normal(size=self.N)

        # Multiply by (1/w)^B to get real and imaginary parts
        real = a1 * np.power((1/w),B/2)
        imaginary = a2 * np.power((1/w),B/2)

        # Obtain time series
        return self._find_inverse(real, imaginary)

    def _simulate_impulse_response(self, s, h):
        pass

    def _find_inverse(self, real, imaginary):
        """
        Forms complex numbers corresponding to real and imaginary
        parts and finds inverse series.
        """

        # Form complex numbers corresponding to each frequency
        f = [complex(r, i) for r,i in zip(real,imaginary)]

        # Obtain real valued time series
        f_conj = np.conjugate(np.array(f))

        # Obtain time series
        return np.real(np.fft.ifft(f_conj))
