from .base import Processor
import numpy as np
from scipy.signal import find_peaks

class FindAmplitudeLegacy(Processor):
    """
    Detect the most prominent peak in the I-T profile at a given peak position.

    Stores the position and value of the peak in the context dictionary.

    Args:
        peak_position (int): Index for voltage sweep peak (default 257).
    """
    def __init__(self, peak_position=257):
        self.peak_position = peak_position

    def process(self, data, context=None):
        """
        Detect the highest peak in the I-T trace and update context with its value and location.

        Args:
            data (np.ndarray): 2D FSCV array.
            context (dict): Optional metadata dictionary.

        Returns:
            np.ndarray: Unchanged data array.
        """
        fx = data[:, self.peak_position]
        peaks, _ = find_peaks(fx, 
                              #prominence=0.2, 
                              #distance=10, 
                              height=0.1)
        # Get the actual peak values
        peak_values = fx[peaks]

        if len(peak_values) > 0:
            max_peak_idx = np.argmax(peak_values)
            peak_position = peaks[max_peak_idx]
            peak_value = peak_values[max_peak_idx]
            print("Peak position:", peak_position)
            print("Peak value:", peak_value)
        else:
            print("No peaks found.")

        # Update context with peak information
        if context is not None:
            context['peak_amplitude_positions'] = peak_position
            context['peak_amplitude_values'] = peak_value
        print(f"Found peaks at positions: {peak_position} with values: {peak_values}")
        return data
    
class FindAmplitude(Processor):
    def __init__(self, peak_position=257):
        self.peak_position = peak_position

    def process(self, data, context=None):
        """
        Usage:
        Finds the dominant local maxima of the input data and updates the context with their positions and values.
        """
        fx = data[:, self.peak_position]
        peaks, _ = find_peaks(fx, 
                              #prominence=0.2, 
                              #distance=10, 
                              height=0.1)
        # Get the actual peak values
        peak_values = fx[peaks]

        if len(peak_values) > 0:
            max_peak_idx = np.argmax(peak_values)
            peak_position = peaks[max_peak_idx]
            peak_value = peak_values[max_peak_idx]
            print("Peak position:", peak_position)
            print("Peak value:", peak_value)
        else:
            print("No peaks found.")

        # Update context with peak information
        if context is not None:
            context['peak_amplitude_positions'] = peak_position
            context['peak_amplitude_values'] = peak_value
        print(f"Found peaks at positions: {peak_position} with values: {peak_values}")
        return data
    
# class FindAmplitudeGradDesc:
#     """
#     A class to find the amplitude of a signal, local maxima using gradient descent. 
#     find peaks seems more appropriate for this task, but this is a placeholder for future work.
#     ""

#     def __init__(self, peak_position=257):
#         self.peak_position = peak_position 

#     def process(self, data, numiterations=1000, alpha=0.01):
#         fx = data[:, self.peak_position]
#         x = np.linspace(0, len(fx) - 1, len(fx))

#         # Interpolate function
#         f_interp = interp1d(x, fx, kind='cubic', fill_value='extrapolate')

#         # Compute and interpolate gradient
#         grad_fx = np.gradient(fx, x)
#         grad_interp = interp1d(x, grad_fx, kind='cubic', fill_value='extrapolate')

#         x_current = x[0]
#         for _ in range(numiterations):
#             grad = grad_interp(x_current)
#             x_next = x_current - alpha * grad

#             if f_interp(x_next) < f_interp(x_current):
#                 x_current = x_next
#             else:
#                 alpha *= 0.5

#         return x_current, f_interp(x_current)