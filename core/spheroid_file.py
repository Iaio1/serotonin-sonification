import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from matplotlib import cm
from dataclasses import dataclass, field
from typing import List

from matplotlib.animation import FuncAnimation

class SpheroidFile:
    """
    A class to handle the loading and processing of FSCV spheroid data files.
    Usage:
    - filepath: Path to the associated FSCV (color plot) data file (txt format).
    - raw_data: The raw data loaded from the file, inverted. We keep it to always come back in case the user does not want to apply any processing.
    - processed_data: The processed data, initially set to raw_data.
    - peak_position: The position of the peak in the I-T profile, default is set to 257 (for serotonin, 5HT).
    - window_size: The size of the window for rolling mean or smoothing, default is None. (not used yet)
    - metadata: A dictionary to hold any additional metadata related to the file.
    """

    def __init__(self, filepath, waveform="5HT"):
        self.filepath = filepath
        self.raw_data = self.load_data()
        self.processed_data = self.raw_data
        # Default peak position for serotonin (5HT) in FSCV plots
        self.peak_position = 257
        self.window_size = None  # Default window size for rolling mean or smoothing
        self.metadata = {}
        self.waveform = waveform  # Default waveform etc.
        self.timeframe = self.raw_data.shape[0]
        print(
            f"Loaded SpheroidFile from {self.filepath} with shape {self.raw_data.shape} and waveform {self.waveform}")

    def load_data(self):
        # Transpose to match (voltage steps, time points)
        initial_data = np.loadtxt(self.filepath).T
        data = -initial_data  # Invert the data since FSCV txt color plots will be inverted
        return data

    def set_peak_position(self, peak_position):
        self.peak_position = peak_position

    def set_waveform(self, waveform):
        """
        Sets the waveform type for the spheroid file.
        This can be used to differentiate between different types of stimulation or analysis.
        """
        self.waveform = waveform

    def get_data(self):
        return self.raw_data
    
    def get_processed_data(self):
        """
        Returns the processed data.
        This is a 2D array where rows represent voltage steps and columns represent time points.
        """
        return self.processed_data
    
    def get_processed_data_IT(self):
        """
        Returns the processed data for the I-T profile.
        This is a 1D array representing the current at the peak position across all time points.
        """
        return self.processed_data[:, self.peak_position]

    def get_filepath(self):
        return self.filepath

    def update_metadata(self, context):
        """
        Updates the metadata with the provided context.
        """
        self.metadata.update(context)

    def get_metadata(self):
        return self.metadata

    def visualize_color_plot_data(self, title_suffix=""):
        # Initialize plot settings
        plot_settings = PLOT_SETTINGS()
        custom_cmap = plot_settings.custom

        # Use the loaded data
        processed_data = self.processed_data

        # Calculate limits
        vmin = np.percentile(processed_data, 1)
        vmax = np.percentile(processed_data, 99)

        # Create and save plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(processed_data.T,  # Transpose to match time points and voltage steps, imshow expects (y, x)
                       aspect='auto',
                       cmap=custom_cmap,
                       origin='lower',
                       extent=[0, processed_data.shape[0],
                               0, processed_data.shape[1]],
                       vmin=vmin,
                       vmax=vmax)
        plt.colorbar(im, ax=ax, label="Current (nA)")
        ax.set_xlabel("Time Points")
        ax.set_ylabel("Voltage Steps")
        ax.set_title(
            f"Color Plot{': ' + title_suffix if title_suffix else ''}\nRange: [{vmin:.2f}, {vmax:.2f}] nA")

        plt.tight_layout()
        plt.show()

        return fig, ax

    def visualize_3d_color_plot(self, title_suffix=""):
        """
        3D surface with the same orientation as:
        ax.imshow(self.processed_data.T, …)
        Time → X, Voltage → Y, Current → Z.
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        # 1) grab raw D of shape (voltage_steps, time_points):
        D = self.processed_data

        # 2) down-sample (optional) but keep (volt, time) ordering
        dy, dx = 10, 10
        volt_idx = np.arange(0, D.shape[0], dy)
        time_idx = np.arange(0, D.shape[1], dx)
        # shape = (V_small, T_small)
        D_small = D[np.ix_(volt_idx, time_idx)]

        # 3) transpose D_small so that rows→time, cols→voltage
        # shape = (T_small, V_small)
        Z = D_small.T

        # 4) build X,Y so that X[i,j] = time_idx[i], Y[i,j] = volt_idx[j]
        T_small = time_idx
        V_small = volt_idx
        # (V_small, T_small)
        X, Y = np.meshgrid(T_small, V_small, indexing='xy')
        X = X.T  # now shape = (T_small, V_small)
        Y = Y.T  # now shape = (T_small, V_small)

        # 5) color-normalize over the **full** D so it matches your 2D scale
        vmin, vmax = np.percentile(D, 1), np.percentile(D, 99)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = PLOT_SETTINGS().custom

        # 6) plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X, Y, Z,
            facecolors=cmap(norm(Z)),
            rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False
        )
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(Z)
        fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label="Current (nA)")

        ax.set_xlabel("Voltage Steps")
        ax.set_ylabel("Time Points")
        ax.set_zlabel("Current (nA)")
        ax.set_title(
            f"3D Surface Color Plot{': ' + title_suffix if title_suffix else ''}")

        # if your 2D used origin='lower', flip Y so the “0 V” is at the front
        ax.invert_yaxis()
        ax.view_init(elev=30, azim=180)
        ax.grid(False)

        plt.tight_layout()
        plt.show()
        return fig, ax

    def animate_3d_color_plot(self, title_suffix=""):
        """
        Creates an animation for the 3D surface plot.
        Starts at elev=90, azim=180, transitions to elev=0, and then to elev=0, azim=270.
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        # 1) Grab raw data of shape (voltage_steps, time_points):
        D = self.processed_data

        # 2) Down-sample (optional) but keep (volt, time) ordering
        dy, dx = 10, 10
        volt_idx = np.arange(0, D.shape[0], dy)
        time_idx = np.arange(0, D.shape[1], dx)
        D_small = D[np.ix_(volt_idx, time_idx)]  # shape = (V_small, T_small)

        # 3) Transpose D_small so that rows → time, cols → voltage
        Z = D_small.T  # shape = (T_small, V_small)

        # 4) Build X, Y so that X[i, j] = time_idx[i], Y[i, j] = volt_idx[j]
        T_small = time_idx
        V_small = volt_idx
        X, Y = np.meshgrid(T_small, V_small, indexing='xy')  # (V_small, T_small)
        X = X.T  # now shape = (T_small, V_small)
        Y = Y.T  # now shape = (T_small, V_small)

        # 5) Color-normalize over the **full** D so it matches your 2D scale
        vmin, vmax = np.percentile(D, 1), np.percentile(D, 99)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = PLOT_SETTINGS().custom

        # 6) Create the figure and initial 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X, Y, Z,
            facecolors=cmap(norm(Z)),
            rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False
        )
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(Z)
        fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label="Current (nA)")

        ax.set_xlabel("Voltage Steps")
        ax.set_ylabel("Time Points")
        ax.set_zlabel("Current (nA)")
        #ax.set_title(f"3D Surface Color Plot{': ' + title_suffix if title_suffix else ''}")

        # Invert the y-axis to match the expected orientation
        ax.invert_yaxis()

        # Set the initial view
        ax.view_init(elev=90, azim=180)


        # Animation parameters
        start_elev = 90
        start_azim = 180
        end_elev = 0
        end_azim = 270
        frames = 100  # Total number of frames for the animation

        def update(frame):
            # Transition from elev=90 to elev=0
            if frame < frames // 2:
                elev = start_elev - (start_elev - end_elev) * (frame / (frames // 2))
                azim = start_azim
            else:
                # Transition from azim=180 to azim=270
                elev = end_elev
                azim = start_azim + (end_azim - start_azim) * ((frame - frames // 2) / (frames // 2))
            ax.view_init(elev=elev, azim=azim)

        # Create the animation
        ani = FuncAnimation(fig, update, frames=frames, interval=50)

        # Save the animation as a video file
        ani.save("3d_plot_animation.mp4", writer="ffmpeg", dpi=300)

        plt.show()
        
    def visualize_IT_profile(self):
        """
        Visualizes the I-T profile at the specified peak position and highlights all detected peaks.
        """
        # Extract the profile at the peak position
        profile = self.processed_data[:, self.peak_position]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(profile, color='blue', linewidth=1.5, label="I-T Profile")

        # Highlight all peak positions if available in metadata
        if 'peak_amplitude_positions' in self.metadata:
            peak_indices = self.metadata['peak_amplitude_positions']
            peak_values = self.metadata.get(
                'peak_amplitude_values', profile[peak_indices])
            print(f"Peak indices from metadata: {peak_indices}")
            # Iterate over index-value pairs
            for peak_idx, peak_val in zip(peak_indices, peak_values):
                # Check the peak is within bounds
                if 0 <= peak_idx < len(profile):
                    ax.scatter(peak_idx, peak_val, color='red',
                               label="Peak Amplitude", zorder=5)
                    ax.annotate(f"({peak_idx}, {peak_val:.2f})",
                                (peak_idx, peak_val),
                                textcoords="offset points",
                                xytext=(10, 10),
                                ha='center',
                                fontsize=10,
                                color='red')

        # Add labels, title, and grid
        ax.set_xlabel("Time Points")
        ax.set_ylabel("Current (nA)")
        ax.set_title(f"I-T Profile at Peak Position {self.peak_position}")
        ax.grid(False)
        ax.legend()

        # Show the plot
        plt.show()

    def visualize_IT_with_exponential_decay(self):
        """
        Visualizes the I-T profile, highlights the detected peak, overlays the exponential decay curve,
        and marks the half-life (t_half).
        """
        # Extract the I-T profile at the specified peak position
        profile = self.processed_data[:, self.peak_position]

        # Extract the exponential decay parameters from metadata
        if "exponential fitting parameters" not in self.metadata:
            raise ValueError(
                "Exponential fitting parameters are missing from metadata.")

        params = self.metadata["exponential fitting parameters"]
        A = params["A"]
        tau = params["tau"]
        C = params["C"]
        t_half = params["t_half"]  # Directly retrieve t_half from params

        # Extract the peak position from metadata
        if "peak_amplitude_positions" not in self.metadata:
            raise ValueError(
                "Peak amplitude positions are missing from metadata.")

        peak_positions = self.metadata["peak_amplitude_positions"]
        if len(peak_positions) == 0:
            raise ValueError("No peak positions found in metadata.")

        peak_position = int(peak_positions[0])  # Use the first peak position

        # Slice the profile starting from the peak position
        y = profile[peak_position:]
        t = np.arange(peak_position, peak_position + len(y))

        # Generate the exponential decay curve
        exp_decay_curve = A * np.exp(-t / tau) + C

        # Calculate the corresponding position and value for t_half
        # Convert t_half to the corresponding position
        t_half_position = peak_position + int(t_half)
        # Value of the exponential decay at t_half
        t_half_value = A * np.exp(-t_half / tau) + C

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(profile, color='blue', linewidth=1.5, label="I-T Profile")
        ax.scatter(peak_position, profile[peak_position],
                   color='red', label="Detected Peak", zorder=5)
        ax.plot(t, exp_decay_curve, color='green',
                linestyle='--', label="Exponential Decay Fit")

        # Plot the half-life (t_half)
        ax.axvline(x=t_half_position, color='purple',
                   linestyle=':', label=f"t_half = {t_half:.2f}")

        # Add labels, title, and legend
        ax.set_xlabel("Time Points")
        ax.set_ylabel("Current (nA)")
        ax.set_title(f"I-T Profile with Exponential Decay Fit and Half-Life")
        ax.legend()
        ax.grid(False)

        # Show the plot
        plt.show()

    def visualize_cv(self, time_sec=84, smooth=False, sg_window=11, sg_poly=3):
        """
        Visualize a cyclic voltammogram (CV) at a given time point using the reconstructed voltage waveform.

        Args:
            time_sec (int): Index or timepoint (in seconds) from which to extract the CV.
            smooth (bool): If True, apply Savitzky-Golay smoothing to the current trace (not implemented here).
            sg_window (int): Window size for Savitzky-Golay filter.
            sg_poly (int): Polynomial order for Savitzky-Golay filter.

        Returns:
            fig, ax: Matplotlib Figure and Axes objects of the generated plot.
        """
        if self.waveform != "5HT":
            raise ValueError(
                "This method is currently only implemented for the 5HT waveform.")

        # Generate the voltage waveform based on the 5HT scan parameters and number of data points
        wf = Waveforms(0.2, [1.0, -0.1], 0.2, 1000,
                       self.processed_data.shape[1])
        voltage = wf.voltage_waveform()

        # Extract current values for the requested CV (time point)
        current = self.processed_data[time_sec, :]

        # Find the peak current and corresponding voltage
        peak_i = np.argmax(current)
        peak_V = voltage[peak_i]
        peak_I = current[peak_i]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(voltage, current, label="CV", color="blue", lw=1.5)
        ax.scatter([peak_V], [peak_I], color="red", zorder=5)
        ax.annotate(f"Current peak: {peak_V:.2f} V",
                    xy=(peak_V, peak_I),
                    xytext=(peak_V + 0.1, peak_I + 0.1),
                    arrowprops=dict(arrowstyle="->"))

        ax.set_xlabel("Voltage (V)", fontsize=18)
        ax.set_ylabel("Current (nA)", fontsize=18)
        ax.set_title(f"Cyclic Voltammogram at {time_sec:.1f} s", fontsize=16)
        ax.legend(fontsize=14)
        ax.grid(False)
        plt.tight_layout()
        plt.show()

        return fig, ax


@dataclass
class Waveforms:
    """FSCV waveform voltage generator

    Attributes:
        Vstart (float): Starting voltage of the waveform.
        Vvertices (List[float]): List of vertex voltages (any number allowed).
        Vend (float): Final voltage at the end of the scan.
        scan_rate (int): Scan rate in V/s.
        N_data_points (int): Total number of data points in the scan.

    Methods:
        get_segments(): Returns list of (start, end) voltage pairs for each segment.
        segment_times(): Returns duration of each segment in seconds.
        total_time(): Returns total scan time.
        voltage_waveform(): Returns numpy array with voltage for each point.
    """
    Vstart: float = 0.2
    Vvertices: List[float] = field(default_factory=lambda: [1, -0.1])
    Vend: float = 0.2
    scan_rate: int = 1000
    N_data_points: int = 1100

    def get_segments(self):
        """Generate consecutive (start, end) pairs for all waveform segments."""
        v_points = [self.Vstart] + self.Vvertices + [self.Vend]
        return list(zip(v_points[:-1], v_points[1:]))

    def segment_times(self):
        """Calculate the duration (s) of each segment based on scan rate."""
        return [abs(e - s) / self.scan_rate for s, e in self.get_segments()]

    def total_time(self):
        """Return total duration of the scan (s)."""
        return sum(self.segment_times())

    def voltage_waveform(self):
        """
        Generate the complete voltage waveform for all points in the scan.

        Returns:
            np.array: Array of voltages, length = N_data_points.
        """
        segs = self.get_segments()
        times = self.segment_times()
        total_time = self.total_time()

        points = [int(t / total_time * self.N_data_points) for t in times]
        points[-1] = self.N_data_points - sum(points[:-1])

        voltage = []
        for (start, end), npts in zip(segs, points):
            voltage.extend(list(np.linspace(start, end, npts, endpoint=False)))
        return np.array(voltage)


class PLOT_SETTINGS:
    def __init__(self):
        # Custom colormap only
        self.custom = self.get_continuous_cmap(
            ['#001524', '#002f5e', '#f4c300', '#a84900',
                '#64005f', '#00a37a', '#00751c', '#00ff00'],
            [0, 0.2478, 0.3805, 0.6555, 0.701, 0.7603, 0.7779, 1]
        )

    def get_continuous_cmap(self, hex_list, float_list=None):
        rgb_list = [self.rgb_to_dec(self.hex_to_rgb(i)) for i in hex_list]
        if float_list is None:
            float_list = list(np.linspace(0, 1, len(rgb_list)))

        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]]
                        for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mcolors.LinearSegmentedColormap(
            'my_cmp', segmentdata=cdict, N=256)
        return cmp

    def hex_to_rgb(self, value):
        value = value.strip("#")
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_dec(self, value):
        return [v/256 for v in value]


if __name__ == "__main__":
    Spheroid_test = SpheroidFile(
        r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241111_batch1_n1_Sert/04_-30_COLOR.txt")
    data = Spheroid_test.get_processed_data()
    print(f"Data shape: {data.shape}")
    # plt.show()
    Spheroid_test.visualize_color_plot_data()
    Spheroid_test.visualize_cv()
    plt.show()
