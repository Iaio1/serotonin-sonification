from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QProgressDialog, QApplication
from PyQt5.QtCore import QSettings
import numpy as np
from scipy.optimize import curve_fit

from PyQt5.QtCore import Qt

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        import matplotlib

        matplotlib.rcParams.update({
            "font.family": ["Helvetica", "Arial"],
            "font.size": 10,
        })
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.add_subplot(111) 
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()
        self.cbar = None

    def plot_color(self, processed_data, peak_pos = None, title_suffix=None):
        from core.spheroid_file import PLOT_SETTINGS  # Ensure it's correctly imported

        plot_settings = PLOT_SETTINGS()
        custom_cmap = plot_settings.custom

        vmin = np.percentile(processed_data, 1)
        vmax = np.percentile(processed_data, 99)

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

        im = self.axes.imshow(
            processed_data.T,       # Transpose to align voltage steps as rows
            aspect='auto',
            cmap=custom_cmap,
            origin='lower',
            extent=[0, processed_data.shape[0], 0, processed_data.shape[1]],
            vmin=vmin,
            vmax=vmax
        )
        if peak_pos != None:
            self.axes.axhline(
                y=peak_pos, 
                color='white', 
                linewidth=1, 
                linestyle='--'
            )
        # Add colorbar using the figure object (Qt-safe)
        self.cbar = self.fig.colorbar(im, ax=self.axes, label="Current (nA)")

        self.axes.set_xlabel("Time Points")
        self.axes.set_ylabel("Voltage Steps")
        title = f"Color Plot{': ' + title_suffix if title_suffix else ''}"
        self.axes.set_title(title, fontweight='bold')       

        self.fig.tight_layout()
        self.draw()

    def plot_IT(self, processed_data, metadata=None, peak_position=None):
        """
        Enhanced plot_IT method that includes decay visualization.
        
        Parameters:
        - processed_data: 2D numpy array 
        - metadata: dict containing peak and decay information
        - peak_position: column index for the peak position
        """
        self.fig.clear()
        self.axes.clear()
        
        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)

        self.axes = self.fig.add_subplot(111)

        profile = processed_data[:, peak_position]
        n = profile.shape[0]
        t = np.arange(n) / freq   # in seconds

        # Plot the main profile
        self.axes.plot(t, profile, label="I-T Profile", color='#4178F2', linewidth=1.5)

        # Plot peak markers if metadata is provided
        if metadata and 'peak_amplitude_positions' in metadata:
            peak_indices = metadata['peak_amplitude_positions']
            peak_values = metadata.get('peak_amplitude_values', profile[peak_indices])
            
            # Handle both single value and list/array for peaks
            if isinstance(peak_indices, (list, np.ndarray)) and isinstance(peak_values, (list, np.ndarray)):
                for idx, val in zip(peak_indices, peak_values):
                    if 0 <= idx < len(profile):
                        self.axes.scatter(t[idx], val, color='#FF3877', s=100, zorder=5)
                        self.axes.annotate(f"{val:.2f}", (t[idx], val), textcoords="offset points",
                                           xytext=(0, 10), ha='center', fontsize=9, color='#FF3877')
            else:
                # Assume single value
                try:
                    idx = int(peak_indices)
                    val = float(peak_values)
                    if 0 <= idx < len(profile):
                        self.axes.scatter(t[idx], val, color='#FF3877', s=100, zorder=5, label='Peak')
                        self.axes.annotate(f"{val:.2f}", (t[idx], val), textcoords="offset points",
                                           xytext=(0, 10), ha='center', fontsize=9, color='#FF3877')
                except Exception:
                    pass

            # Plot decay regions if available
            self._plot_decay_regions(metadata, t, freq)


        # Add validation status to title
        validation_status = ""
        if metadata and 'decay_validation_params' in metadata:
            is_valid = metadata['decay_validation_params'].get('peak_passed_validation', False)
            validation_status = "(Valid" if is_valid else "(Invalid"

        # Axis labeling and formatting
        self.axes.set_xlabel("Time (seconds)")
        self.axes.set_ylabel("Current (nA)")
        title = f"I-T Profile {validation_status}"
        if peak_position is not None:
            title += f" Peak at Position {peak_position})"
        self.axes.set_title(title, fontweight="bold")
        self.axes.grid(True, alpha=0.3)
        self.axes.legend()

        max_t = t[-1]
        tick_interval = 5  # seconds
        ticks = np.arange(0, max_t + tick_interval, tick_interval)
        self.axes.set_xticks(ticks)

        self.draw()

    def _plot_decay_regions(self, metadata, t, freq):
        """Plot decay regions on the current axes"""
        # Plot left decay region
        if 'decay_left_region' in metadata:
            left_data = metadata['decay_left_region']
            if 'indices' in left_data and 'values' in left_data:
                left_indices = list(left_data['indices'])
                left_values = left_data['values']
                if len(left_indices) > 0 and len(left_values) > 0:
                    left_times = [t[i] for i in left_indices if i < len(t)]
                    self.axes.plot(left_times, left_values, 'o-', color='orange', 
                                  alpha=0.8, markersize=4, label='Left Decay Region')

        # Plot right decay region
        if 'decay_right_region' in metadata:
            right_data = metadata['decay_right_region']
            if 'indices' in right_data and 'values' in right_data:
                right_indices = list(right_data['indices'])
                right_values = right_data['values']
                if len(right_indices) > 0 and len(right_values) > 0:
                    right_times = [t[i] for i in right_indices if i < len(t)]
                    self.axes.plot(right_times, right_values, 'o-', color='green', 
                                  alpha=0.8, markersize=4, label='Right Decay Region')

    # Keep all your existing methods unchanged
    def show_average_over_experiments(self, group_analysis):
        """
        Plots the mean amplitudes over time across all experiments,
        with the standard deviation as a shaded area.
        """
        import numpy as np

        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()  # Ensure dialog appears

        # Get data from group_analysis
        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = group_analysis.amplitudes_over_time_all_experiments()
        
        if time_points is None:
            self.axes.clear()
            self.axes.set_title("No data to plot")
            self.draw()
            progress.close()
            return

        all_amplitudes = np.array(all_amplitudes, dtype=float)
        std_amplitudes = np.nanstd(all_amplitudes, axis=0)

        # Clear and plot
        self.axes.clear()
        self.axes.plot(time_points, mean_amplitudes, label='Mean Amplitude', color='purple')
        self.axes.fill_between(time_points, mean_amplitudes - std_amplitudes, mean_amplitudes + std_amplitudes,
                            color='purple', alpha=0.2, label='SD')
        if files_before_treatment > 0:
            treatment_time = files_before_treatment * (time_points[1] - time_points[0])
            self.axes.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
        self.axes.set_xlabel('Time (minutes)')
        self.axes.set_ylabel('Amplitude')
        self.axes.set_title('Mean Amplitude Over Time (All Experiments)')
        self.axes.legend()
        self.axes.grid(False)
        self.fig.tight_layout()
        self.draw()
        progress.close()

    def show_amplitudes_over_time(self, group_analysis):
        """
        Plot all amplitudes over time for each experiment as separate lines on the embedded canvas.
        """
        import numpy as np

        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()  # Ensure dialog appears

        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = group_analysis.amplitudes_over_time_all_experiments()
        if time_points is None:
            self.axes.clear()
            self.axes.set_title("No data to plot")
            self.draw()
            progress.close()
            return

        all_amplitudes = np.array(all_amplitudes, dtype=float)
        treatment_time = files_before_treatment * (time_points[1] - time_points[0])

        self.axes.clear()
        for i, amplitudes in enumerate(all_amplitudes):
            self.axes.plot(time_points, amplitudes, label=f'Experiment {i+1}', alpha=0.7)
        if files_before_treatment > 0:
            self.axes.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
        self.axes.set_xlabel('Time (min)')
        self.axes.set_ylabel('Amplitude')
        self.axes.set_title('Amplitudes Over Time (All Experiments)')
        self.axes.legend()
        self.axes.set_xticks(np.arange(0, max_t + tick_interval, tick_interval))
        self.fig.tight_layout()
        self.draw()
        progress.close()