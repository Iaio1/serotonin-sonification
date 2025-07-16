from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QProgressDialog, QApplication
from PyQt5.QtCore import QSettings
import numpy as np

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

    def plot_IT(self,processed_data, metadata=None, peak_position=None):
        """
        Plot the I-T (Intensity-Time) profile on the Qt canvas.

        Parameters:
        - profile: 1D numpy array representing current over time
        - metadata: dict containing 'peak_amplitude_positions' and optionally 'peak_amplitude_values'
        - peak_position: optional, used for labeling the title
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
                        self.axes.scatter(t[idx], val, color='#FF3877', zorder=5)
                        self.axes.annotate(f"{val:.2f}", (t[idx], val), textcoords="offset points",
                                           xytext=(0, 10), ha='center', fontsize=9, color='#FF3877')
            else:
                # Assume single value
                try:
                    idx = int(peak_indices)
                    val = float(peak_values)
                    if 0 <= idx < len(profile):
                        self.axes.scatter(t[idx], val, color='#FF3877', zorder=5)
                        self.axes.annotate(f"{val:.2f}", (t[idx], val), textcoords="offset points",
                                           xytext=(0, 10), ha='center', fontsize=9, color='#FF3877')
                except Exception:
                    pass

        # Axis labeling and formatting
        self.axes.set_xlabel("Time (seconds)")
        self.axes.set_ylabel("Current (nA)")
        title = "I-T Profile"
        if peak_position is not None:
            title += f" at Peak Position {peak_position}"
        self.axes.set_title(title, fontweight="bold")
        self.axes.grid(False)
        self.axes.legend()

        max_t = t[-1]
        tick_interval = 5  # seconds
        ticks = np.arange(0, max_t + tick_interval, tick_interval)
        self.axes.set_xticks(ticks)

        #self.fig.tight_layout()
        # Render to canvas
        self.draw()

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

    def show_decay_exponential_fitting(self, group_analysis, replicate_time_point=0):
        from PyQt5.QtWidgets import QMessageBox
        # Show loading dialog
        progress = QProgressDialog("Processing data, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()  # Ensure dialog appears

        settings = QSettings("HashemiLab", "NeuroStemVolt")
        freq = settings.value("acquisition_frequency", 10, type=int)

        try:
            result = group_analysis.exponential_fitting_replicated(replicate_time_point)
        except ValueError as e:
            QMessageBox.warning(
                self,
                "Dimension Mismatch",
                f"Error: {str(e)}\n\nPlease ensure all replicates have the same number of time points."
            )
            self.axes.clear()
            self.axes.set_title("Dimension mismatch error")
            self.draw()
            progress.close()
            return
        if result is None:
            self.axes.clear()
            self.axes.set_title("No data to fit")
            self.draw()
            progress.close()
            return
        
        from scipy.stats import t
        import numpy as np

        # Get fit and aligned ITs from group_analysis
        # result = group_analysis.exponential_fitting_replicated(replicate_time_point)
        time_all, cropped_ITs, _, t_half, fit_vals, fit_errs, min_peak = result

        A_fit, k_fit, C_fit = fit_vals
        A_err, k_err, C_err = fit_errs
        tau_fit = 1 / k_fit if k_fit != 0 else np.nan

        n_exps, n_post = cropped_ITs.shape
        t_rel = np.arange(n_post) / freq        # seconds

        mean_IT = np.nanmean(cropped_ITs, axis=0)
        std_IT  = np.nanstd (cropped_ITs, axis=0)
        # do the fit in sample‐point space, then map to seconds:
        t_fit_pts = np.linspace(0, n_post-1, 500)          # in points
        y_fit     = (A_fit - C_fit) * np.exp(-k_fit * t_fit_pts) + C_fit
        t_fit_rel = t_fit_pts / freq                       # now in seconds

        # 95% CI of the fit via Jacobian
        dof  = max(0, len(time_all) - 3)
        tval = t.ppf(0.975, dof)
        J        = np.empty((len(t_fit_rel), 3))
        J[:, 0] = np.exp(-t_fit_rel * k_fit)                            
        J[:, 1] = -(A_fit - C_fit) * t_fit_rel * np.exp(-t_fit_rel * k_fit) 
        J[:, 2] = 1 - np.exp(-t_fit_rel * k_fit)                        
        pcov = np.diag([A_err**2, k_err**2, C_err**2])
        ci = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * tval
        lower_ci = y_fit - ci
        upper_ci = y_fit + ci

        # Plot on the embedded axes
        self.axes.clear()

        # a) each replicate in light gray
        for row in cropped_ITs:
            self.axes.plot(t_rel, row, color='gray', alpha=0.3, lw=1, label='_nolegend_')

        # b) individual data points used for fitting
        ITs_flattened = cropped_ITs.flatten()
        t_data = (np.array(time_all) - min_peak) / freq
        self.axes.scatter(t_data, ITs_flattened, color='black', s=16, alpha=0.7, label='Data points')

        # d) fitted exponential curve
        self.axes.plot(t_fit_rel, y_fit, color='C1', lw=2, label='Exp fit')

        # e) 95% CI around the fit
        self.axes.fill_between(t_fit_rel, lower_ci, upper_ci, color='C1', alpha=0.3, label='95% CI')

        # f) half-life marker
        t_half_s = t_half / freq
        self.axes.axvline(t_half_s, color='magenta', ls='--',label=f't½ ≈ {t_half_s:.2f} s')

        # 7) labels & styling
        self.axes.set_xlabel('Time (seconds)', fontsize=12)
        self.axes.set_ylabel('Current (nA)', fontsize=12)
        self.axes.set_title('Post-peak IT decays & exponential fit', fontsize=14)
        self.axes.legend(frameon=False)
        self.axes.grid(False)
        self.fig.tight_layout()

        max_t = t_rel[-1]  # since t_rel is in seconds

        tick_interval = 5  # seconds
        ticks = np.arange(0, max_t + tick_interval, tick_interval)
        self.axes.set_xticks(ticks)

        self.draw()
        progress.close()

    def show_tau_param_over_time(self, group_analysis):
        """
        Plots the exponential decay parameter tau over replicate time points on the embedded canvas.
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

        tau_list, tau_err_list = group_analysis.get_tau_over_time()
        n_files = group_analysis.get_experiments()[0].get_file_count()
        time_points = np.linspace(
            0,
            group_analysis.get_experiments()[0].get_time_between_files() * (n_files - 1),
            n_files
        )

        self.axes.clear()
        self.axes.errorbar(time_points, tau_list, yerr=tau_err_list, fmt='o-', capsize=4, color='C1', label='Tau (decay constant)')
        self.axes.set_xlabel("Time (minutes)")
        self.axes.set_ylabel("Tau (decay constant)")
        self.axes.set_title("Exponential Decay Tau Over Time Points")
        self.axes.grid(False)
        self.axes.legend()
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
        self.axes.set_xticks(np.arange(0, max(time_points) + 1, 10))
        self.fig.tight_layout()
        self.draw()
        progress.close()