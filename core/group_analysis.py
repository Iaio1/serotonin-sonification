from spheroid_experiment import SpheroidExperiment
from processing.exponentialdecay import exp_decay
from processing.normalize import Normalize
import os
import numpy as np

class GroupAnalysis:
    """
    Usage:
        This class is used to manage and analyze multiple SpheroidExperiment instances.
        It allows adding experiments, retrieving single experiments, and (currently) plotting amplitudes over time.
    """
    def __init__(self, experiments=None):
        if experiments is None:
            self.experiments = []
        elif isinstance(experiments, SpheroidExperiment):
            self.experiments = [experiments]
        elif isinstance(experiments, list):
            self.experiments = experiments
        else:
            raise ValueError("experiments must be None, a SpheroidExperiment, or a list of SpheroidExperiment objects.")
        
    def add_experiment(self, *experiments):
        """
        Add one or more SpheroidExperiment instances to the group analysis.
        :param experiments: One or more SpheroidExperiment instances to be added.
        """
        for experiment in experiments:
            if isinstance(experiment, SpheroidExperiment):
                self.experiments.append(experiment)
            elif isinstance(experiment, list):
                for exp in experiment:
                    if isinstance(exp, SpheroidExperiment):
                        self.experiments.append(exp)
                    else:
                        raise ValueError("All elements in the list must be SpheroidExperiment instances.")
            else:
                raise ValueError("Arguments must be SpheroidExperiment instances or lists of them.")

    def get_single_experiments(self, index: int):
        """
        Get the list of SpheroidExperiments in the group analysis.
        :return: List of SpheroidExperiment instances.
        """
        return self.experiments[index]
    def get_experiments(self):
        """
        Get the list of SpheroidExperiments in the group analysis.
        :return: List of SpheroidExperiment instances.
        """
        return self.experiments
    
    def non_normalized_first_ITs(self):
        """
        This method gets the ITs of the first stimulation per experiment (per replicate)
        The ITs are obtained from the original data, completely unprocessed.
        returns: First ITs over replicates as a matrix
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None
        
        # Assume all experiments have the same number of files/timepoints
        file_count = self.experiments[0].get_file_count()
        n_timepoints = self.experiments[0].get_file_time_points()

        ITs = np.empty((n_experiments, n_timepoints), dtype=float)

        for i, experiment in enumerate(self.experiments):
            first_file = experiment.get_spheroid_file(0)
            IT_individual = first_file.get_original_data_IT()
            ITs[i, :] = IT_individual
        
        return ITs

    def average_IT_over_replicates(self):
        """
        This method gets the IT profiles of all experiments of all files
        and averages them over the replicates, precisely over the replicate time points.
        Averages IT every 10 mins for example.
        :return: Averages of IT profiles over replicates.
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None

        # Assume all experiments have the same number of files/timepoints
        file_count = self.experiments[0].get_file_count()
        n_timepoints = self.experiments[0].get_file_time_points()

        all_ITs = np.empty((file_count, n_timepoints, n_experiments), dtype=float) 
        # 16 x 600 x 4 (n_experiments x n_timepoints x n_replicates)
        # Then do the average over the replicates
        for i, experiment in enumerate(self.experiments):
            for j, spheroid_file in enumerate(experiment.files):
                #print(spheroid_file.get_filepath())
                IT_individual = spheroid_file.get_processed_data_IT()
                all_ITs[j, :, i] = IT_individual
        # Average over the third dimension (replicates)
        mean_ITs = np.nanmean(all_ITs, axis=2)
        print(np.shape(mean_ITs))
        
        return mean_ITs
    
    def amplitudes_first_stim(self):
        """
        Get the unnormalised amplitudes of all experiments over time.
        :return: List of lists containing amplitudes for each experiment.
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None

        # Assume all experiments have the same number of files/timepoints
        n_timepoints = self.experiments[0].get_file_count()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
        time_points = np.linspace(
            0,
            self.experiments[0].get_time_between_files() * (n_timepoints - 1),
            n_timepoints
        )
        amplitudes = []
        for i, experiment in enumerate(self.experiments):
            first_stim_spheroid = experiment.get_spheroid_file(0) #Getting the first stimulation
            # print(first_stim_spheroid.get_filepath()) 
            # Check for Normalize in the processor list
            has_norm = any(isinstance(p, Normalize) for p in experiment.processors)
            if has_norm:
                raise RuntimeError("Experiment must not include Normalize()")
            metadata = first_stim_spheroid.get_metadata()
            peak_amplitude_values = metadata['peak_amplitude_values']
            amplitudes.append(peak_amplitude_values.tolist())

        return amplitudes

    def amplitudes_over_time_single_experiment(self, experiment_index=0):
        """
        Get the amplitudes of all experiments over time.
        :return: List of lists containing amplitudes for each experiment.
        """
        experiment = group_analysis.get_single_experiments(experiment_index)
        files_before_treatment = experiment.get_number_of_files_before_treatment()
        amplitudes = []
        time_points = np.linspace(0, experiment.get_time_between_files() * (experiment.get_file_count() - 1), experiment.get_file_count())
        for spheroid_file in experiment.files:
            # Current metadata:
            # dict_keys(['peak_position', 'stim_start', 'stim_duration', 'stim_frequency', 
            # 'background_subtraction_region', 'baseline', 'experiment_first_peak', 'peak_amplitude_positions', 
            # 'peak_amplitude_values', 'exponential fitting parameters']) dict_values([257, 5.0, 2.0, 20, (0, 10), 
            # array([0.02988434]), np.float64(0.3822367999525832), array([74]), array([1.04633804]), 
            # {'A': np.float64(1.5705880537206165), 'tau': np.float64(119.00028560994886), 
            # 'C': np.float64(0.22708457061563378), 't_half': np.float64(82.48471245636428)}])
            metadata = spheroid_file.get_metadata()
            peak_amplitude_values = metadata['peak_amplitude_values']
            amplitudes.append(peak_amplitude_values.tolist())
        print(f"Time points: {time_points}")
        print(f"Amplitudes: {amplitudes}")
        return time_points, amplitudes, files_before_treatment
    
    def amplitudes_over_time_all_experiments(self):
        """
        Collects amplitudes for all experiments, aligns them by time point,
        and computes the average amplitude at each time point.
        Returns:
            time_points: 1D array of time points
            mean_amplitudes: 1D array of mean amplitudes at each time point
            all_amplitudes: 2D array [experiment, time_point]
        """
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None

        # Assume all experiments have the same number of files/timepoints
        n_timepoints = self.experiments[0].get_file_count()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
        time_points = np.linspace(
            0,
            self.experiments[0].get_time_between_files() * (n_timepoints - 1),
            n_timepoints
        )

        all_amplitudes = np.full((n_experiments, n_timepoints), np.nan, dtype=float)

        for i, experiment in enumerate(self.experiments):
            for j, spheroid_file in enumerate(experiment.files):
                metadata = spheroid_file.get_metadata()
                peak_amplitude_values = metadata['peak_amplitude_values']
                # If peak_amplitude_values is None, empty, or zero, keep as zero
                if peak_amplitude_values is None or (isinstance(peak_amplitude_values, np.ndarray) and peak_amplitude_values.size == 0):
                    all_amplitudes[i, j] = 0.0
                elif isinstance(peak_amplitude_values, np.ndarray):
                    val = float(peak_amplitude_values[0]) if peak_amplitude_values.size == 1 else float(peak_amplitude_values)
                    all_amplitudes[i, j] = val if val != 0 else 0.0
                else:
                    val = float(peak_amplitude_values)
                    all_amplitudes[i, j] = val if val != 0 else 0.0
        mean_amplitudes = np.nanmean(all_amplitudes, axis=0)
        return time_points, mean_amplitudes, all_amplitudes, files_before_treatment
    
    def exponential_fitting_replicated(self, replicate_time_point = 0, global_peak_amplitude_position=None):
        from scipy.optimize import curve_fit
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None
        # Assume all experiments have the same number of files/timepoints
        n_timepoints = self.experiments[0].get_file_time_points()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
        
        all_ITs = np.empty((n_experiments, n_timepoints))
        peak_amplitude_positions = []

        actual_index = replicate_time_point + files_before_treatment
        for i, experiment in enumerate(self.experiments):
            file = experiment.get_spheroid_file(actual_index)
            IT_individual = file.get_processed_data_IT()
            metadata = file.get_metadata()
            peak_amplitude_positions.append((metadata["peak_amplitude_positions"]))
            all_ITs[i, :] = IT_individual

        # Turning peak_amplitude_positions into a list of integers
        peaks = [p.item() for p in peak_amplitude_positions]
        min_peak = np.min(peaks)
        max_peak = np.max(peaks)
        pre_allocated_ITs_array = np.empty((n_experiments,n_timepoints-min_peak))        

        # Fill the pre-allocated array with the cropped ITs, starting from the peak position
        for i, (row, peak) in enumerate(zip(all_ITs, peaks)):
            # Now the array will have from the min peak position to the end of the time points
            # Now this has aligned the peaks, so the first time point is the peak position for all ITs
            cropped = row[peak:]
            length = cropped.shape[0]
            pre_allocated_ITs_array[i, :length] = cropped

        # Crop the ITs from the end to match data sizes
        cropped_ITs = pre_allocated_ITs_array[:, :n_timepoints-max_peak-min_peak]
        print("Cropped ITs:",np.shape(cropped_ITs))
        ITs_flattened = cropped_ITs.flatten()
        print(np.shape(ITs_flattened))
        n_cropped_timepoints = np.shape(cropped_ITs)

        A = np.arange(min_peak, n_timepoints-max_peak)
        print(np.shape(A))
        time_all = np.tile(A, n_experiments)  # Repeat time point
        print(np.shape(time_all))

        #print("Len ITs Flattened:", len(ITs_flattened))
        #print("ITs_Flattened", np.shape(ITs_flattened))
        #print("Time All", np.shape(time_all))

        # Improved initial guess for parameters
        # A: amplitude (difference between max and min of cropped ITs)
        # tau: decay constant (guess as 1/3 of the time range)
        # C: baseline (last value of the mean trace)
        mean_trace = np.mean(cropped_ITs, axis=0)
        A0 = float(np.max(mean_trace) - np.min(mean_trace))
        tau0 = (n_timepoints - min_peak) / 3.0
        C0 = float(mean_trace[-1])
        p0 = [A0, tau0, C0]

        print(f"Initial guess: A={A0:.2f}, tau={tau0:.2f}, C={C0:.2f}")

        # Fit
        popt, pcov = curve_fit(exp_decay, time_all, ITs_flattened, p0=p0)

        # Extract parameter estimates and standard errors
        A_fit, tau_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))  # Approximate symmetric 1-sigma CI

        print(f"Fit results:")
        print(f"A   = {A_fit:.2f} ± {perr[0]:.2f}")
        print(f"tau = {tau_fit:.2f} ± {perr[1]:.2f}")
        print(f"C   = {C_fit:.2f} ± {perr[2]:.2f}")

        t_half = np.log(2) * tau_fit

        # Pre-allocated_ITs_array is the matrix with all data properly aligned on their peaks
        return time_all, cropped_ITs, t_half, popt, pcov,  A_fit, tau_fit, C_fit

    def exponential_fitting_replicated_legacy(self, replicate_time_point = 0, global_peak_amplitude_position=None):
        """
        This function implements an exponential fitting curve over the replicates 
        at specific replicated time points. (e.g. 10min treatment file), this function
        does it considering the data from the max peak amplitude position. Meaning it does not align the peaks
        Input:
        - replicate_time_point is the index to gather the data from. 
        (e.g. if files are collected every 10 min, and there are three files pre-treatment
          index -3 will be the first file before treatment, index 3 will be the first file after treatment)
        returns 
        """

        from scipy.optimize import curve_fit
        
        n_experiments = len(self.experiments)
        if n_experiments == 0:
            return None, None, None, None

        # Assume all experiments have the same number of files/timepoints
        n_timepoints = self.experiments[0].get_file_time_points()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment() # This will be zero if no files before treatment
        
        all_ITs = np.empty((n_experiments, n_timepoints))
        peak_amplitude_positions = []

        actual_index = replicate_time_point + files_before_treatment
        for i, experiment in enumerate(self.experiments):
            file = experiment.get_spheroid_file(actual_index)
            IT_individual = file.get_processed_data_IT()
            metadata = file.get_metadata()
            peak_amplitude_positions.append(metadata["peak_amplitude_positions"])
            all_ITs[i, :] = IT_individual

        print(peak_amplitude_positions)
        latest_peak_amplitude_positions = np.max(peak_amplitude_positions)
        if global_peak_amplitude_position is None:
            global_peak_amplitude_position = latest_peak_amplitude_positions
        else:
            global_peak_amplitude_position = int(global_peak_amplitude_position)
        print(global_peak_amplitude_position)
        cropped_ITs = all_ITs[:, global_peak_amplitude_position:]

        ITs_flattened = cropped_ITs.flatten()
        n_cropped_timepoints = np.shape(cropped_ITs)

        A = np.arange(global_peak_amplitude_position, n_timepoints)
        time_all = np.tile(A, n_experiments)  # Repeat time point

        #print("Len ITs Flattened:", len(ITs_flattened))
        #print("ITs_Flattened", np.shape(ITs_flattened))
        #print("Time All", np.shape(time_all))

        # Improved initial guess for parameters
        # A: amplitude (difference between max and min of cropped ITs)
        # tau: decay constant (guess as 1/3 of the time range)
        # C: baseline (last value of the mean trace)
        mean_trace = np.mean(cropped_ITs, axis=0)
        A0 = float(np.max(mean_trace) - np.min(mean_trace))
        tau0 = (n_timepoints - global_peak_amplitude_position) / 3.0
        C0 = float(mean_trace[-1])
        p0 = [A0, tau0, C0]

        print(f"Initial guess: A={A0:.2f}, tau={tau0:.2f}, C={C0:.2f}")

        # Fit
        popt, pcov = curve_fit(exp_decay, time_all, ITs_flattened, p0=p0)

        # Extract parameter estimates and standard errors
        A_fit, tau_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))  # Approximate symmetric 1-sigma CI

        print(f"Fit results:")
        print(f"A   = {A_fit:.2f} ± {perr[0]:.2f}")
        print(f"tau = {tau_fit:.2f} ± {perr[1]:.2f}")
        print(f"C   = {C_fit:.2f} ± {perr[2]:.2f}")

        t_half = np.log(2) * tau_fit

        return time_all, ITs_flattened, t_half, popt, pcov,  A_fit, tau_fit, C_fit
    
    def plot_exponential_fit_aligned(self, replicate_time_point=0):
        """
        Plot each post-peak IT trace, the mean decay, the exponential fit, 
        its 95% CI, and mark the half-life, all on a common “time since peak” axis.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import t
        import numpy as np

        # 1) run your fit and get the aligned, cropped IT matrix
        time_all, cropped_ITs, t_half, popt, pcov, A_fit, tau_fit, C_fit = \
            self.exponential_fitting_replicated(replicate_time_point)
        # cropped_ITs: shape (n_experiments, n_post_peak_points)

        # 2) build a “time since peak” axis
        n_exps, n_post = cropped_ITs.shape
        t_rel = np.arange(n_post)

        # 3) compute mean ± SD across replicates
        mean_IT = np.nanmean(cropped_ITs, axis=0)
        std_IT  = np.nanstd (cropped_ITs, axis=0)

        # 4) smooth fit curve on that same relative axis
        t_fit_rel = np.linspace(0, n_post-1, 500)
        y_fit     = A_fit * np.exp(-t_fit_rel / tau_fit) + C_fit

        # 5) 95% CI of the fit via Jacobian
        dof  = max(0, len(time_all) - len(popt))
        tval = t.ppf(0.975, dof)
        # Jacobian of f(t) = A e^{-t/τ} + C wrt [A, τ, C]
        J        = np.empty((len(t_fit_rel), 3))
        J[:, 0]  = np.exp(-t_fit_rel / tau_fit)
        J[:, 1]  = A_fit * (t_fit_rel / tau_fit**2) * np.exp(-t_fit_rel / tau_fit)
        J[:, 2]  = 1
        ci       = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * tval
        lower_ci = y_fit - ci
        upper_ci = y_fit + ci

        # 6) start plotting
        plt.figure(figsize=(10,6))

        # a) each replicate in light gray
        for row in cropped_ITs:
            plt.plot(t_rel, row, color='gray', alpha=0.3, lw=1, label='_nolegend_')

        # b) mean ± 1 SD ribbon
        plt.fill_between(t_rel,
                        mean_IT - std_IT,
                        mean_IT + std_IT,
                        color='C0', alpha=0.2,
                        label='Mean ± 1 SD')

        # c) mean trace
        plt.plot(t_rel, mean_IT, color='C0', lw=2, label='Mean trace')

        # d) fitted exponential curve
        plt.plot(t_fit_rel, y_fit, color='C1', lw=2, label='Exp fit')

        # e) 95% CI around the fit
        plt.fill_between(t_fit_rel,
                        lower_ci,
                        upper_ci,
                        color='C1', alpha=0.3,
                        label='95% CI')

        # f) half-life marker
        plt.axvline(t_half, color='magenta', ls='--',
                    label=f't½ ≈ {t_half:.1f} pts')

        # 7) labels & styling
        plt.xlabel('Time since peak (points)', fontsize=12)
        plt.ylabel('Current (nA)', fontsize=12)
        plt.title('Post-peak IT decays & exponential fit', fontsize=14)
        plt.legend(frameon=False)
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_exponential_fit_with_CI_legacy(self, replicate_time_point=0, global_peak_position=None):
        import matplotlib.pyplot as plt
        from scipy.stats import t
        import numpy as np

        # Run the fitting function to get values and data
        time_all, ITs_flattened, t_half, popt, pcov, A_fit, tau_fit, C_fit = self.exponential_fitting_replicated(replicate_time_point=replicate_time_point)
        A_fit, tau_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))  # 1-sigma CI

        # Retrieve full ITs and metadata again for plotting individual traces
        n_experiments = len(self.experiments)
        file_duration = self.experiments[0].get_file_length()
        n_timepoints = self.experiments[0].get_file_time_points()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment()
        actual_index = replicate_time_point + files_before_treatment

        all_ITs = np.empty((n_experiments, n_timepoints))
        peak_positions = []

        for i, experiment in enumerate(self.experiments):
            file = experiment.get_spheroid_file(actual_index)
            IT_individual = file.get_processed_data_IT()
            metadata = file.get_metadata()
            peak_pos = metadata.get("peak_amplitude_positions")
            peak_positions.append(peak_pos)
            all_ITs[i, :] = IT_individual

        global_peak_position = int(np.max(peak_positions))
        if global_peak_position is None:
            global_peak_position = int(np.max(peak_positions))
        else:
            global_peak_position = global_peak_position
        # Time array for full profile
        full_time = np.arange(n_timepoints)

        # Time for fitting and prediction
        t_fit = np.linspace(global_peak_position, n_timepoints - 1, 500)
        y_fit = A_fit * np.exp(-t_fit / tau_fit) + C_fit

        # 95% CI using Jacobian
        dof = max(0, len(time_all) - len(popt))
        t_val = t.ppf(0.975, dof)

        J = np.empty((len(t_fit), 3))
        J[:, 0] = np.exp(-t_fit / tau_fit)
        J[:, 1] = A_fit * t_fit / tau_fit**2 * np.exp(-t_fit / tau_fit)
        J[:, 2] = 1

        ci = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * t_val
        y_lower = y_fit - ci
        y_upper = y_fit + ci

        # Plot
        plt.figure(figsize=(10, 6))

        # Plot each replicate I-T trace
        for i in range(n_experiments):
            plt.plot(full_time, all_ITs[i, :], alpha=0.4, linewidth=1, label=f"Replicate {i+1}" if i == 0 else None)

        # Plot fit and CI
        plt.plot(t_fit, y_fit, label='Exponential Fit', color='red', linewidth=2)
        plt.fill_between(t_fit, y_lower, y_upper, color='red', alpha=0.3, label='95% CI')

        # Mark peak and t_half
        plt.axvline(global_peak_position, color='orange', linestyle='--', label=f"Peak @ {global_peak_position}")
        plt.axvline(global_peak_position + int(t_half), color='purple', linestyle=':', label=f"t_half ≈ {t_half:.2f}")

        # Plot peak positions of each replicate
        for i, peak_pos in enumerate(peak_positions):
            if isinstance(peak_pos, (np.ndarray, list)) and len(peak_pos) > 0:
                pos = int(np.max(peak_pos))
            else:
                pos = int(peak_pos)
            plt.scatter(pos, all_ITs[i, pos], color='red', marker='x', s=10, label='Peak Position' if i == 0 else None, zorder=5)

        plt.xlabel("Time Points (seconds)")
        print("File duration:", file_duration)
        # Set ticks at every 10 seconds => every 100 time points
        tick_locs = np.arange(0, 601, 100)       # 0, 100, 200, ..., 600
        tick_labels = [str(int(x / 10)) for x in tick_locs]  # convert to seconds: 0, 10, ..., 60
        plt.xticks(tick_locs, tick_labels, fontsize=10)
        plt.ylabel("Current (nA)")
        plt.title("Replicate I-T Profiles with Exponential Fit & CI")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_amplitudes_over_time_single_experiment(self, experiment_index=0):
        """
        Plot the amplitudes of a single experiment over time, with treatment point clearly marked.
        """
        import matplotlib.pyplot as plt

        time_points, amplitudes, files_before_treatment = self.amplitudes_over_time_single_experiment(experiment_index=experiment_index)

        # Compute treatment time in seconds
        experiment = self.experiments[experiment_index]
        treatment_time = files_before_treatment * experiment.get_time_between_files() # his will be zero if no files before treatment

        # Split data
        before_treatment_x = time_points[:files_before_treatment]
        before_treatment_y = amplitudes[:files_before_treatment]

        after_treatment_x = time_points[files_before_treatment:]
        after_treatment_y = amplitudes[files_before_treatment:]

        plt.figure(figsize=(10, 6))

        # Plot before and after separately
        plt.plot(before_treatment_x, before_treatment_y, label='Pre-Treatment', color='blue')
        plt.plot(after_treatment_x, after_treatment_y, label='Post-Treatment', color='green')

        # Vertical line for treatment start
        plt.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')

        # Optional scatter for emphasis
        plt.scatter(time_points, amplitudes, color='black', s=20, alpha=0.6)

        # Scientific styling
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.xticks(np.arange(0, max(time_points) + 1, 10), fontsize=10)
        plt.title('Amplitude Over Time Relative to Treatment', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_mean_amplitudes_over_time(self):
        """
        Plot the mean amplitudes over time across all experiments,
        with the standard deviation as a shaded area.
        """
        import matplotlib.pyplot as plt

        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = self.amplitudes_over_time_all_experiments()
        all_amplitudes = np.array(all_amplitudes, dtype=float)
        std_amplitudes = np.nanstd(all_amplitudes, axis=0)

        treatment_time = files_before_treatment * (time_points[1] - time_points[0])

        plt.figure(figsize=(10, 6))
        plt.plot(time_points, mean_amplitudes, label='Mean Amplitude', color='purple')
        plt.fill_between(time_points, mean_amplitudes - std_amplitudes, mean_amplitudes + std_amplitudes,
                         color='purple', alpha=0.2, label='SD')
        if files_before_treatment > 0:
            plt.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Amplitude')
        plt.title('Mean Amplitude Over Time (All Experiments)')
        plt.legend()
        plt.xticks(np.arange(0, max(time_points) + 1, 10), fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def plot_all_amplitudes_over_time(self):
        """
        Plot all amplitudes over time for each experiment as separate lines.
        """
        import matplotlib.pyplot as plt
        
        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = self.amplitudes_over_time_all_experiments()
        all_amplitudes = np.array(all_amplitudes, dtype=float)
        treatment_time = files_before_treatment * (time_points[1] - time_points[0])

        plt.figure(figsize=(10, 6))
        for i, amplitudes in enumerate(all_amplitudes):
            plt.plot(time_points, amplitudes, label=f'Experiment {i+1}', alpha=0.7)
        if files_before_treatment > 0:
            plt.axvline(x=treatment_time, color='red', linestyle='--', label='Treatment Start')
        plt.xlabel('Time (min)')
        plt.ylabel('Amplitude')
        plt.title('Amplitudes Over Time (All Experiments)')
        plt.legend()
        plt.xticks(np.arange(0, max(time_points) + 1, 10), fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_first_stim_amplitudes(self):
        """
        Plot the unnormalized amplitudes of the first stimulation for all experiments (replicates).
        Each replicate is shown as a bar or point, with optional mean and std.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        amplitudes = self.amplitudes_first_stim()  # List of lists (one per replicate)
        if amplitudes is None or len(amplitudes) == 0:
            print("No amplitudes to plot.")
            return

        # Flatten in case each amplitude is a list (e.g., [ [1.2], [1.1], ... ])
        flat_amps = [a[0] if isinstance(a, (list, np.ndarray)) and len(a) > 0 else np.nan for a in amplitudes]
        n_replicates = len(flat_amps)
        x = np.arange(1, n_replicates + 1)

        plt.figure(figsize=(8, 6))
        plt.bar(x, flat_amps, color='skyblue', edgecolor='k', alpha=0.8, label='First Stimulation Amplitude')
        plt.scatter(x, flat_amps, color='blue', zorder=5)

        # Mean and std
        mean_amp = np.nanmean(flat_amps)
        std_amp = np.nanstd(flat_amps)
        plt.axhline(mean_amp, color='red', linestyle='--', label=f'Mean = {mean_amp:.2f}')
        plt.fill_between([0, n_replicates+1], mean_amp-std_amp, mean_amp+std_amp, color='red', alpha=0.15, label='±1 SD')

        plt.xlabel("Replicate", fontsize=14)
        plt.ylabel("Amplitude (nA)", fontsize=14)
        plt.title("First Stimulation Amplitudes Across Replicates", fontsize=16)
        plt.xticks(x, [f"Rep {i}" for i in x])
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_mean_ITs(self):
        """
        Plots the mean IT profiles over replicates, highlighting files before treatment and marking the first file after treatment.
        
        Args:
            mean_ITs (np.ndarray): Matrix of mean IT profiles (rows = files, columns = time points).
            files_before_treatment (int): Number of files before treatment.
            title_suffix (str): Optional suffix for the plot title.
        """
        import matplotlib.pyplot as plt

        mean_ITs = self.average_IT_over_replicates()
        files_before_treatment = self.experiments[0].get_number_of_files_before_treatment()  # Assuming all experiments have the same number of files before treatment

        # Number of files (rows in mean_ITs)
        n_files = mean_ITs.shape[0]

        # Time points (columns in mean_ITs)
        time_points = np.arange(mean_ITs.shape[1])

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot files before treatment
        for i in range(files_before_treatment):
            plt.plot(time_points, mean_ITs[i, :], label=f"File {i+1} (Before Treatment)", color="blue", alpha=0.7)

        # Plot files after treatment
        for i in range(files_before_treatment, n_files):
            plt.plot(time_points, mean_ITs[i, :], label=f"File {i+1} (After Treatment)", color="green", alpha=0.7)

        # Highlight the first file after treatment
        if files_before_treatment < n_files:
            plt.plot(time_points, mean_ITs[files_before_treatment, :], label="First File After Treatment", color="red", linewidth=2)

        # Add labels, title, and legend
        plt.xlabel("Time Points", fontsize=14)
        plt.ylabel("Mean IT (nA)", fontsize=14)
        plt.title(f"Mean IT Profiles Over Replicates", fontsize=16)
        plt.legend(fontsize=10, loc="upper right")
        plt.grid(False)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_unprocessed_first_ITs(self):
        """
        Plots the unprocessed first ITs of replicates.
        The x-axis represents the amplitude of the signal, and the y-axis represents time in seconds.
        """
        import matplotlib.pyplot as plt

        # Get the unprocessed first ITs
        first_ITs = self.non_normalized_first_ITs()

        # Number of replicates (rows in first_ITs)
        n_replicates = first_ITs.shape[0]

        # Time points (y-axis: seconds)
        time_points = np.linspace(0, self.experiments[0].get_file_length(), first_ITs.shape[1])

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot each replicate
        for i in range(n_replicates):
            plt.plot(time_points,first_ITs[i, :], label=f"Replicate {i+1}", alpha=0.7)

        # Add labels, title, and legend
        plt.ylabel("Amplitude (nA)", fontsize=14)
        plt.xlabel("Time (seconds)", fontsize=14)
        plt.title("Unprocessed First ITs of Replicates", fontsize=16)
        plt.legend(fontsize=10, loc="upper right")
        plt.grid(False)

        # Show the plot
        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    import time
    start_time = time.time()
    # Example usage
    folder_first_experiment = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241111_batch1_n1_Sert"
    #folder_first_experiment = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241111_batch1_n1_Sert"
    filepaths_first_experiment = [os.path.join(folder_first_experiment, f) for f in os.listdir(folder_first_experiment) if f.endswith('.txt')]
    experiment_one = SpheroidExperiment(filepaths_first_experiment, treatment="Sertraline")
    experiment_one.run()

    folder_second_experiment = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241115_batch1_n2_Sert"
    #folder_second_experiment = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241115_batch1_n2_Sert"
    filepaths_second_experiment = [os.path.join(folder_second_experiment, f) for f in os.listdir(folder_second_experiment) if f.endswith('.txt')]   
    experiment_two = SpheroidExperiment(filepaths_second_experiment, treatment="Sertraline")
    experiment_two.run()

    folder_third_experiment = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241116_batch1_n3_Sert"
    #folder_third_experiment = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241116_batch1_n3_Sert"
    filepaths_third_experiment = [os.path.join(folder_third_experiment, f) for f in os.listdir(folder_third_experiment) if f.endswith('.txt')]
    experiment_three = SpheroidExperiment(filepaths_third_experiment, treatment="Sertraline")
    experiment_three.run()

    folder_fourth_experiment = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241128_batch2_n4_Sert"
    #folder_fourth_experiment = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241128_batch2_n4_Sert"
    filepaths_fourth_experiment = [os.path.join(folder_fourth_experiment, f) for f in os.listdir(folder_fourth_experiment) if f.endswith('.txt')]
    experiment_four = SpheroidExperiment(filepaths_fourth_experiment, treatment="Sertraline")
    experiment_four.run()

    group_analysis = GroupAnalysis()
    group_analysis.add_experiment(experiment_one, experiment_two, experiment_three, experiment_four)

    time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    #group_analysis.plot_mean_ITs()
    #group_analysis.exponential_fitting_replicated()
    #group_analysis.plot_exponential_fit_aligned(replicate_time_point=0)
    #group_analysis.plot_unprocessed_first_ITs()
    #group_analysis.plot_mean_amplitudes_over_time()
    #group_analysis.plot_all_amplitudes_over_time()
    group_analysis.amplitudes_first_stim()
    group_analysis.plot_first_stim_amplitudes()

