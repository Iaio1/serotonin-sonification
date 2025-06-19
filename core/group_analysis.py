from spheroid_experiment import SpheroidExperiment
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
        import matplotlib.pyplot as plt
        first_ITs = self.non_normalized_first_ITs()
        # Number of replicates (rows in mean_ITs)
        n_files = first_ITs.shape[0]
        # Time points (columns in mean_ITs)
        time_points = np.arange(first_ITs.shape[1])
        for i in range(n_files):
            plt.plot(time_points, first_ITs[i,:], label=f"Replicate {i+1}", color="blue", alpha=0.7)
        plt.title("Unprocessed First ITs of Replicates")
        plt.legend(fontsize=10, loc="upper right")
        plt.grid(False)
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

    group_analysis.plot_mean_ITs()
    group_analysis.plot_unprocessed_first_ITs()
    group_analysis.plot_mean_amplitudes_over_time()
    group_analysis.plot_all_amplitudes_over_time()

