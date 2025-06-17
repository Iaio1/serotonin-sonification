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
    def add_experiment(self, experiment: SpheroidExperiment):
        """
        Add a SpheroidExperiment to the group analysis.
        :param experiment: An instance of SpheroidExperiment to be added.
        """
        self.experiments.append(experiment)

    def get_single_experiments(self, index: int):
        """
        Get the list of SpheroidExperiments in the group analysis.
        :return: List of SpheroidExperiment instances.
        """
        return self.experiments[index]
    
    def amplitudes_over_time(self, experiment_index=0):
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
    
    def plot_amplitudes_over_time(self, experiment_index=0):
        """
        Plot the amplitudes of a single experiment over time, with treatment point clearly marked.
        """
        import matplotlib.pyplot as plt

        time_points, amplitudes, files_before_treatment = self.amplitudes_over_time(experiment_index=experiment_index)

        # Compute treatment time in seconds
        experiment = self.experiments[experiment_index]
        treatment_time = files_before_treatment * experiment.get_time_between_files()

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



if __name__ == "__main__":
    # Example usage
    #folder = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/data/241111_batch1_n1_Sert"
    folder = r"C:\Users\pablo\OneDrive\Documentos\1st_Year_PhD\Projects\NeuroStemVolt\data\241111_batch1_n1_Sert"
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]
    experiment = SpheroidExperiment(filepaths, treatment="Sertraline")
    experiment.run()
    group_analysis = GroupAnalysis()
    group_analysis.add_experiment(experiment)
    group_analysis.plot_amplitudes_over_time()
    