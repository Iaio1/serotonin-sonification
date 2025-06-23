from spheroid_experiment import SpheroidExperiment
from group_analysis import GroupAnalysis
import os
import pandas as pd

class OutputManager:
    @staticmethod
    def save_ITs(group_experiments : GroupAnalysis, output_folder_path):
        # This function takes all experiments (after processing) 
        # and creates output_csv files for all of them
        n_experiments = len(group_experiments.get_experiments())
        if n_experiments == 0:
            return None
        # Initialise the matrix 
        for i, experiment in enumerate(group_experiments.get_experiments()):
            it_matrix = []
            file_names = []
            for j, spheroid_file in enumerate(experiment.files):
                IT_individual = spheroid_file.get_processed_data_IT()
                it_matrix.append(IT_individual)
                file_name = spheroid_file.get_filepath()
                file_names.append(file_name)
            # Transpose so each column is a file
            df = pd.DataFrame(it_matrix).T
            df.columns = [f"File_{i}" for i in range(len(file_names))]
            df.columns = [file_name.split("/")[-1] for file_name in file_names]
            output_csv = "All_ITs_experiment_n{0}.csv".format(i)
            output_IT_folder = os.path.join(output_folder_path,"replicate_ITs")
            if os.path.isdir(output_IT_folder) == False:
                os.mkdir(output_IT_folder)
            output_path = os.path.join(output_IT_folder, output_csv)
            df.to_csv(output_path, index_label="TimePoint")
    @staticmethod
    def save_original_ITs(group_experiments : GroupAnalysis, output_folder_path):
        # This function takes all experiments (after processing) 
        # and creates output_csv files for all of them
        n_experiments = len(group_experiments.get_experiments())
        if n_experiments == 0:
            return None
        # Initialise the matrix 
        for i, experiment in enumerate(group_experiments.get_experiments()):
            it_matrix = []
            file_names = []
            for j, spheroid_file in enumerate(experiment.files):
                IT_individual = spheroid_file.get_original_data_IT()
                it_matrix.append(IT_individual)
                file_name = spheroid_file.get_filepath()
                file_names.append(file_name)
            # Transpose so each column is a file
            df = pd.DataFrame(it_matrix).T
            df.columns = [f"File_{i}" for i in range(len(file_names))]
            df.columns = [file_name.split("/")[-1] for file_name in file_names]
            output_csv = "Original_ITs_experiment_n{0}.csv".format(i)
            output_IT_folder = os.path.join(output_folder_path,"original_ITs_per_replicate")
            if os.path.isdir(output_IT_folder) == False:
                os.mkdir(output_IT_folder)
            output_path = os.path.join(output_IT_folder, output_csv)
            df.to_csv(output_path, index_label="TimePoint")
    @staticmethod
    def save_peak_amplitudes_metrics(group_experiments : GroupAnalysis, output_folder_path):
        # This function saves the following keys per file in a folder named metadata_files
        # Particularly these are the saves the method can save:
        # # Current metadata:
            # dict_keys(['peak_position', 'stim_start', 'stim_duration', 'stim_frequency', 
            # 'background_subtraction_region', 'baseline', 'experiment_first_peak', 'peak_amplitude_positions', 
            # 'peak_amplitude_values', 'exponential fitting parameters']) dict_values([257, 5.0, 2.0, 20, (0, 10), 
            # array([0.02988434]), np.float64(0.3822367999525832), array([74]), array([1.04633804]), 
            # {'A': np.float64(1.5705880537206165), 'tau': np.float64(119.00028560994886), 
            # 'C': np.float64(0.22708457061563378), 't_half': np.float64(82.48471245636428)}])
        # This method saves the following keys: keys = ['peak_amplitude_positions','peak_amplitude_values']
        keys = ['peak_amplitude_positions','peak_amplitude_values']
        for i, experiment in enumerate(group_experiments.get_experiments()):
            records = []
            for j, spheroid_file in enumerate(experiment.files):
                meta = spheroid_file.get_metadata()
                if keys is None:
                    # Save all keys
                    records.append(meta)
                else:
                    # Save only selected keys
                    records.append({k: meta.get(k, None) for k in keys})
                df = pd.DataFrame(records)
                output_csv = "Files_Amplitudes_experiment_n{0}.csv".format(i)
                output_IT_folder = os.path.join(output_folder_path,"amplitudes_files")
                if os.path.isdir(output_IT_folder) == False:
                    os.mkdir(output_IT_folder)
                output_path = os.path.join(output_IT_folder, output_csv)
                df.to_csv(output_path, index_label="File Number")


if __name__ == "__main__":
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

    group_analysis = GroupAnalysis()
    group_analysis.add_experiment(experiment_one,experiment_two)

    # Save ITs
    output_folder = r"/Users/pabloprieto/Library/CloudStorage/OneDrive-Personal/Documentos/1st_Year_PhD/Projects/NeuroStemVolt/output"
    #OutputManager.save_ITs(group_analysis,output_folder)
    OutputManager.save_original_ITs(group_analysis,output_folder)
    OutputManager.save_peak_amplitudes_metrics(group_analysis,output_folder)


