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
    def save_all_ITs(group_experiments : GroupAnalysis, output_folder_path):
        # This function takes all experiments (after processing) 
        # and creates output_csv files for all of them
        experiments = group_experiments.get_experiments()
        n_experiments = len(group_experiments.get_experiments())
        n_ITs = group_experiments.get_experiments()[0].get_file_count()
        n_timepoints = group_experiments.get_experiments()[0].get_file_time_points()
        if n_experiments == 0:
            return None
        # Initialise the matrix
        arrays = []
        data = []
         # For each experiment (replicate)
        for exp_idx, experiment in enumerate(experiments):
            rep_name = f"Rep{exp_idx+1}"
            for file_idx, spheroid_file in enumerate(experiment.files):
                file_short = os.path.basename(spheroid_file.get_filepath())
                arrays.append((rep_name, file_short))    
                
        for t in range(n_timepoints):
            row = []
            for exp_idx, experiment in enumerate(experiments):
                for file_idx, spheroid_file in enumerate(experiment.files):
                    IT_individual = spheroid_file.get_processed_data_IT()
                    if t < len(IT_individual):
                        row.append(IT_individual[t])
                    else:
                        row.append(None)
            data.append(row)

        # Create MultiIndex columns
        columns = pd.MultiIndex.from_tuples(arrays, names=["Replicate", "File"])
        df = pd.DataFrame(data, columns=columns)
        df.index.name = "TimePoint"

        # Save to CSV
        output_IT_folder = os.path.join(output_folder_path, "all_replicates_ITs")
        os.makedirs(output_IT_folder, exist_ok=True)
        output_path = os.path.join(output_IT_folder, "All_ITs_all_replicates.csv")
        df.to_csv(output_path)
        print(f"Saved all ITs for all replicates to {output_path}")

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
    @staticmethod
    def save_all_peak_amplitudes(group_experiments : GroupAnalysis, output_folder_path):
        keys = ['peak_amplitude_values']

        experiments = group_experiments.get_experiments()
        n_experiments = len(experiments)
        n_files = experiments[0].get_file_count()
        n_before = experiments[0].get_number_of_files_before_treatment()
        interval = experiments[0].get_time_between_files()  # e.g., 10

        if n_experiments == 0:
            return None
        # Initialise the time axis (first column)
        if n_before > 0:
            time_points = [interval * (i - n_before) for i in range(n_files)]
        else:
            time_points = [i * interval for i in range(n_files)]

        all_amplitudes = []
        for i, experiment in enumerate(group_experiments.get_experiments()):
            records = []
            for j, spheroid_file in enumerate(experiment.files):
                meta = spheroid_file.get_metadata()
                # Save only selected keys
                records.append(meta.get(keys[0], None) if keys else meta)
            all_amplitudes.append(records)
        
        # Build DataFrame
        df = pd.DataFrame(all_amplitudes).T  # shape: (n_files, n_experiments)
        df.columns = [f"Rep{idx+1}" for idx in range(n_experiments)]
        df.insert(0, "Time", time_points)

        # Save to CSV
        output_folder = os.path.join(output_folder_path, "all_replicates_amplitudes")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "All_amplitudes_all_replicates.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved all amplitudes for all replicates to {output_path}")


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
    OutputManager.save_all_ITs(group_analysis,output_folder)
    OutputManager.save_all_peak_amplitudes(group_analysis,output_folder)
    #OutputManager.save_original_ITs(group_analysis,output_folder)
    #OutputManager.save_peak_amplitudes_metrics(group_analysis,output_folder)


