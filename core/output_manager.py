from spheroid_experiment import SpheroidExperiment
from group_analysis import GroupAnalysis
import os
import pandas as pd

class OutputManager:
    @staticmethod
    def save_all_ITs(group_experiments : GroupAnalysis, output_folder_path):
        # This function takes all experiments (after processing) 
        # and creates output_csv files for all of them
        n_experiments = len(group_experiments.get_experiments())
        if n_experiments == 0:
            return None
        # Initialise the matrix 
        it_matrix = []
        file_names = []
        for i, experiment in enumerate(group_experiments.get_experiments()):
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
    def save_metadata_metrics(group_experiments : GroupAnalysis, output_folder_path):
        pass



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


