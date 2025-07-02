from PyQt5.QtWidgets import (
    QApplication, QWizard, QComboBox, QLineEdit, QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QListWidget, QFileDialog, QInputDialog, QGridLayout, QFormLayout, QLineEdit, QDialog, QCheckBox, QDialogButtonBox, QMessageBox
)
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import json
import numpy as np
import os
from core.group_analysis import GroupAnalysis
from core.output_manager import OutputManager
from core.spheroid_experiment import SpheroidExperiment
from core.processing import *

class IntroPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setTitle("Welcome")
        #self.setSubTitle("Load replicates to begin.")
        # Initialising the group_analysis object
        self.group_analysis = GroupAnalysis()
        self.display_names_list = []

        self.registerField("replicateCount*", self, "replicateCount")

        # This will hold our backend experiment objects
        self.stim_params = None
        # This will hold the current experiment settings
        self.experiment_settings = None

        # 1) Add the ListWidget & “Load” button
        self.list_widget = QListWidget()
        self.btn_new = QPushButton("Clear Replicates")
        self.btn_new.clicked.connect(self.clear_replicates)
        self.btn_load = QPushButton("Load Replicates")
        self.btn_load.clicked.connect(self.load_replicate)
        self.btn_exp_settings = QPushButton("Experiment Settings")
        self.btn_exp_settings.clicked.connect(self.show_experiment_settings_dialog)
        
        # 3) Layout
        v = QVBoxLayout()
        v.addWidget(self.btn_new)
        v.addWidget(self.btn_load)
        v.addWidget(QLabel("Loaded Replicates:"))
        v.addWidget(self.list_widget)
        v.addWidget(self.btn_exp_settings)
        self.setLayout(v)

    def isComplete(self):
        return len(self.display_names_list) > 0

    def show_experiment_settings_dialog(self):
        dlg = ExperimentSettingsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            # Here, extract the settings from the dialog and store them
            self.experiment_settings = dlg.get_settings()

    def clear_replicates(self):
        """Start a brand-new experiment group."""
        self.group_analysis.clear_experiments()
        self.list_widget.clear()

        # overwrite the wizard‐level shared data
        wiz = self.wizard()
        wiz.group_analysis        = self.group_analysis
        wiz.display_names_list    = []

        # Also clear our own display_names_list
        self.display_names_list = []
        
        # Re-evaluate isComplete
        self.completeChanged.emit()

    def load_replicate(self):
        """Ask the user to pick a folder, build & run the SpheroidExperiment, and display it."""
        
        if self.experiment_settings is None:
            if not self.show_experiment_settings_dialog():
                return   # user cancelled, bail out

        settings = self.experiment_settings

        folder = QFileDialog.getExistingDirectory(self, "Select replicate folder")
        if not folder:
            return
        
        # collect all .txt files in that folder
        paths = [os.path.join(folder, f)
                 for f in os.listdir(folder)
                 if f.lower().endswith(".txt")]
        if not paths:
            # optional: warn “no .txt found”
            return

        filtered = {k: v for k, v in settings.items() if k != "output_folder"}
        exp = SpheroidExperiment(paths,**filtered)
        
        self.group_analysis.add_experiment(exp)
        # store it and show it in the list
        self.display_names_list.append(f"{os.path.basename(folder)}")
        display_name = f"{os.path.basename(folder)}"
        self.list_widget.addItem(display_name)

        wiz = self.wizard()
        wiz.group_analysis = self.group_analysis
        wiz.display_names_list = self.display_names_list

        # if your Next button is gated on isComplete(), let Qt know the page state changed:
        self.completeChanged.emit()

    def validatePage(self):
        """
        This is called automatically when the user clicks 'Continue'.
        We can use it to stash our replicates on the wizard for later pages.
        """
        # e.g. store into the wizard object:
        self.wizard().group_analysis = self.group_analysis
        self.wizard().display_names_list = self.display_names_list
        return True
    
    def validatePage(self):
        if len(self.display_names_list) == 0:
            QMessageBox.warning(self, "No Replicates Loaded", "Please load at least one replicate before continuing.")
            return False

        self.wizard().group_analysis = self.group_analysis
        self.wizard().display_names_list = self.display_names_list
        return True
    

class ExperimentSettingsDialog(QDialog):
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Experiment Settings")

        self.qsettings = QSettings("HashemiLab", "NeuroStemVolt")

        defaults = {
            "file_length":           self.qsettings.value("file_length",           100,    type=int),
            "acquisition_frequency": self.qsettings.value("acquisition_frequency", 10,     type=int),
            "peak_position":         self.qsettings.value("peak_position",         257,    type=int),
            "treatment":             self.qsettings.value("treatment",             "None", type=str),
            "waveform":              self.qsettings.value("waveform",              "5HT",  type=str),
            "time_between_files":    self.qsettings.value("time_between_files",    10,     type=int),
            "files_before_treatment":self.qsettings.value("files_before_treatment",3,      type=int),
            "file_type":             self.qsettings.value("file_type",             "None", type=str),
            # stim_params might be stored as JSON
            "stim_params":           json.loads(self.qsettings.value("stim_params", "{}")),
            "output_folder": self.qsettings.value("output_folder", "", type=str),
        }

        vbox = QVBoxLayout()
        
        # Form layout for labeled fields
        form = QFormLayout()
        vbox.addLayout(form)

        self.le_file_length = QLineEdit(str(defaults["file_length"]));            form.addRow("File Length:", self.le_file_length)
        self.le_acq_freq    = QLineEdit(str(defaults["acquisition_frequency"]));  form.addRow("Acquisition Freq:", self.le_acq_freq)
        self.le_peak_pos    = QLineEdit(str(defaults["peak_position"]));          form.addRow("Peak Pos:", self.le_peak_pos)
        self.le_treatment   = QLineEdit(defaults["treatment"]);                   form.addRow("Treatment:", self.le_treatment)

        self.cb_waveform    = QComboBox();  self.cb_waveform.addItems(["5HT","Else"])
        self.cb_waveform.setCurrentText(defaults["waveform"]);                     form.addRow("Waveform:", self.cb_waveform)

        self.le_time_btw    = QLineEdit(str(defaults["time_between_files"]));     form.addRow("Time Between Files:", self.le_time_btw)
        self.le_files_before= QLineEdit(str(defaults["files_before_treatment"])); form.addRow("Files Before Treatment:", self.le_files_before)

        self.cb_file_type   = QComboBox(); self.cb_file_type.addItems(["None","Spontaneous","Stimulation"])
        self.cb_file_type.setCurrentText(defaults["file_type"]);                   form.addRow("File Type:", self.cb_file_type)

        # store loaded stim_params so get_settings() can return it if user doesn’t change it
        self.stim_params = defaults["stim_params"]

        h_output = QHBoxLayout()
        self.le_output_folder = QLineEdit(defaults["output_folder"])
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self.browse_output_folder)
        h_output.addWidget(self.le_output_folder)
        h_output.addWidget(btn_browse_output)
        form.addRow("Output Folder:", h_output)

        self.setLayout(vbox)

        # Add dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        vbox.addWidget(buttons)
        buttons.accepted.connect(self.handle_accept)
        buttons.rejected.connect(self.reject)

    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.le_output_folder.setText(folder)

    def handle_accept(self):
        # if they choose stimulation, pop the sub-dialog
        if self.cb_file_type.currentText() == "Stimulation":
            dlg = StimParamsDialog(self, defaults=self.stim_params)
            if dlg.exec_() == QDialog.Accepted:
                self.stim_params = dlg.get_params()
            else:
                return  # abort if they cancelled stim-params

        # now persist *all* fields
        self.qsettings.setValue("file_length",           int(self.le_file_length.text()))
        self.qsettings.setValue("acquisition_frequency", int(self.le_acq_freq.text()))
        self.qsettings.setValue("peak_position",         int(self.le_peak_pos.text()))
        self.qsettings.setValue("treatment",             self.le_treatment.text())
        self.qsettings.setValue("waveform",              self.cb_waveform.currentText())
        self.qsettings.setValue("time_between_files",    int(self.le_time_btw.text()))
        self.qsettings.setValue("files_before_treatment",int(self.le_files_before.text()))
        self.qsettings.setValue("file_type",             self.cb_file_type.currentText())
        self.qsettings.setValue("output_folder", self.le_output_folder.text())
        # stim_params → JSON string
        print("Saving stim_params:", self.stim_params)
        self.qsettings.setValue("stim_params", json.dumps(self.stim_params))
        print("After setValue, reload raw:", self.qsettings.value("stim_params"))

        # close dialog
        self.accept()

    def get_settings(self):
        return {
            "file_length":            int(self.le_file_length.text()),
            "acquisition_frequency":  int(self.le_acq_freq.text()),
            "peak_position":          int(self.le_peak_pos.text()),
            "treatment":              self.le_treatment.text(),
            "waveform":               self.cb_waveform.currentText(),
            "time_between_files":     float(self.le_time_btw.text()),
            "files_before_treatment": int(self.le_files_before.text()),
            "file_type":              self.cb_file_type.currentText(),
            "stim_params":            self.stim_params,    # initialized in __init__
            "output_folder": self.le_output_folder.text(),
        }

class StimParamsDialog(QDialog):
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Stimulation Parameters")
        form = QFormLayout(self)
        self.edits = {}
        params = ["start", "duration", "frequency", "amplitude", "pulses"]
        defaults = defaults or {"start": 5.0, "duration": 2.0, "frequency": 20, "amplitude": 0.5, "pulses": 50}
        for p in params:
            edit = QLineEdit(str(defaults[p]))
            form.addRow(f"{p.capitalize()}:", edit)
            self.edits[p] = edit
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def get_params(self):
        return {k: float(self.edits[k].text()) for k in self.edits}


### Second Page

class ColorPlotPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Color Plot")

        self.selected_processors = []
    
        # Left controls
        btn_revert = QPushButton("Revert Changes")
        btn_revert.clicked.connect(self.revert_processing)
        btn_eval = QPushButton("Evaluate")
        btn_eval.clicked.connect(self.run_processing)

        self.cbo_rep = QComboBox(); 
        self.cbo_rep.currentIndexChanged.connect(self.on_replicate_changed)
        
        #### Handle the signal from cbo_rep

        self.txt_file = QLineEdit(); 
        self.txt_file.setReadOnly(True)

        # Default indexes to visualize
        self.current_rep_index = 0
        self.current_file_index = 0

        btn_prev = QPushButton("Previous"); btn_next = QPushButton("Next")
        btn_prev.clicked.connect(self.on_prev_clicked)
        btn_next.clicked.connect(self.on_next_clicked)

        #### Handle the signal from prev and next btn

        btn_filter = QPushButton("Filter Options"); #btn_apply = QPushButton("Apply Filtering")
        btn_filter.clicked.connect(self.show_processing_options)
        btn_save = QPushButton("Save Current Plots"); 
        btn_save.clicked.connect(self.save_IT_ColorPlot_Plots)
        btn_export = QPushButton("Export Current IT")
        btn_export.clicked.connect(self.save_processed_data_IT)
        btn_export_all = QPushButton("Export All ITs")
        btn_export_all.clicked.connect(self.save_all_ITs)

        left = QVBoxLayout()
        left.addWidget(btn_revert)
        left.addWidget(btn_eval)
        left.addWidget(self.cbo_rep)
        left.addWidget(self.txt_file)

        nav = QHBoxLayout(); nav.addWidget(btn_prev); nav.addWidget(btn_next)
    
        left.addLayout(nav)
        left.addWidget(btn_filter)
        #left.addWidget(btn_apply)
        left.addWidget(btn_save)
        left.addWidget(btn_export)
        left.addWidget(btn_export_all)
        left.addStretch(1)

        # Right plots
        self.main_plot = PlotCanvas(self, width=5, height=4)

        self.it_plot = PlotCanvas(self, width=2.5, height=2)

        #cv_plot = PlotCanvas(self, width=2.5, height=2)
        #cv_plot.plot_line()

        bottom = QHBoxLayout()
        bottom.addWidget(self.it_plot)
        #bottom.addWidget(cv_plot)

        right = QVBoxLayout()
        right.addWidget(self.main_plot)
        right.addLayout(bottom)

        layout = QHBoxLayout()
        layout.addLayout(left)
        layout.addLayout(right)
        self.setLayout(layout)
    
    def initializePage(self):
        # Default index
        def_index = 0

        group_analysis = self.wizard().group_analysis
        display_names_list = self.wizard().display_names_list
        print(display_names_list)
        names = self.wizard().display_names_list or []

        self.cbo_rep.clear()
        self.cbo_rep.addItems(display_names_list)
        self.cbo_rep.setCurrentIndex(def_index)
        self.cbo_rep.setEnabled(True)
        
        if not names:
            # no replicates → disable everything
            self.cbo_rep.setEnabled(False)
            self.txt_file.clear()
            self.clear_all()   # blank the plots too
            return
        else:
            current_exp = group_analysis.get_single_experiments(def_index)
            current_file = current_exp.get_spheroid_file(def_index)
            current_file_name = os.path.basename(current_file.get_filepath())

            self.txt_file.setText(current_file_name)

            group_analysis = self.wizard().group_analysis
            processed_data = current_file.get_processed_data()
            metadata = current_file.get_metadata()
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position")

            self.main_plot.plot_color(processed_data=processed_data)
            self.it_plot.plot_IT(processed_data=processed_data,metadata=metadata,peak_position=peak_pos)

    def clear_all(self):
        # reset indices
        self.current_rep_index = 0
        self.current_file_index = 0

        # empty the combo box & filename display
        self.cbo_rep.clear()
        self.txt_file.clear()

        # clear both canvases
        for canvas in (self.main_plot, self.it_plot):
            canvas.fig.clear()
            canvas.draw()

    def on_replicate_changed(self, index):
        self.current_rep_index = index
        self.current_file_index = 0
        self.update_file_display()

    def update_file_display(self):
        group_analysis = self.wizard().group_analysis
        try:
            exp = group_analysis.get_single_experiments(self.current_rep_index)
            sph_file = exp.get_spheroid_file(self.current_file_index)
            file_name = os.path.basename(sph_file.get_filepath())
            self.txt_file.setText(file_name)

            processed_data = sph_file.get_processed_data()
            metadata = sph_file.get_metadata()
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position")

            self.main_plot.plot_color(processed_data=processed_data)
            self.it_plot.plot_IT(processed_data=processed_data,metadata=metadata,peak_position=peak_pos)
        except IndexError:
            self.txt_file.setText("No file at this index")

    def on_next_clicked(self):
        exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
        if self.current_file_index < exp.get_file_count() - 1:
            self.current_file_index += 1
            self.update_file_display()

    def on_prev_clicked(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.update_file_display()

    def run_processing(self):
        group_analysis = self.wizard().group_analysis
        peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)

        # Always include FindAmplitude, but ensure it's not duplicated
        user_processors = self.selected_processors or []
        mandatory = FindAmplitude(peak_pos)

        processors = [p for p in user_processors if not isinstance(p, FindAmplitude)]
        processors.append(mandatory)

        group_analysis.set_processing_options_exp(processors)
        for exp in group_analysis.get_experiments():
            exp.run()

        self.update_file_display()

    def revert_processing(self):
        group_analysis = self.wizard().group_analysis
        for exp in group_analysis.get_experiments():
            exp.revert_processing()
        self.update_file_display()

    def show_processing_options(self):
        dlg = ProcessingOptionsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            selected_names = dlg.get_selected_processors()
            peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position")
            self.selected_processors = [
                ProcessingOptionsDialog.get_processor_instance(name, peak_pos)
                for name in selected_names
                if ProcessingOptionsDialog.get_processor_instance(name, peak_pos) is not None
            ]

    def validatePage(self):
        # Automatically add FindAmplitude processor and run it before proceeding
        group_analysis = self.wizard().group_analysis
        peak_pos = QSettings("HashemiLab", "NeuroStemVolt").value("peak_position", type=int)

        processor = FindAmplitude(peak_pos)

        for exp in group_analysis.get_experiments():
            exp.set_processing_steps([processor])  # only FindAmplitude
            exp.run()

        return True  # allow transition to next page

    def save_all_ITs(self):
        group_analysis = self.wizard().group_analysis
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        OutputManager.save_all_ITs(group_analysis, output_folder_path)

    def save_IT_ColorPlot_Plots(self):
        exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
        sph_file = exp.get_spheroid_file(self.current_file_index)
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        sph_file.visualize_color_plot_data(title_suffix = "", save_path=output_folder_path)
        sph_file.visualize_IT_profile(QSettings("HashemiLab", "NeuroStemVolt").value("output_folder"))

    def save_processed_data_IT(self):
        exp = self.wizard().group_analysis.get_single_experiments(self.current_rep_index)
        sph_file = exp.get_spheroid_file(self.current_file_index)
        output_folder_path = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        OutputManager.save_IT_profile(sph_file,output_folder_path)

class ProcessingOptionsDialog(QDialog):
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)

        self.setWindowTitle("Filtering Options")

        self.qsettings = QSettings("HashemiLab", "NeuroStemVolt")

        # List of available processors and their default checked state
        self.processor_options = [
            ("Background Subtraction", True),
            ("Gaussian Smoothing 2D", False),
            ("Rolling Mean", False),
            ("Butterworth Filter", True),
            ("Savitzky-Golay Filter", False),
            ("Baseline Correction", True),
            ("Normalize", True),
            ("Find Amplitude", True),
            #("Exponential Fitting", True),
        ]

        self.checkboxes = {}
        layout = QVBoxLayout()

        for name, checked in self.processor_options:
            if name == "Find Amplitude":
                continue
            cb = QCheckBox(name)
            cb.setChecked(checked)
            layout.addWidget(cb)
            self.checkboxes[name] = cb
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_processor_instance(name, peak_position=None):
        if name == "Background Subtraction":
            return BackgroundSubtraction(region=(0, 10))
        elif name == "Gaussian Smoothing 2D":
            return GaussianSmoothing2D()
        elif name == "Rolling Mean":
            return RollingMean()
        elif name == "Butterworth Filter":
            return ButterworthFilter()
        elif name == "Savitzky-Golay Filter":
            return SavitzkyGolayFilter(w=20, p=2)
        elif name == "Baseline Correction":
            return BaselineCorrection()
        elif name == "Normalize":
            return Normalize(peak_position)
        elif name == "Find Amplitude":
            return FindAmplitude(peak_position)
        elif name == "Exponential Fitting":
            return ExponentialFitting()
        else:
            return None

    def get_selected_processors(self):
        """Return a list of selected processor names."""
        return [name for name, cb in self.checkboxes.items() if cb.isChecked()]


### Third Page

class ResultsPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Results")

        btn_back = QPushButton("Back")

        # Analysis buttons
        btn_avg = QPushButton("Average Over Experiments")
        btn_fit = QPushButton("Decay Exponential Fitting")
        btn_param = QPushButton("Exponential Parameter over time")
        btn_amp = QPushButton("Amplitudes Over Time")

        analysis = QGridLayout()
        analysis.addWidget(btn_avg, 0, 0)
        analysis.addWidget(btn_fit, 0, 1)
        analysis.addWidget(btn_param, 1, 0)
        analysis.addWidget(btn_amp, 1, 1)

        # Result plot & export
        self.result_plot = PlotCanvas(self, width=5, height=4)
        # Connect analysis buttons to their respective methods
        btn_avg.clicked.connect(lambda: self.result_plot.show_average_over_experiments(self.wizard().group_analysis))
        btn_fit.clicked.connect(self.handle_decay_fit)
        #btn_fit.clicked.connect(lambda: self.result_plot.show_decay_exponential_fitting(self.wizard().group_analysis))
        btn_param.clicked.connect(lambda: self.result_plot.show_tau_param_over_time(self.wizard().group_analysis))
        btn_amp.clicked.connect(lambda: self.result_plot.show_amplitudes_over_time(self.wizard().group_analysis))

        btn_save = QPushButton("Save Current Plot")
        btn_save.clicked.connect(self.save_current_plot)
        btn_save_all = QPushButton("Save All Plots")
        btn_save_all.clicked.connect(self.save_all_plots)
        btn_export = QPushButton("Export metrics as csv")
        btn_export.clicked.connect(self.export_all_as_csv)

        self.analysis_buttons = [btn_avg, btn_fit, btn_param, btn_amp, btn_save, btn_save_all, btn_export]

        right = QVBoxLayout()
        right.addWidget(self.result_plot)
        right.addWidget(btn_save)
        right.addWidget(btn_save_all)
        right.addWidget(btn_export)
        right.addStretch(1)

        layout = QHBoxLayout()
        left = QVBoxLayout(); left.addWidget(btn_back); left.addLayout(analysis); left.addStretch(1)
        layout.addLayout(left)
        layout.addLayout(right)
        self.setLayout(layout)

    def export_all_as_csv(self):
        """Export all relevant metrics as CSV files using OutputManager."""
        output_folder = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        group_analysis = self.wizard().group_analysis
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.warning(self, "No Output Folder", "Please set a valid output folder in Experiment Settings.")
            return

        # Save all relevant CSVs using OutputManager
        OutputManager.save_all_ITs(group_analysis, output_folder)
        OutputManager.save_all_peak_amplitudes(group_analysis, output_folder)
        OutputManager.save_all_reuptake_curves(group_analysis, output_folder)
        OutputManager.save_all_exponential_fitting_params(group_analysis, output_folder)

        QMessageBox.information(
            self,
            "CSV Export Complete",
            f"All metrics exported as CSV files to:\n{output_folder}"
        )

    def save_current_plot(self):
        """Save the currently displayed plot as a PNG file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Plot As",
            "",
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if file_path:
            self.result_plot.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Plot Saved", f"Plot saved to:\n{file_path}")

    def save_all_plots(self):
        """Save all group-level plots using OutputManager."""
        output_folder = QSettings("HashemiLab", "NeuroStemVolt").value("output_folder")
        group_analysis = self.wizard().group_analysis
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.warning(self, "No Output Folder", "Please set a valid output folder in Experiment Settings.")
            return

        # Save all group-level plots using OutputManager
        OutputManager.save_mean_ITs_plot(group_analysis, output_folder)
        OutputManager.save_plot_tau_over_time(group_analysis, output_folder)
        OutputManager.save_plot_exponential_fit_aligned(group_analysis, output_folder)
        OutputManager.save_plot_all_amplitudes_over_time(group_analysis, output_folder)
        OutputManager.save_plot_mean_amplitudes_over_time(group_analysis, output_folder)
        # Add more OutputManager plot saves as needed

        QMessageBox.information(self, "Plots Saved", f"All group-level plots saved to:\n{os.path.join(output_folder, 'plots')}")
    
    def initializePage(self):
        if self.wizard().group_analysis.get_experiments():
            self.enable_analysis_buttons()
        else:
            self.clear_all()

    def clear_all(self):
        # Clear the plot
        self.result_plot.fig.clear()
        self.result_plot.draw()
        # Optionally disable analysis/export buttons
        for btn in getattr(self, 'analysis_buttons', []):
            btn.setEnabled(False)

    def enable_analysis_buttons(self):
        for btn in getattr(self, 'analysis_buttons', []):
            btn.setEnabled(True)

    def handle_decay_fit(self):
        group_analysis = self.wizard().group_analysis
        experiments = group_analysis.get_experiments()
        if not experiments:
            return
        # Use the first experiment to get the list of files (timepoints)
        exp = experiments[0]
        file_names = [os.path.basename(exp.get_spheroid_file(i).get_filepath()) for i in range(exp.get_file_count())]
        dlg = TimepointSelectionDialog(file_names, self)
        if dlg.exec_() == QDialog.Accepted:
            idx = dlg.get_selected_index()
            self.result_plot.show_decay_exponential_fitting(group_analysis, replicate_time_point=idx)

class TimepointSelectionDialog(QDialog):
    def __init__(self, file_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Timepoint for Exponential Fit")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the file/timepoint to fit:"))
        self.combo = QComboBox(self)
        self.combo.addItems(file_names)
        layout.addWidget(self.combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_index(self):
        return self.combo.currentIndex()

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.add_subplot(111) 
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()
        self.cbar = None

    def plot_color(self, processed_data, title_suffix=None):
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

        # Add colorbar using the figure object (Qt-safe)
        self.cbar = self.fig.colorbar(im, ax=self.axes, label="Current (nA)")

        self.axes.set_xlabel("Time Points")
        self.axes.set_ylabel("Voltage Steps")
        title = f"Color Plot{': ' + title_suffix if title_suffix else ''}\nRange: [{vmin:.2f}, {vmax:.2f}] nA"
        self.axes.set_title(title)

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

        self.axes = self.fig.add_subplot(111)

        profile = processed_data[:, peak_position]

        # Plot the main profile
        self.axes.plot(profile, label="I-T Profile", color='blue', linewidth=1.5)

        # Plot peak markers if metadata is provided
        if metadata and 'peak_amplitude_positions' in metadata:
            peak_indices = metadata['peak_amplitude_positions']
            peak_values = metadata.get('peak_amplitude_values', profile[peak_indices])
            # Handle both single value and list/array for peaks
            if isinstance(peak_indices, (list, np.ndarray)) and isinstance(peak_values, (list, np.ndarray)):
                for idx, val in zip(peak_indices, peak_values):
                    if 0 <= idx < len(profile):
                        self.axes.scatter(idx, val, color='red', zorder=5)
                        self.axes.annotate(f"{val:.2f}", (idx, val), textcoords="offset points",
                                           xytext=(0, 10), ha='center', fontsize=9, color='red')
            else:
                # Assume single value
                try:
                    idx = int(peak_indices)
                    val = float(peak_values)
                    if 0 <= idx < len(profile):
                        self.axes.scatter(idx, val, color='red', zorder=5)
                        self.axes.annotate(f"{val:.2f}", (idx, val), textcoords="offset points",
                                           xytext=(0, 10), ha='center', fontsize=9, color='red')
                except Exception:
                    pass

        # Axis labeling and formatting
        self.axes.set_xlabel("Time Points")
        self.axes.set_ylabel("Current (nA)")
        title = "I-T Profile"
        if peak_position is not None:
            title += f" at Peak Position {peak_position}"
        self.axes.set_title(title)
        self.axes.grid(False)
        self.axes.legend()

        #self.fig.tight_layout()
        # Render to canvas
        self.draw()

    def show_average_over_experiments(self, group_analysis):
        """
        Plots the mean amplitudes over time across all experiments,
        with the standard deviation as a shaded area.
        """
        import numpy as np

        # Get data from group_analysis
        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = group_analysis.amplitudes_over_time_all_experiments()
        if time_points is None:
            self.axes.clear()
            self.axes.set_title("No data to plot")
            self.draw()
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

    def show_decay_exponential_fitting(self, group_analysis, replicate_time_point=0):
        """
        Plots post-peak IT decays, mean trace, exponential fit, 95% CI, and half-life on the embedded canvas.
        """
        from scipy.stats import t
        import numpy as np

        # Get fit and aligned ITs from group_analysis
        result = group_analysis.exponential_fitting_replicated(replicate_time_point)
        if result is None:
            self.axes.clear()
            self.axes.set_title("No data to fit")
            self.draw()
            return

        time_all, cropped_ITs, _, t_half, popt, pcov, A_fit, tau_fit, C_fit = result
        n_exps, n_post = cropped_ITs.shape
        t_rel = np.arange(n_post)
        mean_IT = np.nanmean(cropped_ITs, axis=0)
        std_IT  = np.nanstd (cropped_ITs, axis=0)
        t_fit_rel = np.linspace(0, n_post-1, 500)
        y_fit     = A_fit * np.exp(-t_fit_rel / tau_fit) + C_fit

        # 95% CI of the fit via Jacobian
        dof  = max(0, len(time_all) - len(popt))
        tval = t.ppf(0.975, dof)
        J        = np.empty((len(t_fit_rel), 3))
        J[:, 0]  = np.exp(-t_fit_rel / tau_fit)
        J[:, 1]  = A_fit * (t_fit_rel / tau_fit**2) * np.exp(-t_fit_rel / tau_fit)
        J[:, 2]  = 1
        ci       = np.sqrt(np.sum((J @ pcov) * J, axis=1)) * tval
        lower_ci = y_fit - ci
        upper_ci = y_fit + ci

        # Plot on the embedded axes
        self.axes.clear()

        # a) each replicate in light gray
        for row in cropped_ITs:
            self.axes.plot(t_rel, row, color='gray', alpha=0.3, lw=1, label='_nolegend_')

        # b) mean ± 1 SD ribbon
        self.axes.fill_between(t_rel, mean_IT - std_IT, mean_IT + std_IT, color='C0', alpha=0.2, label='Mean ± 1 SD')

        # c) mean trace
        self.axes.plot(t_rel, mean_IT, color='C0', lw=2, label='Mean trace')

        # d) fitted exponential curve
        self.axes.plot(t_fit_rel, y_fit, color='C1', lw=2, label='Exp fit')

        # e) 95% CI around the fit
        self.axes.fill_between(t_fit_rel, lower_ci, upper_ci, color='C1', alpha=0.3, label='95% CI')

        # f) half-life marker
        self.axes.axvline(t_half, color='magenta', ls='--', label=f't½ ≈ {t_half:.1f} pts')

        # 7) labels & styling
        self.axes.set_xlabel('Time since peak (points)', fontsize=12)
        self.axes.set_ylabel('Current (nA)', fontsize=12)
        self.axes.set_title('Post-peak IT decays & exponential fit', fontsize=14)
        self.axes.legend(frameon=False)
        self.axes.grid(False)
        self.fig.tight_layout()
        self.draw()

    def show_tau_param_over_time(self, group_analysis):
        """
        Plots the exponential decay parameter tau over replicate time points on the embedded canvas.
        """
        import numpy as np

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

    def show_amplitudes_over_time(self, group_analysis):
        """
        Plot all amplitudes over time for each experiment as separate lines on the embedded canvas.
        """
        import numpy as np

        time_points, mean_amplitudes, all_amplitudes, files_before_treatment = group_analysis.amplitudes_over_time_all_experiments()
        if time_points is None:
            self.axes.clear()
            self.axes.set_title("No data to plot")
            self.draw()
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