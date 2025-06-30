from PyQt5.QtWidgets import (
    QApplication, QWizard, QComboBox, QLineEdit, QWizardPage, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QListWidget, QFileDialog, QInputDialog, QGridLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
from core.group_analysis import GroupAnalysis
from core.spheroid_experiment import SpheroidExperiment

class IntroPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("NeuroStemVolt")
        
        # This will hold our backend experiment objects
        self.replicates = []  
        # This will hold the current experiment settings
        self.experiment_settings = None

        # 1) Add the ListWidget & “Load” button
        self.list_widget = QListWidget()
        self.btn_new = QPushButton("Clear Replicates.")
        self.btn_new.clicked.connect(self.clear_replicates)
        self.btn_load = QPushButton("Load Replicate.")
        self.btn_load.clicked.connect(self.load_replicate)
        self.btn_exp_settings = QPushButton("Experiment Settings")
        self.btn_exp_settings.clicked.connect(self.load_experiment_settings)
        
        # 3) Layout
        v = QVBoxLayout()
        v.addWidget(self.btn_new)
        v.addWidget(self.btn_load)
        v.addWidget(QLabel("Loaded Replicates:"))
        v.addWidget(self.list_widget)
        v.addWidget(self.btn_exp_settings)
        self.setLayout(v)
    
    def clear_replicates(self):
        """Start a brand-new experiment group."""
        self.replicates.clear()
        self.list_widget.clear()
    
    def load_replicate(self):
        """Ask the user to pick a folder, build & run the SpheroidExperiment, and display it."""
        
        if self.experiment_settings is None:
            # Prompt user for settings (could be a custom dialog)
            self.experiment_settings = self.load_experiment_settings()
        settings = self.experiment_settings
        
        #exp = SpheroidExperiment(
        #    paths,
        #    file_length=
        #    acquisition_frequency=
        #    acquisition_frequency=10, 
        #    peak_position=257,
        #    treatment="",
        #    waveform="",  # Added waveform parameter
        #    stim_params=None,
        #    processors,  # Default to None
        #    time_between_files= 10.0, # Default time between files (between stimulations and recodings (e.g. stimulating every 10 min and recoding) in minutes
        #    files_before_treatment = 3  # Default number of files before treatment (e.g. baseline recordings)
        #)

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
    
        #exp = SpheroidExperiment(paths)
        
        # store it and show it in the list
        #self.replicates.append(exp)
        display_name = f"{os.path.basename(folder)}"
        self.list_widget.addItem(display_name)
    
    def validatePage(self):
        """
        This is called automatically when the user clicks 'Continue'.
        We can use it to stash our replicates on the wizard for later pages.
        """
        # e.g. store into the wizard object:
        self.wizard().group_analysis = GroupAnalysis()
        for exp in self.replicates:
            self.wizard().group_analysis.add_experiment(exp)
        return True
    
    def load_experiment_settings(self):
        None


class ColorPlotPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Color Plot")

        # Left controls
        btn_back = QPushButton("Back")
        btn_eval = QPushButton("Evaluate")
        cbo_rep = QComboBox(); cbo_rep.addItem("Replicate x")
        txt_file = QLineEdit(); txt_file.setPlaceholderText("File name")
        btn_prev = QPushButton("Previous"); btn_next = QPushButton("Next")
        btn_filter = QPushButton("Filter Options"); btn_apply = QPushButton("Apply Filtering")
        btn_save = QPushButton("Save Plots"); btn_export = QPushButton("Export Results")

        left = QVBoxLayout()
        left.addWidget(btn_back)
        left.addWidget(btn_eval)
        left.addWidget(cbo_rep)
        left.addWidget(txt_file)
        nav = QHBoxLayout(); nav.addWidget(btn_prev); nav.addWidget(btn_next)
        left.addLayout(nav)
        left.addWidget(btn_filter)
        left.addWidget(btn_apply)
        left.addWidget(btn_save)
        left.addWidget(btn_export)
        left.addStretch(1)

        # Right plots
        main_plot = PlotCanvas(self, width=5, height=4)
        main_plot.plot_color()
        it_plot = PlotCanvas(self, width=2.5, height=2)
        it_plot.plot_line()
        cv_plot = PlotCanvas(self, width=2.5, height=2)
        cv_plot.plot_line()

        bottom = QHBoxLayout()
        bottom.addWidget(it_plot)
        bottom.addWidget(cv_plot)

        right = QVBoxLayout()
        right.addWidget(main_plot)
        right.addLayout(bottom)

        layout = QHBoxLayout()
        layout.addLayout(left)
        layout.addLayout(right)
        self.setLayout(layout)


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
        result_plot = PlotCanvas(self, width=5, height=4)
        result_plot.plot_line()
        btn_save = QPushButton("Save Plot"); btn_export = QPushButton("Export All")

        right = QVBoxLayout()
        right.addWidget(result_plot)
        right.addWidget(btn_save)
        right.addWidget(btn_export)
        right.addStretch(1)

        layout = QHBoxLayout()
        left = QVBoxLayout(); left.addWidget(btn_back); left.addLayout(analysis); left.addStretch(1)
        layout.addLayout(left)
        layout.addLayout(right)
        self.setLayout(layout)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()

    def plot_color(self):
        data = np.random.rand(10, 10)
        self.axes.clear()
        self.axes.imshow(data, aspect='auto', cmap='viridis')
        self.draw()

    def plot_line(self):
        t = np.linspace(0, 5, 100)
        y = np.exp(-t) * (1 + 0.1 * np.random.randn(len(t)))
        self.axes.clear()
        self.axes.plot(t, y)
        self.draw()