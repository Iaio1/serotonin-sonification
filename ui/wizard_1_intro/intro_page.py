
from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QPushButton, QVBoxLayout, QListWidget, QMessageBox, QFileDialog, QShortcut, QDialog
)
from PyQt5.QtCore import Qt
from core.group_analysis import GroupAnalysis
from core.spheroid_experiment import SpheroidExperiment
from ui.utils.styles import apply_custom_styles
from ui.wizard_1_intro.settings_dialog import ExperimentSettingsDialog
from ui.wizard_1_intro.settings_dialog import StimParamsDialog

import os

class IntroPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setTitle("Welcome")
        #self.setSubTitle("Load replicates to begin.")
        # Initialising the group_analysis object
        self.group_analysis = GroupAnalysis()
        self.display_names_list = []
        self.number_of_files = 0

        self.registerField("replicateCount*", self, "replicateCount")

        # This will hold our backend experiment objects
        self.stim_params = None
        # This will hold the current experiment settings
        self.experiment_settings = None

        # Add the ListWidget & “Load” button
        self.list_widget = QListWidget()

        # after creating self.list_widget we try and catch if the 
        # user is using delete to get rid of a single experiment
        self.delete_sc = QShortcut(Qt.Key_Backspace, self.list_widget)
        self.delete_sc.setContext(Qt.WidgetWithChildrenShortcut)
        self.delete_sc.activated.connect(self._on_delete_selected)

        apply_custom_styles(self.list_widget)
        self.btn_new = QPushButton("Clear Replicates")
        apply_custom_styles(self.btn_new)
        self.btn_new.clicked.connect(self.clear_replicates)
        self.btn_load = QPushButton("Load Replicates")
        apply_custom_styles(self.btn_load)
        self.btn_load.clicked.connect(self.load_replicate)
        self.btn_exp_settings = QPushButton("Experiment Settings")
        apply_custom_styles(self.btn_exp_settings)
        self.btn_exp_settings.clicked.connect(self.show_experiment_settings_dialog)
        
        # 3) Layout
        v = QVBoxLayout()
        v.addWidget(self.btn_new)
        v.addWidget(self.btn_load)
        label_loaded = QLabel("Loaded Replicates:")
        label_loaded.setStyleSheet("""
            color: black;
            font-family: Helvetica;
            font-weight: bold;
        """)
        footer = QLabel("© 2025 Hashemi Lab · NeuroStemVolt · v1.0.0")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("""
            color: gray;
            font-family: Helvetica;
            font-size: 10pt;
            margin-top: 12px;
        """)
        v.addWidget(label_loaded)
        v.addWidget(self.list_widget)
        v.addWidget(self.btn_exp_settings)
        v.addWidget(footer)
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
                self.load_replicate()
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
        
        if self.number_of_files != 0 and self.number_of_files != len(paths):
            QMessageBox.warning(
                self,
                "Warning! Missing Files!",
                "Folders do not contain the same number of files.\n"
                f"Expected: {self.number_of_files}, Found: {len(paths)}"
            )
            return
        
        # If first replicate, set number_of_files
        if self.number_of_files == 0:
            self.number_of_files = len(paths)

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
    
    def _on_delete_selected(self):
        
        wiz = self.wizard()
    
        selected = self.list_widget.selectedItems()

        rows = sorted((self.list_widget.row(item) for item in selected), reverse=True)
        
        for row in rows:
            self.list_widget.takeItem(row)
            del self.display_names_list[row]
            self.group_analysis.clear_single_experiment(row)
        
        # push the updated data back onto the wizard
        wiz.display_names_list = self.display_names_list
        wiz.group_analysis     = self.group_analysis

        self.completeChanged.emit()