from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QPushButton, QVBoxLayout, QFileDialog, QGridLayout, QDialog, QMessageBox
)
from PyQt5.QtCore import QSettings, Qt

from core.output_manager import OutputManager
from core.processing import *

from ui.utils.styles import apply_custom_styles
from ui.widgets.plot_canvas import PlotCanvas
from ui.wizard_3_results.timepoint_dialog import TimepointSelectionDialog
import os

### Third Page

class ResultsPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)

        # analysis buttons
        btn_avg    = QPushButton("Mean Amplitude Over Experiments");      apply_custom_styles(btn_avg)
        btn_fit    = QPushButton("Decay Exponential Fitting");     apply_custom_styles(btn_fit)
        btn_param  = QPushButton("Tau Over Time");                 apply_custom_styles(btn_param)
        btn_amp    = QPushButton("Individual Amplitudes Over Time");          apply_custom_styles(btn_amp)

        # grid of analysis buttons
        analysis = QGridLayout()
        analysis.addWidget(btn_avg,    0, 0)
        analysis.addWidget(btn_fit,    1, 1)
        analysis.addWidget(btn_param,  1, 0)
        analysis.addWidget(btn_amp,    0, 1)

        # save/export buttons
        btn_save     = QPushButton("Save Current Plot");          apply_custom_styles(btn_save)
        btn_save_all = QPushButton("Save All Plots");             apply_custom_styles(btn_save_all)
        btn_export   = QPushButton("Export metrics as csv");      apply_custom_styles(btn_export)

        self.analysis_buttons = [btn_avg, btn_fit, btn_param, btn_amp, btn_save, btn_save_all, btn_export]

        # placeholder & plotCanvas
        self.placeholder = QLabel("Select an analysis option to show plot")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.result_plot = PlotCanvas(self, width=5, height=4)
        self.result_plot.hide()  # start hidden

        # connect buttons to show‐and‐plot
        for btn, fn in (
            (btn_avg,   lambda: self.result_plot.show_average_over_experiments(self.wizard().group_analysis)),
            (btn_fit,   self.handle_decay_fit),
            (btn_param, lambda: self.result_plot.show_tau_param_over_time(self.wizard().group_analysis)),
            (btn_amp,   lambda: self.result_plot.show_amplitudes_over_time(self.wizard().group_analysis)),
        ):
            btn.clicked.connect(lambda _, f=fn: self._reveal_and_call(f))

        btn_save.clicked.connect(self.save_current_plot)
        btn_save_all.clicked.connect(self.save_all_plots)
        btn_export.clicked.connect(self.export_all_as_csv)

        # layout assembly
        main_layout = QVBoxLayout(self)

        # analysis buttons at top
        main_layout.addLayout(analysis)

        # save/export
        main_layout.addWidget(btn_save)
        main_layout.addWidget(btn_save_all)
        main_layout.addWidget(btn_export)

        # placeholder + future plot
        main_layout.addWidget(self.placeholder, stretch=1)
        main_layout.addWidget(self.result_plot, stretch=3)

        # footer
        footer = QLabel("© 2025 Hashemi Lab · NeuroStemVolt · v1.0.0")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray; font-size: 10pt;")
        main_layout.addWidget(footer)

        self.setLayout(main_layout)


    def _reveal_and_call(self, plot_fn):
        """Hide placeholder and show canvas, then call the plotting fn."""
        if self.placeholder.isVisible():
            self.placeholder.hide()
            self.result_plot.show()
        plot_fn()

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




