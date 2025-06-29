import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWizard, QWizardPage, QLabel, QPushButton, QComboBox,
                             QLineEdit, QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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


class IntroPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("NeuroStemVolt")

        btn_new = QPushButton("New Experiment")
        btn_new.clicked
        btn_load = QPushButton("Load Replicates")

        rep_layout = QGridLayout()
        for i in range(1, 5):
            grp = QGroupBox(f"Replicate {i}" if i <= 2 else "Replicate x")
            cbo = QComboBox()
            cbo.addItem("None")
            if i == 1:
                cbo.addItem("Sertraline")
            v = QVBoxLayout(); v.addWidget(cbo)
            grp.setLayout(v)
            rep_layout.addWidget(grp, (i-1)//2, (i-1)%2)

        btn_settings = QPushButton("Experiment Settings")

        layout = QVBoxLayout()
        layout.addWidget(btn_new)
        layout.addWidget(btn_load)
        layout.addLayout(rep_layout)
        layout.addStretch(1)
        layout.addWidget(btn_settings)
        self.setLayout(layout)


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


def main():
    app = QApplication(sys.argv)
    wizard = QWizard()
    wizard.setWindowTitle("NeuroStemVolt Wizard")
    wizard.addPage(IntroPage())
    wizard.addPage(ColorPlotPage())
    wizard.addPage(ResultsPage())
    wizard.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
