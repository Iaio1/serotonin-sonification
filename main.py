from ui.neurostemvolt_ui import *
import sys
import os

app = QApplication(sys.argv)
wizard = QWizard()
wizard.setWindowTitle("NeuroStemVolt Wizard")
icon_path = os.path.join(os.path.dirname(__file__), "ui", "LogoNeuroStemVolt.png")
app.setWindowIcon(QIcon(icon_path))
wizard.addPage(IntroPage())
wizard.addPage(ColorPlotPage())
wizard.addPage(ResultsPage())
wizard.show()
sys.exit(app.exec_())