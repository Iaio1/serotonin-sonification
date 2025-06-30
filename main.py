from ui.neurostemvolt_ui import *
import sys

app = QApplication(sys.argv)
wizard = QWizard()
wizard.setWindowTitle("NeuroStemVolt Wizard")
wizard.addPage(IntroPage())
wizard.addPage(ColorPlotPage())
wizard.addPage(ResultsPage())
wizard.show()
sys.exit(app.exec_())

