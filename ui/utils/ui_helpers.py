from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QMessageBox

def make_labeled_field_with_help(label_text, widget, help_text):
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)  # Less space between field and button

    help_btn = QPushButton("?")
    help_btn.setFixedSize(20, 20)
    help_btn.setToolTip(help_text)
    help_btn.setStyleSheet("QPushButton { border-radius: 10px; padding: 0px; font-weight: bold; }")

    help_btn.clicked.connect(lambda: QMessageBox.information(container, label_text, help_text))

    layout.addWidget(widget)
    layout.addWidget(help_btn)

    return container
