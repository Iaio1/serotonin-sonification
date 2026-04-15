from PyQt5.QtWidgets import QPushButton, QListWidget

def apply_custom_styles(widget):
    """
    Apply custom visual styles to QPushButton or QListWidget widgets.

    This function assigns background color, font, padding, and hover effects
    depending on the button label content (e.g., "save", "clear") or widget type.

    Args:
        widget (QWidget): The widget to which styles should be applied. 
            Currently supports:
                - QPushButton: Styles vary by function (e.g., "save", "clear", general).
                - QListWidget: Styled with a pastel background and bold text.

    Returns:
        None
    """
    if isinstance(widget, QPushButton):
        label = widget.text().lower()

        if "save" in label or "export" in label:
            # Save/export buttons (blue)
            widget.setStyleSheet("""
                QPushButton {
                    background-color: #7850C8;
                    color: white;
                    font-family: Arial, sans-serif;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #6A45B5;
                }
                QPushButton:pressed {
                    background-color: #5A38A0;
                }
                QPushButton:disabled {
                    background-color: #C4B0E8;
                    color: white;
                }
            """)
        elif "clear" in label or "reverse" in label or "previous" in label or "next" in label:
            # Clear buttons (pink/red)
            widget.setStyleSheet("""
                QPushButton {
                    background-color: #E054A0;
                    color: white;
                    font-family: Arial, sans-serif;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #C4458D;
                }
                QPushButton:pressed {
                    background-color: #A8377A;
                }
                QPushButton:disabled {
                    background-color: #F0B0D0;
                    color: white;
                }
            """)
        else:
            # General buttons (green)
            widget.setStyleSheet("""
                QPushButton {
                    background-color: #7850C8;
                    color: white;
                    font-family: Arial, sans-serif;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #6A45B5;
                }
                QPushButton:pressed {
                    background-color: #5A38A0;
                }
                QPushButton:disabled {
                    background-color: #C4B0E8;
                    color: white;
                }
            """)
    
    elif isinstance(widget, QListWidget):
        widget.setStyleSheet("""
            QListWidget {
                background-color: #CAF2FB;
                color: black;
                font-family: Arial, sans-serif;
                font-weight: bold;
                border-radius: 8px;
                padding: 4px;
            }
        """)