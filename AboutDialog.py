from PyQt5 import QtWidgets
import numpy as np
from PyQt5.QtWidgets import QGraphicsScene
import utils
from utils import *
from img_proc import *
from ui.AboutDialogUI import Ui_AboutDialog
from PyQt5.QtCore import pyqtSignal


class AboutDialog(QtWidgets.QDialog):
    closed = pyqtSignal()

    def __init__(self):
        super(AboutDialog, self).__init__()
        self.ui = Ui_AboutDialog()
        self.ui.setupUi(self)
        self.ui.confirmBtn.clicked.connect(self.close)

        # utils.apply_style_sheet(self)

    def closeEvent(self, a0):
        ret=super().closeEvent(a0)
        if a0.isAccepted():
            self.closed.emit()
        return ret

