from PyQt5 import QtWidgets
import numpy as np
from PyQt5.QtWidgets import QGraphicsScene
import utils
from utils import *
from img_proc import *
from ui.PlotDialogUI import Ui_PlotDialog
from PyQt5.QtCore import pyqtSignal


class PlotDialog(QtWidgets.QDialog):
    closed = pyqtSignal()

    def __init__(self):
        super(PlotDialog, self).__init__()
        self.ui = Ui_PlotDialog()
        self.ui.setupUi(self)

        self.scene = QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)

        self.currentImage: np.ndarray = None

        # utils.apply_style_sheet(self)

    def plot(self, image: np.ndarray):
        self.currentImage = image
        pixmap = np2QPixmap(image)
        pixmap.scaled(self.ui.graphicsView.width(),
                      self.ui.graphicsView.height())
        self.scene.addPixmap(pixmap)

    def closeEvent(self, a0):
        ret=super().closeEvent(a0)
        if a0.isAccepted():
            self.closed.emit()
        return ret
