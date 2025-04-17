from PyQt5 import QtCore, QtGui, QtWidgets
from ui.MainWindowUI import Ui_MainWindow
import cv2
import utils
import logging
import numpy as np
from ultralytics import YOLO
from utils import *
from PyQt5.QtCore import Qt
import traceback
from PyQt5.QtWidgets import QMessageBox
import time
import sys
from img_proc import *
from PlotDialog import PlotDialog
from AboutDialog import AboutDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem


class LoggingHandler(logging.Handler):

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        log_message = self.format(record)
        self.widget.append(log_message)


class MainWindow(QtWidgets.QMainWindow):
    CLASS_NAME = [
        'Creeper', 'Enderman', 'Fox', 'Mooshroom', 'Ocelot', 'Panda',
        'Polar Bear', 'Skeleton', 'Slime', 'Spider', 'Witch',
        'Wither Skeleton', 'Wolf', 'Zombie', 'Zombified Piglin', 'Bee',
        'Chicken', 'Cow', 'Iron Golem', 'Pig'
    ]
    CLASS_PALETTE = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 青色
        (128, 0, 0),  # 暗红色
        (0, 128, 0),  # 暗绿色
        (0, 0, 128),  # 深蓝色
        (128, 128, 0),  # 暗黄色
        (128, 0, 128),  # 暗品红
        (0, 128, 128),  # 暗青色
        (255, 165, 0),  # 橙色
        (139, 69, 19),  # 棕色
        (255, 192, 203),  # 粉红色
        (102, 205, 170),  # 薄荷绿
        (173, 216, 230),  # 浅蓝色
        (240, 230, 140),  # 卡其色
        (218, 165, 32),  # 金色
        (148, 0, 211),  # 深紫色
    ]

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scene = QtWidgets.QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)

        self.capture = None
        self.detector = YOLO('./weights/best.pt')

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.onTimerTimeout)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionPlay.triggered.connect(self.onPlay)
        self.ui.actionPause.triggered.connect(self.onPause)
        self.ui.actionAbout.triggered.connect(self.onAbout)
        self.ui.actionOpenVideo.triggered.connect(self.onOpenVideo)
        self.ui.actionNextFrame.triggered.connect(self.onNextFrame)
        self.ui.actionPrevFrame.triggered.connect(self.onPrevFrame)
        self.ui.actionRefreshFrame.triggered.connect(
            self.onRefreshCurrentFrame)
        self.ui.refreshFrameBtn.clicked.connect(self.onRefreshCurrentFrame)
        self.ui.grayCheckBox.toggled.connect(self.onGrayCheckBoxToggled)
        self.ui.histEqGroupBox.toggled.connect(self.onHistEqGroupBoxToggled)
        self.ui.imgProcGroupBox.toggled.connect(
            self.onImageProcGroupBoxToggled)
        self.ui.playBtn.clicked.connect(self.onPlay)
        self.ui.pauseBtn.clicked.connect(self.onPause)
        self.ui.playSlider.sliderReleased.connect(self.onPlaySliderReleased)
        self.ui.playSlider.sliderPressed.connect(self.onPlaySliderPressed)
        self.ui.playSlider.rangeChanged.connect(self.onPlaySliderRangeChanged)
        self.ui.playSlider.valueChanged.connect(self.onPlaySliderValueChanged)
        self.ui.detGroupBox.toggled.connect(self.onDetGroupBoxToggled)
        self.ui.showHistogramCheckBox.toggled.connect(
            self.onShowHistogramCheckBoxToggled)

        self.ui.classMaskListView.clicked.connect(
            self.onClassMaskListViewClicked)
        self.ui.clsMaskCheckAllBtn.clicked.connect(
            self.onClassMaskCheckAllBtnClicked)
        self.ui.clsMaskUncheckAllBtn.clicked.connect(
            self.onClassMaskUncheckAllBtnClicked)
        self.ui.clsMaskInverseCheckBtn.clicked.connect(
            self.onClassMaskInvertCheckBtnClicked)

        # self.ui.actionLightTheme.triggered.connect(self.onChangeToLightTheme)
        # self.ui.actionDarkTheme.triggered.connect(self.onChangeToDarkTheme)

        self.enableDetection = True
        self.enabelImageProcess = False

        self.isPlaying = False
        self.fineTuning = False
        self.lastFrameTime = time.time()

        sys.stdout = self  # 重定向std到自身
        sys.stderr = self
        logging.getLogger().addHandler(LoggingHandler(
            self.ui.outputTextBroser))

        self.ui.menu_V.addAction(self.ui.logDockWidget.toggleViewAction())
        self.ui.menu_V.addAction(self.ui.toolDockWidget.toggleViewAction())
        self.ui.menu_V.addAction(
            self.ui.classFilterDockWidget.toggleViewAction())

        self.histogramDialog = createPlotDialog(PlotDialog, '直方图', False)
        self.histogramDialog.closed.connect(self.onHistogramDialogClosed)

        self.aboutDialog = AboutDialog()

        self.classMask = [True] * len(self.CLASS_NAME)
        self.initClassMaskListView()

        # self.openVideo('./video/mc_video.mp4')

        # utils.apply_style_sheet(self)

    def onRefreshCurrentFrame(self):
        self.refreshCurrentFrame()
        self.logToScreen('刷新当前帧')

    def initClassMaskListView(self):
        model = QStandardItemModel()
        self.ui.classMaskListView.setModel(model)

        for class_name in self.CLASS_NAME:
            item = QStandardItem(class_name)
            item.setCheckable(True)
            item.setCheckState(Qt.CheckState.Checked)
            model.appendRow(item)

    def updateClassMask(self, index):
        item = self.ui.classMaskListView.model().itemFromIndex(index)
        text = item.text()
        checked = item.checkState() == Qt.CheckState.Checked
        class_index = self.CLASS_NAME.index(text)
        if class_index in range(len(self.CLASS_NAME)):
            self.classMask[class_index] = checked
        return text, checked

    def onClassMaskListViewClicked(self, index):
        text, checked = self.updateClassMask(index)
        self.logToScreen(f"显示类别 '{text}' = {checked}")

        if not self.isPlaying:
            self.refreshCurrentFrame()

    def onClassMaskCheckAllBtnClicked(self):
        model = self.ui.classMaskListView.model()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            item = model.itemFromIndex(index)
            item.setCheckState(Qt.CheckState.Checked)
            self.updateClassMask(index)

        if not self.isPlaying:
            self.refreshCurrentFrame()

    def onClassMaskUncheckAllBtnClicked(self):
        model = self.ui.classMaskListView.model()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            item = model.itemFromIndex(index)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.updateClassMask(index)

        if not self.isPlaying:
            self.refreshCurrentFrame()

    def onClassMaskInvertCheckBtnClicked(self):
        model = self.ui.classMaskListView.model()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            item = model.itemFromIndex(index)
            item.setCheckState(Qt.CheckState.Unchecked if item.checkState(
            ) == Qt.CheckState.Checked else Qt.CheckState.Checked)
            self.updateClassMask(index)

        if not self.isPlaying:
            self.refreshCurrentFrame()

    def onOpenVideo(self):
        fileName, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self, '选择一个视频文件', filter='Videos (*.mp4;*.avi);;All Files(*)')
        if fileName == "":  # 取消
            return
        if not self.openVideo(fileName):
            QtWidgets.QMessageBox.warning('打开文件失败！')

    def onExit(self):
        self.close()

    def onPlay(self):
        if self.capture is None:
            self.logToScreen('请先打开视频文件！')
            return
        self.timer.start(
            int(1000.0 / (float(self.ui.fpsSpinBox.text()) + 1e-5)))
        self.isPlaying = True
        self.lastFrameTime = time.time()
        self.logToScreen('开始播放')

    def onPause(self):
        self.isPlaying = False
        self.timer.stop()
        self.logToScreen('停止播放')

    def onAbout(self):
        self.aboutDialog.show()

    def onTimerTimeout(self):
        self.nextFrame()

    def detectObjects(self, frame):
        detections = []
        nms_thres = float(self.ui.detNMSSpinBox.text())
        device = 'cuda:0' if self.ui.detDeviceComboBox.currentIndex(
        ) == 0 else 'cpu'
        # half = self.ui.detHalfPrecCheckBox.isChecked()
        results = self.detector.predict(
            frame,
            device=device,
            iou=nms_thres,
            # half=half,
        )
        frameWidth, frameHeight = frame.shape[1], frame.shape[0]
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy_data = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy_data.astype(int)
                    x1, y1, x2, y2 = float(x1) / frameWidth, float(
                        y1) / frameHeight, float(x2) / frameWidth, float(
                            y2) / frameHeight

                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())

                    detections.append((x1, y1, x2, y2, conf, cls_id))
        return detections

    def drawDetections(self, frame, detections):
        frameWidth, frameHeight = frame.shape[1], frame.shape[0]
        for detection in detections:
            x1, y1, x2, y2, conf, cls_id = detection
            x1, y1, x2, y2 = x1 * frameWidth, y1 * frameHeight, x2 * frameWidth, y2 * frameHeight
            if conf < float(self.ui.detConfSpinBox.text()):
                continue
            class_name, class_color = 'Unknown', (0, 0, 0)
            # 获取类别名称:
            if cls_id in range(len(self.CLASS_NAME)):
                class_name = self.CLASS_NAME[cls_id]
                class_color = self.CLASS_PALETTE[cls_id]

                if not self.classMask[cls_id]:
                    continue

            # 画矩形框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          class_color,
                          2,
                          lineType=cv2.LINE_AA)

            # 绘制类别标签和置信度
            label = f"{class_name}: {conf:.2f}"
            # cv2.putText(frame, label, (int(x1), int(y1 - 10)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.5, 1)
            text_bg_x1 = x1-1
            text_bg_y1 = y1 - text_h - 8
            text_bg_x2 = x1 + text_w
            text_bg_y2 = y1
            # 绘制文字底纹
            cv2.rectangle(frame, (int(text_bg_x1), int(text_bg_y1)),
                          (int(text_bg_x2), int(text_bg_y2)),
                          class_color,
                          -1,
                          lineType=cv2.LINE_AA)
            # 在底纹上绘制文字
            cv2.putText(frame, label, (int(x1), int(y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        tuple((255 - np.array(class_color)).tolist()), 1,
                        cv2.LINE_AA)

        return frame

    def refreshCurrentFrame(self):
        if self.capture is not None:
            ret, frame = self.capture.retrieve()
            if ret:
                self.updateFrame(frame)

    def openVideo(self, fileName):
        capture = cv2.VideoCapture(filename=fileName)
        if not capture.isOpened():
            self.logToScreen(f'打开视频 {fileName} 失败')
            return False
        self.logToScreen(f'打开视频 {fileName} 成功')

        self.capture = capture
        self.ui.playSlider.setRange(0, self.getVideoFrameCount() - 1)
        return True

    def setVideoFramePos(self, frameIndex):
        min_val, max_val = self.ui.playSlider.minimum(
        ), self.ui.playSlider.maximum()
        frameIndex = min_val if frameIndex < min_val else frameIndex
        frameIndex = max_val if frameIndex > max_val else frameIndex
        self.ui.playSlider.setValue(frameIndex)

    def getVideoFramePos(self):
        if self.capture is not None and self.capture.isOpened():
            current_frame_index = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
            return current_frame_index
        return -1

    def getVideoFrameCount(self):
        if self.capture is not None and self.capture.isOpened():
            totalFrames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            return totalFrames
        return 0

    def imageProcess(self, frame):
        enableGray = self.ui.grayCheckBox.isChecked()
        if enableGray:
            frame = image_gray(frame)

        enableHistEq = self.ui.histEqGroupBox.isChecked()
        clitLimit = int(self.ui.histLimitSpinBox.text())
        tileGridSize = int(self.ui.tileSizeSpinBox.text())
        if enableHistEq:
            frame = image_equalize_hist(frame,
                                        method='adaptive',
                                        clip_limit=clitLimit,
                                        tile_grid_size=(tileGridSize,
                                                        tileGridSize))

        return frame

    def getRealFPS(self, lastTime):
        return 1.0 / (time.time() - lastTime)

    def updateFrame(self, newframe, forceNoDetection=False):
        try:
            frame = newframe
            dets = None
            if self.enabelImageProcess:
                frame = self.imageProcess(frame)
            if self.enableDetection and not forceNoDetection:
                dets = self.detectObjects(frame)

            scaleFactor = float(self.ui.scaleSpinBox.text())
            frame = cv2.resize(frame, (int(frame.shape[1] / scaleFactor),
                                       int(frame.shape[0] / scaleFactor)))

            if self.enableDetection and not forceNoDetection:
                self.drawDetections(frame, dets)

            pixmap = QtGui.QPixmap.fromImage(utils.cv2QImage(frame))

            self.scene.clear()
            self.scene.addPixmap(pixmap)

            # self.ui.graphicsView.fitInView(self.scene.itemsBoundingBox(),
            #                                Qt.AspectRatioFixed)

            self.ui.realFPSLabel.setText(
                f"{self.getRealFPS(self.lastFrameTime):.3f}")
            self.lastFrameTime = time.time()

            if self.histogramDialog.isVisible():
                self.histogramDialog.plot(image_histogram(
                    frame, (255, 222, 2)))
        except BaseException as e:
            traceback_info = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__))

            QMessageBox.critical(None, '错误',
                                 f"发生错误: {str(e)}\n\n堆栈跟踪:\n{traceback_info}",
                                 QMessageBox.StandardButton.Ok)

    def prevFrame(self):
        if not (self.capture is not None and self.capture.isOpened()):
            self.logToScreen('Video capture is None or closed')
            return
        self.setVideoFramePos(self.getVideoFramePos() - 1)

    def nextFrame(self):
        if not (self.capture is not None and self.capture.isOpened()):
            self.logToScreen('Video capture is None or closed')
            return
        self.setVideoFramePos(self.getVideoFramePos() + 1)

    def onNextFrame(self):
        self.nextFrame()
        self.logToScreen('下一帧')

    def onPrevFrame(self):
        self.prevFrame()
        self.logToScreen('上一帧')

    def onGrayCheckBoxToggled(self):
        pass

    def onHistEqGroupBoxToggled(self):
        pass

    def onHistogramDialogClosed(self):
        self.ui.showHistogramCheckBox.setCheckState(Qt.CheckState.Unchecked)
        self.logToScreen('关闭直方图')

    def onShowHistogramCheckBoxToggled(self):
        if self.ui.showHistogramCheckBox.isChecked():
            self.histogramDialog.show()
            self.logToScreen('打开直方图')
        else:
            self.histogramDialog.hide()
            self.logToScreen('关闭直方图')

    def onDetGroupBoxToggled(self):
        self.enableDetection = self.ui.detGroupBox.isChecked()
        self.logToScreen(f'启用目标检测 = {self.enableDetection}')

    def onImageProcGroupBoxToggled(self):
        self.enabelImageProcess = self.ui.imgProcGroupBox.isChecked()
        self.logToScreen(f'启用图像处理 = {self.enabelImageProcess}')

    def onPlaySliderPressed(self):
        try:
            self.fineTuning = True
            if self.isPlaying:
                self.timer.stop()
        except BaseException as e:
            traceback_info = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__))
            self.logToScreen(f"发生错误: {str(e)}\n\n堆栈跟踪:\n{traceback_info}")

    def onPlaySliderReleased(self):
        try:
            self.fineTuning = False
            if self.isPlaying:
                self.timer.start(
                    int(1000.0 / (float(self.ui.fpsSpinBox.text()) + 1e-5)))
                self.lastFrameTime = time.time()
            else:
                self.refreshCurrentFrame()
        except BaseException as e:
            traceback_info = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__))
            self.logToScreen(f"发生错误: {str(e)}\n\n堆栈跟踪:\n{traceback_info}")

    def onPlaySliderValueChanged(self):
        try:
            if self.capture is not None and self.capture.isOpened():
                totalFrames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                frameIndex = self.ui.playSlider.value()
                frameIndex = 0 if frameIndex < 0 else frameIndex
                frameIndex = totalFrames - 1 if frameIndex >= totalFrames else frameIndex
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                ret, frame = self.capture.retrieve()
                self.updateFrame(frame, self.fineTuning)
                self.ui.playLabel.setText(
                    f"{self.ui.playSlider.value()}/{self.ui.playSlider.maximum()}"
                )
        except BaseException as e:
            traceback_info = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__))
            self.logToScreen(f"发生错误: {str(e)}\n\n堆栈跟踪:\n{traceback_info}")

    def onPlaySliderRangeChanged(self):
        self.ui.playLabel.setText(
            f"{self.ui.playSlider.value()}/{self.ui.playSlider.maximum()}")

    def logToScreen(self, text):
        self.ui.outputTextBroser.append(text + "\n")

    def write(self, text):
        self.logToScreen(text)

    def flush(self):
        pass

    # def onChangeToLightTheme(self):
    #     utils.apply_style_sheet(self, './theme/legacy/MacOS.qss')
    #     from qt_material import apply_stylesheet

    # def onChangeToDarkTheme(self):
    #     utils.apply_style_sheet(self, './theme/legacy/MaterialDark.qss')
