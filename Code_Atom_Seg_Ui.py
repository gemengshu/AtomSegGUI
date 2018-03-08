#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from Atom_Seg_Ui import Ui_MainWindow
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
import torch
import os
from os.path import exists, join
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import numpy as np
import torchvision
from skimage.morphology import opening, watershed, disk
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.draw import set_color
from utils import GetIndexRangeOfBlk, load_model, PIL2Pixmap, map01


class Code_MainWindow(Ui_MainWindow):
    def __init__(self, parent = None):
        super(Code_MainWindow, self).__init__()

        self.setupUi(self)
        self.open.clicked.connect(self.BrowseFolder)
        self.load.clicked.connect(self.LoadModel)
        self.se_num.valueChanged.connect(self.Denoise)
        self.min_thre.valueChanged.connect(self.ChangeThreshold)
        self.max_thre.valueChanged.connect(self.ChangeThreshold)
        self.circle_detect.clicked.connect(self.CircleDetect)

        self.revert.clicked.connect(self.RevertAll)

        self.save.clicked.connect(self.Save)

#        self.ori.mousePressEvent = self.drawPoint
#        self.model_output.mousePressEvent = self.drawPoint
#        self.preprocess.mousePressEvent = self.drawPoint
#        self.detect_result.mousePressEvent = self.drawPoint

        self.__curdir = os.getcwd() #current directory

        self.ori_content = None  #original image, PIL format
        self.output_image = None #output image of model, PIL format
        self.ori_markers = None  #for saving usage, it's a rgb image of original, and with detection result on it
        self.out_markers = None  #for saving usage, it's a rgb image of result after denoising, and with detection result on it
        self.model_output_content = None # 2d array of model output

        self.pos = None

        self.denoised_image = None

        self.__models = {
                'Model 1' : 1,
                'Model 2' : 2,
                'Model 3' : 3,
                'Model 4' : 4,
                'Model 5' : 5,
                'Model 6' : 6
        }

        self.__model_paths = ['/model1.pth',
                              '/model2.pth',
                              '/model3.pth',
                              '/atomseg_bupt_new_10/model_epoch_200.pth',
                              '/atomseg_bupt_new_100/model_epoch_200.pth',
                              '/atom_seg_gaussian_mask/model_epoch_200.pth']


    def BrowseFolder(self):
        self.imagePath_content, _ = QFileDialog.getOpenFileName(self,
                                                            "open",
                                                            "/home/",
                                                            "All Files (*);; Image Files (*.png *.tif *.jpg)")
        if self.imagePath_content:
            self.imagePath.setText(self.imagePath_content)
            self.ori_content = Image.open(self.imagePath_content).convert('L')
            self.height, self.width = self.ori_content.size
            pix_image = PIL2Pixmap(self.ori_content)
            self.ori.setPixmap(pix_image)
            self.ori.show()

    def __load_model(self):
        if not self.ori_content:
            raise Exception("No image is selected.")
        self.cuda = self.use_cuda.isChecked()
        model_path = self.__curdir + self.__model_paths[self.model_num - 1]

        if self.height > 1000 and self.height < 1600:
            blk_row = 2
        elif self.height >1600:
            blk_row = 4
        else:
            blk_row = 1

        if self.width > 1000 and self.width < 1600:
            blk_col = 2
        elif self.height >1600:
            blk_col = 4
        else:
            blk_col = 1
        result = np.zeros((self.width, self.height)) - 100

        for r in range(0, blk_row):
            for c in range(0, blk_col):

                inner_blk, outer_blk = GetIndexRangeOfBlk(self.height, self.width, blk_row, blk_col, r,c, over_lap = 2)
                temp_image = self.ori_content.crop((outer_blk[0], outer_blk[1], outer_blk[2], outer_blk[3]))
                temp_result = load_model(model_path, self.model_num, temp_image, self.cuda)
                result[outer_blk[1] : outer_blk[3], outer_blk[0] : outer_blk[2]] = np.maximum(temp_result,
                                          result[outer_blk[1] : outer_blk[3], outer_blk[0] : outer_blk[2]])

        self.model_output_content = map01(result)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype('uint8')
        self.output_image = Image.fromarray((self.model_output_content), mode = 'L')
        pix_image = PIL2Pixmap(self.output_image)
        self.model_output.setPixmap(pix_image)
        self.model_output.show()
        del temp_image
        del temp_result
        del result # free memory caused by temporary matrix-result

    def LoadModel(self):

        self.modelPath_content = self.modelPath.currentText()
        self.model_num = self.__models[self.modelPath_content]

        self.__load_model()

        self.Denoise()

    def Denoise(self):
        radius = self.se_num.value()
        """changes should be done on the kernel generation"""
        kernel = disk(radius)

        self.denoised_image = opening(self.model_output_content, kernel)

        temp_image = Image.fromarray(self.denoised_image, mode = 'L')

        pix_image = PIL2Pixmap(temp_image)
        self.preprocess.setPixmap(pix_image)
        self.preprocess.show()
        del temp_image


    def CircleDetect(self):
        elevation_map = sobel(self.denoised_image)
        markers = np.zeros_like(self.denoised_image)

        markers[self.denoised_image < 30] = 1
        markers[self.denoised_image > 150] = 2

        seg_1 = watershed(elevation_map, markers)

        filled_regions = ndi.binary_fill_holes(seg_1 - 1)

        label_objects, nb_labels = ndi.label(filled_regions)

        self.props = regionprops(label_objects)

        self.out_markers = Image.fromarray(np.dstack((self.denoised_image,self.denoised_image,self.denoised_image)), mode = 'RGB')

        ori_array = np.array(self.ori_content)
        self.ori_markers = Image.fromarray(np.dstack((ori_array,ori_array,ori_array)), mode = 'RGB')

        del elevation_map
        del markers, seg_1,filled_regions,label_objects, nb_labels
        del ori_array

        draw_out = ImageDraw.Draw(self.out_markers)
        draw_ori = ImageDraw.Draw(self.ori_markers)

        for p in self.props:
            c_y, c_x = p.centroid
            draw_out.ellipse([min([max([c_x - 2, 0]), self.width]),min([max([c_y - 2, 0]), self.height]),
                min([max([c_x + 2, 0]), self.width]),min([max([c_y + 2, 0]), self.height])],
                fill = 'red', outline = 'red')
            draw_ori.ellipse([min([max([c_x - 2, 0]), self.width]),min([max([c_y - 2, 0]), self.height]),
                min([max([c_x + 2, 0]), self.width]),min([max([c_y + 2, 0]), self.height])],
                fill = 'red', outline = 'red')

        pix_image = PIL2Pixmap(self.out_markers)
        self.detect_result.setPixmap(pix_image)
        self.detect_result.show()
#        del props

    def ChangeThreshold(self):
        min_thre_content = self.min_thre.value()
        max_thre_content = self.max_thre.value()

        elevation_map = sobel(self.denoised_image)

        markers = np.zeros_like(self.denoised_image)

        markers[self.denoised_image < min_thre_content] = 1
        markers[self.denoised_image > max_thre_content] = 2

        seg_1 = watershed(elevation_map, markers)

        filled_regions = ndi.binary_fill_holes(seg_1 - 1)
        label_objects, nb_labels = ndi.label(filled_regions)

        self.props = regionprops(label_objects)

        self.out_markers = Image.fromarray(np.dstack((self.denoised_image,self.denoised_image,self.denoised_image)), mode = 'RGB')

        ori_array = np.array(self.ori_content)
        self.ori_markers = Image.fromarray(np.dstack((ori_array,ori_array,ori_array)), mode = 'RGB')

        del ori_array
        del elevation_map
        del markers, seg_1,filled_regions,label_objects, nb_labels
        del ori_array

        draw_out = ImageDraw.Draw(self.out_markers)
        draw_ori = ImageDraw.Draw(self.ori_markers)

        for p in self.props:
            c_y, c_x = p.centroid

            draw_out.ellipse([min([max([c_x - 2, 0]), self.width]),min([max([c_y - 2, 0]), self.height]),
                min([max([c_x + 2, 0]), self.width]),min([max([c_y + 2, 0]), self.height])],
                fill = 'red', outline = 'red')
            draw_ori.ellipse([min([max([c_x - 2, 0]), self.width]),min([max([c_y - 2, 0]), self.height]),
                min([max([c_x + 2, 0]), self.width]),min([max([c_y + 2, 0]), self.height])],
                fill = 'red', outline = 'red')

        pix_image = PIL2Pixmap(self.out_markers)
        self.detect_result.setPixmap(pix_image)
        self.detect_result.show()
#        del props

    def RevertAll(self):
        self.model_output.clear()
        self.se_num.setValue(0)
        self.preprocess.clear()
        self.min_thre.setValue(30)
        self.detect_result.clear()
        self.max_thre.setValue(150)
        del self.props

    def GetSavePath(self):

        file_name = self.imagePath_content.split('/')[-1]
        suffix = '.' + file_name.split('.')[-1]
        name_no_suffix = file_name.replace(suffix, '')

        if self.auto_save.isChecked():
            if os.name == 'posix':
                save_path = self.__curdir + '/' + name_no_suffix
            else:
                save_path = self.__curdir + '\\' + name_no_suffix
        else:
            if os.name == 'posix':
                path = QFileDialog.getExistingDirectory(self, "save", "/home",
                                                            QFileDialog.ShowDirsOnly
                                                            | QFileDialog.DontResolveSymlinks)
                save_path = path + '/' + name_no_suffix
            else:
                path = QFileDialog.getExistingDirectory(self, "save", self.__curdir,
                                                        QFileDialog.ShowDirsOnly
                                                        | QFileDialog.DontResolveSymlinks)
                save_path = path + '\\' + name_no_suffix

        if not exists(save_path):
            os.mkdir(save_path)

        if os.name == 'posix':
            temp_path = save_path + '/' + name_no_suffix
        else:
            temp_path = save_path + '\\' + name_no_suffix

        return temp_path, suffix

    def Save(self):
        opt = self.save_option.currentText()
        _path, suffix = self.GetSavePath()
        if opt == 'Model output':
            new_save_name = _path + '_output_' + self.modelPath_content + suffix
            self.output_image.save(new_save_name)

        if opt == 'Original image with markers':
            new_save_name = _path + '_origin_' + self.modelPath_content + suffix
            self.ori_markers.save(new_save_name)

        if opt == 'Four-panel image':
            new_save_name = _path + '_four_panel_' + self.modelPath_content + suffix
            im_save = Image.new('RGB', ((self.width + 1) * 2, (self.height + 1) * 2))
            im_save.paste(self.ori_content, (0,0))
            im_save.paste(self.output_image, (self.width + 2, 0))
            im_save.paste(self.ori_markers, (0, self.height + 2))
            im_save.paste(self.out_markers, (self.width + 2, self.height + 2))
            im_save.save(new_save_name)
            del im_save

        if opt == 'Atom positions':
            new_save_name = _path + '_pos_' + self.modelPath_content + '.txt'
            file = open(new_save_name, 'w')
            for p in self.props:
                c_y, c_x = p.centroid
                min_row, min_col, max_row, max_col = p.bbox
                file.write( "%s"%((c_y, c_x, min_row, min_col, max_row, max_col),))
                file.write("\n")
            file.close()

        if opt == 'Save ALL':
            new_save_name = _path + '_output_' + self.modelPath_content + suffix
            self.output_image.save(new_save_name)
            new_save_name = _path + '_origin_' + self.modelPath_content + suffix
            self.ori_markers.save(new_save_name)
            new_save_name = _path + '_four_panel_' + self.modelPath_content + suffix
            im_save = Image.new('RGB', ((self.width + 1) * 2, (self.height + 1) * 2))
            im_save.paste(self.ori_content, (0,0))
            im_save.paste(self.output_image, (self.width + 2, 0))
            im_save.paste(self.ori_markers, (0, self.height + 2))
            im_save.paste(self.out_markers, (self.width + 2, self.height + 2))
            im_save.save(new_save_name)
            del im_save
            new_save_name = _path + '_pos_' + self.modelPath_content + '.txt'
            file = open(new_save_name, 'w')
            for p in self.props:
                c_y, c_x = p.centroid
                min_row, min_col, max_row, max_col = p.bbox
                file.write( "%s"%((c_y, c_x, min_row, min_col, max_row, max_col),))
                file.write("\n")
            file.close()


    def drawPoint(self, event):
        self.pos = event.pos()
        self.update()

 #   def paintEvent(self, event):
 #       if self.pos:
 #           p = QPainter()
 #           p.begin(self)
 #           p.setBrush(QtGui.QColor(0,255,0))
 #           p.drawEllipse(self.ori.mapToParent(QtCore.QPoint(self.pos.x(), self.pos.y())), 5,5)
 #           self.centralwidget.raise_()
 #           p.end()


    def release(self):
        self.model_output.clear()
        self.se_num.setValue(0)
        self.preprocess.clear()
        self.min_thre.setValue(30)
        self.detect_result.clear()
        self.max_thre.setValue(150)
        self.ori.clear()
        del self.props
        del self.output_image
        del self.ori_markers
        del self.out_markers
        return


    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self,
                                            "Confirm Exit...",
                                            "Are you sure you want to exit?",
                                            QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No)
        event.ignore()

        if result == QtWidgets.QMessageBox.Yes:
            self.release()
            event.accept()


qtCreatorFile = "AtomSeg_V1.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Code_MainWindow()
    window.show()
    sys.exit(app.exec_())
