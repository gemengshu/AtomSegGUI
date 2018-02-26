#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from Atom_Seg_Ui import Ui_MainWindow
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
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
from utils import load_model1, load_model2, load_model3, GetIndexRangeOfBlk

def PILImageToQImage(img):
    """Convert PIL image to QImage """
    imageq = ImageQt(img)
    qimage = QImage(imageq)
    return qimage

def map01(mat):
    return (mat - mat.min())/(mat.max() - mat.min())

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

        self.__curdir = os.getcwd()

        self.ori_content = None
        self.output_image = None
        self.ori_markers = None
        self.out_markers = None

        self.__models = {
                'Model 1' : self.__load_model1,
                'Model 2' : self.__load_model2,
                'Model 3' : self.__load_model3
        }


    def BrowseFolder(self):
        self.imagePath_content, _ = QFileDialog.getOpenFileName(self,
                                                            "open",
                                                            "/home/",
                                                            "All Files (*);; Image Files (*.png *.tif *.jpg)")
        if self.imagePath_content:
            self.imagePath.setText(self.imagePath_content)
            self.ori_content = Image.open(self.imagePath_content).convert('L')
            self.height, self.width = self.ori_content.size
            ori_content_qt = PILImageToQImage(self.ori_content)
            pix_image = QPixmap(ori_content_qt)
            self.ori.setPixmap(pix_image)
            self.ori.show()

    def __load_model1(self):
        if not self.ori_content:
            raise Exception("No image is selected.")
        self.cuda = self.use_cuda.isChecked()
        model_path = self.__curdir + '/model1.pth'

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
                temp_result = load_model1(model_path,
                                          temp_image, self.cuda)
                result[outer_blk[1] : outer_blk[3], outer_blk[0] : outer_blk[2]] = np.maximum(temp_result,
                                          result[outer_blk[1] : outer_blk[3], outer_blk[0] : outer_blk[2]])

        self.model_output_content = map01(result)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype('uint8')
        self.output_image = Image.fromarray((self.model_output_content), mode = 'L')
        ori_content_qt = PILImageToQImage(self.output_image)
        pix_image = QPixmap(ori_content_qt)
        self.model_output.setPixmap(pix_image)
        self.model_output.show()


    def __load_model2(self):
        if not self.ori_content:
            raise Exception("No image is selected.")
        self.cuda = self.use_cuda.isChecked()
        model_path = self.__curdir + '/model2.pth'

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
                temp_result = load_model2(model_path,
                                          temp_image, self.cuda)
                result[outer_blk[1] : outer_blk[3], outer_blk[0] : outer_blk[2]] = np.maximum(temp_result,
                                          result[outer_blk[1] : outer_blk[3], outer_blk[0] : outer_blk[2]])

        self.model_output_content = map01(result)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype('uint8')
        self.output_image = Image.fromarray((self.model_output_content), mode = 'L')
        ori_content_qt = PILImageToQImage(self.output_image)
        pix_image = QPixmap(ori_content_qt)
        self.model_output.setPixmap(pix_image)
        self.model_output.show()

    def __load_model3(self):
        if not self.ori_content:
            raise Exception("No image is selected.")
        self.cuda = self.use_cuda.isChecked()
        model_path = self.__curdir + '/model3.pth'

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
                temp_result = load_model3(model_path,
                                          temp_image, self.cuda)
                result[outer_blk[1] : outer_blk[3], outer_blk[0] : outer_blk[2]] = np.maximum(temp_result,
                                          result[outer_blk[1] : outer_blk[3], outer_blk[0] : outer_blk[2]])

        self.model_output_content = map01(result)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype('uint8')
        self.output_image = Image.fromarray((self.model_output_content), mode = 'L')
        ori_content_qt = PILImageToQImage(self.output_image)
        pix_image = QPixmap(ori_content_qt)
        self.model_output.setPixmap(pix_image)
        self.model_output.show()

    def LoadModel(self):

        self.modelPath_content = self.modelPath.currentText()
        self.__models[self.modelPath_content]()

    def Denoise(self):
        radius = self.se_num.value()
        """changes should be done on the kernel generation"""
        kernel = disk(radius)

        opened_image = opening(self.model_output_content, kernel)

        self.denoised_image = opened_image

        temp_image = Image.fromarray(self.denoised_image, mode = 'L')

        ori_content_qt = PILImageToQImage(temp_image)
        pix_image = QPixmap(ori_content_qt)
        self.preprocess.setPixmap(pix_image)
        self.preprocess.show()


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

        ori_content_qt = PILImageToQImage(self.out_markers)
        pix_image = QPixmap(ori_content_qt)
        self.detect_result.setPixmap(pix_image)
        self.detect_result.show()

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

        ori_content_qt = PILImageToQImage(self.out_markers)
        pix_image = QPixmap(ori_content_qt)
        self.detect_result.setPixmap(pix_image)
        self.detect_result.show()

    def RevertAll(self):
        self.model_output.clear()
        self.preprocess.clear()
        self.detect_result.clear()
        self.se_num.setValue(0)
        self.min_thre.setValue(30)
        self.max_thre.setValue(150)

    def Save(self):


        if os.name == 'posix':
            file_name = self.imagePath_content.split('/')[-1]
        elif os.name == 'nt':
            file_name = self.imagePath_content.split('\\')[-1]
        else:
            raise Exception("Not supported system.")

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
            new_save_name = save_path + '/' + name_no_suffix + self.modelPath_content + suffix
        else:
            new_save_name = save_path + '\\' + name_no_suffix + self.modelPath_content + suffix

        im_save = Image.new('RGB', ((self.width + 1) * 2, (self.height + 1) * 2))
        im_save.paste(self.ori_content, (0,0))
        im_save.paste(self.output_image, (self.width + 2, 0))
        im_save.paste(self.ori_markers, (0, self.height + 2))
        im_save.paste(self.out_markers, (self.width + 2, self.height + 2))
        im_save.save(new_save_name)




    def release(self):
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
