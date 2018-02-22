#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from Atom_Seg_Ui import Ui_MainWindow
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image
from PIL.ImageQt import ImageQt
import torch
import os
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import numpy as np
import torchvision
from utils import generateMask

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
        self.cuda = True

        self.__curdir = os.getcwd()

        self.ori_content = None

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
            ori_content_qt = PILImageToQImage(self.ori_content)
            pix_image = QPixmap(ori_content_qt)
            self.ori.setPixmap(pix_image)
            self.ori.show()

    def __load_model1(self):
        """/home/student/Documents/u-net_pytorch/epochs200_layer3_ori_256/"""
        from model1 import UNet
        unet = UNet()
        model_path = self.__curdir + '/model1.pth'
        if self.cuda:
            unet = unet.cuda()

        if not self.cuda:
            unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            unet.load_state_dict(torch.load(model_path))
        if not self.ori_content:
            raise Exception("No image is selected.")
        transform = ToTensor()
        ori_tensor = transform(self.ori_content)
        if self.cuda:
            ori_tensor = Variable(ori_tensor.cuda())
        else:
            ori_tensor = Variable(ori_tensor)
        ori_tensor = torch.unsqueeze(ori_tensor,0)
        output = unet(ori_tensor)

        if self.cuda:
            self.model_output_content = (output.data).cpu().numpy()
        else:
            self.model_output_content = (output.data).numpy()

        self.model_output_content = self.model_output_content[0,0,:,:]

        self.model_output_content = map01(self.model_output_content)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype('uint8')
        output_image = Image.fromarray((self.model_output_content), mode = 'L')

        ori_content_qt = PILImageToQImage(output_image)
        pix_image = QPixmap(ori_content_qt)
        self.model_output.setPixmap(pix_image)
        self.model_output.show()


    def __load_model2(self):
        """/home/student/Documents/u-net-pytorch-original/lr001_weightdecay00001/"""
        from model2 import UNet
        unet = UNet()
        model_path = self.__curdir + '/model2.pth'
        if self.cuda:
            unet = unet.cuda()

        if not self.cuda:
            unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            unet.load_state_dict(torch.load(model_path))
        if not self.ori_content:
            raise Exception("No image is selected.")
        transform = ToTensor()
        ori_tensor = transform(self.ori_content)
        if self.cuda:
            ori_tensor = Variable(ori_tensor.cuda())
        else:
            ori_tensor = Variable(ori_tensor)
        ori_tensor = torch.unsqueeze(ori_tensor,0)
        output = unet(ori_tensor)

        if self.cuda:
            self.model_output_content = (output.data).cpu().numpy()
        else:
            self.model_output_content = (output.data).numpy()

        self.model_output_content = self.model_output_content[0,0,:,:]

        self.model_output_content = map01(self.model_output_content)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype('uint8')
        output_image = Image.fromarray((self.model_output_content), mode = 'L')

        ori_content_qt = PILImageToQImage(output_image)
        pix_image = QPixmap(ori_content_qt)
        self.model_output.setPixmap(pix_image)
        self.model_output.show()


    def __load_model3(self):
        """/home/student/Documents/u-net_denoising/dataset_small_mask/"""
        from model3 import UNet
        unet = UNet()
        model_path = self.__curdir + '/model3.pth'
        if self.cuda:
            unet = unet.cuda()

        if not self.cuda:
            unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            unet.load_state_dict(torch.load(model_path))
        if not self.ori_content:
            raise Exception("No image is selected.")
        transform = ToTensor()
        ori_tensor = transform(self.ori_content)
        if self.cuda:
            ori_tensor = Variable(ori_tensor.cuda())
        else:
            ori_tensor = Variable(ori_tensor)
        ori_tensor = torch.unsqueeze(ori_tensor,0)
        output = unet(ori_tensor)

        if self.cuda:
            self.model_output_content = (output.data).cpu().numpy()
        else:
            self.model_output_content = (output.data).numpy()

        self.model_output_content = self.model_output_content[0,0,:,:]

        self.model_output_content = map01(self.model_output_content)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype('uint8')
        output_image = Image.fromarray((self.model_output_content), mode = 'L')

        ori_content_qt = PILImageToQImage(output_image)
        pix_image = QPixmap(ori_content_qt)
        self.model_output.setPixmap(pix_image)
        self.model_output.show()



    def LoadModel(self):

        self.modelPath_content = self.modelPath.currentText()
        self.__models[self.modelPath_content]()


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
