from PyQt5 import QtCore, QtGui, QtWidgets, uic

import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter

class DrawPointLabel(QtWidgets.QLabel):
	def __init__(self, parent = None):
		QtWidgets.QLabel.__init__(self, parent)
#		super(DrawPointLabel, self).__init__(parent)
		self.pos = None

	def paintEvent(self, event):
		if self.pos:
			super().paintEvent(event)
			qp = QPainter(self)
			qp.setBrush(QtGui.QColor(0,255,0))
			qp.drawEllipse(QtCore.QPoint(self.pos.x(), self.pos.y()), 5, 5)