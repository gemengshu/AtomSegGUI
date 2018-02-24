import cv2
import numpy as np


def map01(data):
	return (data - data.min())/(data.max() - data.min())