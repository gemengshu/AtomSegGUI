import torch
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import numpy as np
import torchvision

def GetIndexRangeOfBlk(height, width, blk_row, blk_col, blk_r, blk_c, over_lap = 0):
	blk_h_size = height//blk_row
	blk_w_size = width//blk_col

	if blk_r >= blk_row or blk_c >= blk_col:
		raise Exception("index is out of range...")

	upper_left_r = blk_r * blk_h_size
	upper_left_c = blk_c * blk_w_size
	ol_upper_left_r = max(upper_left_r - over_lap, 0)
	ol_upper_left_c = max(upper_left_c - over_lap, 0)

	if blk_r == (blk_row - 1):
		lower_right_r = height
		ol_lower_right_r = lower_right_r
	else:
		lower_right_r = upper_left_r + blk_h_size
		ol_lower_right_r = min(lower_right_r + over_lap, height)

	if blk_c == (blk_col - 1):
		lower_right_c = width
		ol_lower_right_c = lower_right_c
	else:
		lower_right_c = upper_left_c + blk_w_size
		ol_lower_right_c = min(lower_right_c + over_lap, height)

	return (upper_left_r, upper_left_c, lower_right_r, lower_right_c), (ol_upper_left_r, ol_upper_left_c, ol_lower_right_r, ol_lower_right_c)

def load_model1(model_path, data, cuda):
	"""/home/student/Documents/u-net_pytorch/epochs200_layer3_ori_256/"""
	from model1 import UNet
	use_padding = False
	unet = UNet()
	if cuda:
		unet = unet.cuda()

	if not cuda:
		unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
	else:
		unet.load_state_dict(torch.load(model_path))

	transform = ToTensor()
	ori_tensor = transform(data)
	if cuda:
		ori_tensor = Variable(ori_tensor.cuda())
	else:
		ori_tensor = Variable(ori_tensor)
	ori_tensor = torch.unsqueeze(ori_tensor,0)

	padding_left = 0
	padding_right = 0
	padding_top = 0
	padding_bottom = 0
	ori_height = ori_tensor.size()[2]
	ori_width = ori_tensor.size()[3]
	if ori_height % 4:
		padding_top = (4 - ori_height % 4)//2
		padding_bottom = 4 - ori_height % 4 - padding_top
		use_padding = True
	if ori_width % 4:
		padding_left = (4 - ori_width % 4)//2
		padding_right = 4 - ori_width % 4 - padding_left
		use_padding = True
	if use_padding:
		padding_transform = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
		ori_tensor = padding_transform(ori_tensor)

	output = unet(ori_tensor)

	if use_padding:
		output = output[:,:,padding_top : (padding_top + ori_height), padding_left : (padding_left + ori_width)]

	if cuda:
		result = (output.data).cpu().numpy()
	else:
		result = (output.data).numpy()

	result = result[0,0,:,:]
	return result

def load_model2(model_path, data, cuda):
	"""/home/student/Documents/u-net-pytorch-original/lr001_weightdecay00001/"""
	from model2 import UNet
	use_padding = False
	unet = UNet()
	if cuda:
		unet = unet.cuda()

	if not cuda:
		unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
	else:
		unet.load_state_dict(torch.load(model_path))

	transform = ToTensor()
	ori_tensor = transform(data)
	if cuda:
		ori_tensor = Variable(ori_tensor.cuda())
	else:
		ori_tensor = Variable(ori_tensor)
	ori_tensor = torch.unsqueeze(ori_tensor,0)

	padding_left = 0
	padding_right = 0
	padding_top = 0
	padding_bottom = 0
	ori_height = ori_tensor.size()[2]
	ori_width = ori_tensor.size()[3]
	if ori_height % 4:
		padding_top = (4 - ori_height % 4)//2
		padding_bottom = 4 - ori_height % 4 - padding_top
		use_padding = True
	if ori_width % 4:
		padding_left = (4 - ori_width % 4)//2
		padding_right = 4 - ori_width % 4 - padding_left
		use_padding = True
	if use_padding:
		padding_transform = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
		ori_tensor = padding_transform(ori_tensor)

	output = unet(ori_tensor)

	if use_padding:
		output = output[:,:,padding_top : (padding_top + ori_height), padding_left : (padding_left + ori_width)]

	if cuda:
		result = (output.data).cpu().numpy()
	else:
		result = (output.data).numpy()

	result = result[0,0,:,:]
	return result

def load_model3(model_path, data, cuda):
	"""/home/student/Documents/u-net_denoising/dataset_small_mask/"""
	from model3 import UNet
	use_padding = False
	unet = UNet()
	if cuda:
		unet = unet.cuda()

	if not cuda:
		unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
	else:
		unet.load_state_dict(torch.load(model_path))

	transform = ToTensor()
	ori_tensor = transform(data)
	if cuda:
		ori_tensor = Variable(ori_tensor.cuda())
	else:
		ori_tensor = Variable(ori_tensor)
	ori_tensor = torch.unsqueeze(ori_tensor,0)

	padding_left = 0
	padding_right = 0
	padding_top = 0
	padding_bottom = 0
	ori_height = ori_tensor.size()[2]
	ori_width = ori_tensor.size()[3]
	if ori_height % 4:
		padding_top = (4 - ori_height % 4)//2
		padding_bottom = 4 - ori_height % 4 - padding_top
		use_padding = True
	if ori_width % 4:
		padding_left = (4 - ori_width % 4)//2
		padding_right = 4 - ori_width % 4 - padding_left
		use_padding = True
	if use_padding:
		padding_transform = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
		ori_tensor = padding_transform(ori_tensor)

	output = unet(ori_tensor)

	if use_padding:
		output = output[:,:,padding_top : (padding_top + ori_height), padding_left : (padding_left + ori_width)]

	if cuda:
		result = (output.data).cpu().numpy()
	else:
		result = (output.data).numpy()

	result = result[0,0,:,:]
	return result


def load_model4(model_path, data, cuda):
	"""/home/student/Documents/Atom Segmentation APP/AtomSegGUI/atomseg_bupt_new_10/"""
	from model4 import UNet
	use_padding = False
	unet = UNet()
	if cuda:
		unet = unet.cuda()

	if not cuda:
		unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
	else:
		unet.load_state_dict(torch.load(model_path))

	transform = ToTensor()
	ori_tensor = transform(data)
	if cuda:
		ori_tensor = Variable(ori_tensor.cuda())
	else:
		ori_tensor = Variable(ori_tensor)
	ori_tensor = torch.unsqueeze(ori_tensor,0)

	padding_left = 0
	padding_right = 0
	padding_top = 0
	padding_bottom = 0
	ori_height = ori_tensor.size()[2]
	ori_width = ori_tensor.size()[3]
	if ori_height % 4:
		padding_top = (4 - ori_height % 4)//2
		padding_bottom = 4 - ori_height % 4 - padding_top
		use_padding = True
	if ori_width % 4:
		padding_left = (4 - ori_width % 4)//2
		padding_right = 4 - ori_width % 4 - padding_left
		use_padding = True
	if use_padding:
		padding_transform = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
		ori_tensor = padding_transform(ori_tensor)

	output = unet(ori_tensor)

	if use_padding:
		output = output[:,:,padding_top : (padding_top + ori_height), padding_left : (padding_left + ori_width)]

	if cuda:
		result = (output.data).cpu().numpy()
	else:
		result = (output.data).numpy()

	result = result[0,0,:,:]
	return result

def load_model5(model_path, data, cuda):
	"""/home/student/Documents/Atom Segmentation APP/AtomSegGUI/atomseg_bupt_new_100/"""
	from model5 import UNet
	use_padding = False
	unet = UNet()
	if cuda:
		unet = unet.cuda()

	if not cuda:
		unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
	else:
		unet.load_state_dict(torch.load(model_path))

	transform = ToTensor()
	ori_tensor = transform(data)
	if cuda:
		ori_tensor = Variable(ori_tensor.cuda())
	else:
		ori_tensor = Variable(ori_tensor)
	ori_tensor = torch.unsqueeze(ori_tensor,0)

	padding_left = 0
	padding_right = 0
	padding_top = 0
	padding_bottom = 0
	ori_height = ori_tensor.size()[2]
	ori_width = ori_tensor.size()[3]
	if ori_height % 4:
		padding_top = (4 - ori_height % 4)//2
		padding_bottom = 4 - ori_height % 4 - padding_top
		use_padding = True
	if ori_width % 4:
		padding_left = (4 - ori_width % 4)//2
		padding_right = 4 - ori_width % 4 - padding_left
		use_padding = True
	if use_padding:
		padding_transform = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
		ori_tensor = padding_transform(ori_tensor)

	output = unet(ori_tensor)

	if use_padding:
		output = output[:,:,padding_top : (padding_top + ori_height), padding_left : (padding_left + ori_width)]

	if cuda:
		result = (output.data).cpu().numpy()
	else:
		result = (output.data).numpy()

	result = result[0,0,:,:]
	return result


def load_model6(model_path, data, cuda):
	"""/home/student/Documents/Atom Segmentation APP/AtomSegGUI/atom_seg_gaussian_mask/"""
	from model6 import UNet
	use_padding = False
	unet = UNet()
	if cuda:
		unet = unet.cuda()

	if not cuda:
		unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
	else:
		unet.load_state_dict(torch.load(model_path))

	transform = ToTensor()
	ori_tensor = transform(data)
	if cuda:
		ori_tensor = Variable(ori_tensor.cuda())
	else:
		ori_tensor = Variable(ori_tensor)
	ori_tensor = torch.unsqueeze(ori_tensor,0)

	padding_left = 0
	padding_right = 0
	padding_top = 0
	padding_bottom = 0
	ori_height = ori_tensor.size()[2]
	ori_width = ori_tensor.size()[3]
	if ori_height % 4:
		padding_top = (4 - ori_height % 4)//2
		padding_bottom = 4 - ori_height % 4 - padding_top
		use_padding = True
	if ori_width % 4:
		padding_left = (4 - ori_width % 4)//2
		padding_right = 4 - ori_width % 4 - padding_left
		use_padding = True
	if use_padding:
		padding_transform = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
		ori_tensor = padding_transform(ori_tensor)

	output = unet(ori_tensor)

	if use_padding:
		output = output[:,:,padding_top : (padding_top + ori_height), padding_left : (padding_left + ori_width)]

	if cuda:
		result = (output.data).cpu().numpy()
	else:
		result = (output.data).numpy()

	result = result[0,0,:,:]
	return result