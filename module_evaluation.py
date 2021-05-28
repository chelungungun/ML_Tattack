"""
This file is to evaluate the attack performance
"""
import numpy as np

def ratio_success_hurt(batch_target, batch_pred, ind_A, ind_B, svm_cla):
	# batch_target is the attack target of batch data
	# batch_pred is the prediction of batch data
	# ind is the indices of attacked labels
	# batch_target and batch_pred are converted to 0~1
	if svm_cla == True:
		batch_pred[batch_pred >= 0] = 1
		batch_pred[batch_pred < 0] = -1
	else:
		batch_pred[batch_pred >= 0.5] = 1
		batch_pred[batch_pred < 0.5] = 0

	batch_target[batch_target < 0 ] = 0
	batch_pred[batch_pred < 0] = 0
	# S = np.arange(np.shape(batch_target)[1])
	A = ind_A
	B = ind_B
	# B = np.array([i for i in S if i not in A])

	TA = np.transpose(np.transpose(batch_target)[A])
	TB = np.transpose(np.transpose(batch_target)[B])
	PA = np.transpose(np.transpose(batch_pred)[A])
	PB = np.transpose(np.transpose(batch_pred)[B])

	success = 1 - np.mean(np.abs(TA - PA))
	hurt = np.mean(np.abs(TB - PB))

	return success, hurt


def costA_lossB(batch_x, batch_x_adv, batch_y, batch_pred, ind_B, svm_cla):
	cost = np.mean(np.sqrt(np.sum(np.square(batch_x - batch_x_adv), axis=1)))

	if svm_cla == True:
		batch_yt = batch_y * 2 - 1
	else:
		batch_yt = batch_y
	S = np.arange(np.shape(batch_y)[1])
	# ind_ = np.array([i for i in S if i not in ind])
	ind_ = ind_B
	loss_hinge = np.mean(np.maximum(1-batch_yt * batch_pred, 0)[:,ind_])
	# loss_entropy = np.mean(-(batch_yt * np.log(batch_pred)) - (1 - batch_yt) * np.log(1 - batch_pred))
	loss_entropy = 1
	if svm_cla == True:
		loss = loss_hinge
	else:
		loss = loss_entropy

	return cost, loss


