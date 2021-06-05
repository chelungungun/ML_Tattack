"""
This file is to generate the desired attack target
"""
from loaddata import load_y
import numpy as np

global_seed = 2020
np.random.seed(global_seed)

def random_target(dataset, min_a, batch_id, batch_size, classifier_name):
	ytest = load_y(dataset)
	ytest[ytest==0] = -1
	num_label = np.shape(ytest)[1]
	a = np.arange(num_label)
	np.random.shuffle(a)
	ind = np.sort(a[0:min_a])

	one = np.ones(shape=(1, num_label), dtype=int)
	for i in ind:
		one[0][i] = -1

	cinds = np.tile(one, reps=(batch_size,1))
	y_batch = ytest[batch_id * batch_size:batch_id * batch_size + batch_size,:]
	y_batch_target = y_batch * cinds

	if classifier_name == 'svm':
		target_batch = y_batch_target
	else:
		y_batch_target[y_batch_target < 0] = 0
		target_batch = y_batch_target

	return target_batch, ind


def random_target_AB(dataset, min_a, batch_id, batch_size, classifier_name, random_seed, mode):
	np.random.seed(random_seed)
	dataset_predcorrect = dataset + "correct_test"

	li_creepware , li_voc2012 , li_planet = [0 , 1 , 4 , 5 , 6 , 7] , [0 , 1 , 2 , 3 , 7 , 11 , 12 , 13 , 14 , 16 ,
	                                                                   18] , [2 , 4 , 7 , 10 , 11, 14]
	lc_creepware , lc_voc2012 , lc_planet = [8 , 9 , 10 , 11 , 12 , 13 , 15] , [4 , 5 , 6 , 8 , 10 , 15 , 17 , 19] , \
	                                        [0 , 1 , 5 , 8 , 9 , 12 , 13]

	ytest = load_y(dataset_predcorrect)
	ytest[ytest==0] = -1
	num_label = np.shape(ytest)[1]
	if mode == 'A+B+':
		var = 'lc_' + dataset
		a = eval(var)
		np.random.shuffle(a)
		ind_A = np.sort(a[0:min_a])
		ind_B = np.array([i for i in a if i not in ind_A])
		np.random.shuffle(ind_B)
		ind_B = np.sort(ind_B[0:min_a - 1])
	elif mode == 'A+B-':
		var_a = 'lc_' + dataset
		a = eval(var_a)
		np.random.shuffle(a)
		ind_A = np.sort(a[0:min_a])

		var_b = "li_" + dataset
		b = eval(var_b)
		np.random.shuffle(b)
		ind_B = np.sort(b[0:min_a - 1])
	else:   #mode == 'ab'
		var = 'li_' + dataset
		a = eval(var)
		np.random.shuffle(a)
		ind_A = np.sort(a[0:min_a])

		ind_B = np.array([i for i in a if i not in ind_A])
		np.random.shuffle(ind_B)
		ind_B = np.sort(ind_B[0:min_a])

	one = np.ones(shape=(1, num_label), dtype=int)
	for i in ind_A:
		one[0][i] = -1

	cinds = np.tile(one, reps=(batch_size,1))
	y_batch = ytest[batch_id * batch_size:batch_id * batch_size + batch_size,:]
	y_batch_target = y_batch * cinds

	if classifier_name == 'svm':
		target_batch = y_batch_target
	else:
		y_batch_target[y_batch_target < 0] = 0
		target_batch = y_batch_target

	return target_batch, ind_A, ind_B


def corre_target(dataset, batch_id, batch_size, classifier_name):
	ytest = load_y(dataset + 'test')
	ytest[ytest == 0] = -1
	num_label = np.shape(ytest)[1]
	lc_creep , lc_voc , lc_planet = [8 , 9 , 10 , 11 , 12 , 13 , 14 , 15] , [4 , 5 , 6 , 8 , 10 , 15 , 17 ,
	                                                                                 19] , \
	                                        [0 ,1, 5 , 6,  8 , 9 , 12 , 13 , 16]
	one = np.ones(shape=(1 , num_label) , dtype=int)

	if "creepware" in dataset:
		ind_A = lc_creep
	elif "voc2012" in dataset:
		ind_A = lc_voc
	else:
		ind_A = lc_planet

	ind_A = np.sort(np.array(ind_A))
	a = np.arange(num_label)
	ind_B = np.array([i for i in a if i not in ind_A])
	for i in ind_A:
		one[0][i] = -1
	cinds = np.tile(one , reps=(batch_size , 1))
	y_batch = ytest[batch_id * batch_size:batch_id * batch_size + batch_size,:]
	y_batch_target = y_batch * cinds

	if classifier_name == 'svm':
		target_batch = y_batch_target
	else:
		y_batch_target[y_batch_target < 0] = 0
		target_batch = y_batch_target

	return target_batch, ind_A, ind_B


def target_single(dataset, batch_id, batch_size, classifier_name, label_ind):
	dataset_predcorrect = dataset + "correct_test"
	ytest = load_y(dataset_predcorrect)
	ytest[ytest == 0] = -1
	num_label = np.shape(ytest)[1]

	one = np.ones(shape=(1 , num_label) , dtype=int)
	one[0][label_ind] = -1

	cinds = np.tile(one , reps=(batch_size , 1))
	y_batch = ytest[batch_id * batch_size:batch_id * batch_size + batch_size , :]
	y_batch_target = y_batch * cinds

	if classifier_name == 'svm':
		target_batch = y_batch_target
	else:
		y_batch_target[y_batch_target < 0] = 0
		target_batch = y_batch_target

	a = np.arange(num_label)
	ind_A = np.array([label_ind])
	ind_B = np.array([i for i in a if i not in ind_A])

	return target_batch , ind_A , ind_B