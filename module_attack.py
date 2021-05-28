"""
This file is to implement the attack methods for multi-label targeted evasion attack
"""
import pickle
import numpy as np
import scipy.stats as ss
import pandas as pd
from scipy.stats import pearsonr
import copy
from termcolor import colored
from art.attacks import ProjectedGradientDescent
from art.classifiers import KerasClassifier
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras import optimizers


def ML_PGD(batch_x, batch_target, classifier, step_ori, eps, eps_bgt, max_iter, classifier_name, ind_A, ind_B, trade_AB, cw_alpha, batch_size):
	svm_cla = False
	if classifier_name == 'svm':
		svm_cla = True
	attack = ProjectedGradientDescent(classifier=classifier , eps=eps , max_iter=max_iter , eps_step=step_ori ,
	                                  norm=2 ,
	                                  targeted=True , min=True , project=False , inputlike=svm_cla , batch_size=batch_size)

	adv_eps , x_test_adv , ind_unsuccess = attack.generate(x=batch_x , y=batch_target, eps_step=step_ori ,
	                                       x_init=batch_x, ind_A=ind_A, eps_bgt=eps_bgt, trade_AB=trade_AB, ind_B=ind_B, cw_alpha=cw_alpha)

	return  adv_eps , x_test_adv, ind_unsuccess