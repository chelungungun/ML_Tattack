"""
This file is to run the targeted attack
"""
from module_target_generate import *
from module_attack import *
from module_evaluation import *
from loaddata import *
from module_attackability import attackability_svm
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from keras import optimizers
from keras.layers import Input , Dense
from keras import Model , regularizers
from art.classifiers import KerasClassifier
import keras
import copy

loss=keras.losses.BinaryCrossentropy()

global_seed = 2020
np.random.seed(global_seed)

dataset = 'creepware'
mode = 'A+B+'
min_a = 4
batch_id = 0
batch_size = 50
classifier_name = 'svm'
svm_cla = False
trade_AB = 1
cw_alpha = 0
item_num = 3
if classifier_name == 'svm':
	svm_cla = True

## Build data
dataset_predcorrect = dataset + "correct_test"
X, Y = load_data_correct(dataset_predcorrect)
num_of_label = Y.shape[1]
num_of_feature = X.shape[1]
batch_x = X[batch_id * batch_size: batch_id * batch_size + batch_size]
batch_y = Y[batch_id * batch_size: batch_id * batch_size + batch_size]

if classifier_name == 'svm':
	batch_yt = batch_y * 2 - 1
else:
	batch_yt = batch_y

## Build classifiers
def svm(inputshape , labelshape):
	inputs = Input(shape=(inputshape ,))
	outputs = Dense(labelshape , activation='linear' , kernel_regularizer=regularizers.l1_l2(l1=0.000000 , l2=0.000000))(
		inputs)
	model = Model(inputs , outputs)
	return model


ds_creep = [4073, 16]
ds_voc = [(299, 299, 3), 20]
ds_resnetvoc = [(224, 224, 3), 20]
ds_planet = [(255, 255, 3), 17]
ds_resnetplanet = [(224, 224, 3), 17]
ds = []
if dataset == 'creepware':
	ds = ds_creep
elif dataset == 'voc2012':
	ds = ds_voc
elif dataset == 'resnetvoc2012':
	ds = ds_resnetvoc
elif dataset == 'planet':
	ds = ds_planet
else:
	ds = ds_resnetplanet

if dataset == 'voc2012' or dataset == 'planet':
	min_pixel_value , max_pixel_value = -1.0 , 1.0
	clip_values = (min_pixel_value , max_pixel_value)
elif dataset == 'resnetvoc2012' or dataset == 'resnetplanet':
	min_pixel_value , max_pixel_value = -123.68 , 151.061
	clip_values = (min_pixel_value , max_pixel_value)
else:
	clip_values = None

model = svm(ds[0], ds[1])
model_name = dataset + '_svmbasic' + '.h5'
model.load_weights('model/' + model_name)
model.compile(optimizer=optimizers.Adam(lr=0.01) , loss='hinge' ) #The loss is the key, must recompile.
classifier = KerasClassifier(model=model , use_logits=False , clip_values=clip_values , svm_cla=svm_cla)

##attack settings
if dataset == 'creepware':
	step_ori = 0.01
	eps = 10
	eps_bgt = [0.1, 0.2, 0.3]
elif dataset == 'voc2012':
	step_ori = 0.3
	eps = 50
	eps_bgt = [1.5 , 3 , 4.5]
elif dataset == 'planet':
	step_ori = 0.1
	eps = 20
	eps_bgt = [0.25 , 0.5 , 0.75]
elif dataset == 'resnetvoc2012':
	step_ori = 10
	eps = 3000
else:
	step_ori = 10
	eps = 1500

max_iter = 1000

random_seeds = np.arange(20)
ratio_successes, ratio_hurts, costAs, lossBs = [] , [] , [] , []
ratios_suc_eps0, ratios_suc_eps1, ratios_suc_eps2 = [], [] ,[]
ratios_hurt_eps0, ratios_hurt_eps1, ratios_hurt_eps2 = [], [] ,[]
losss_B_eps0, losss_B_eps1, losss_B_eps2 = [], [] ,[]
attackabilitys_AB = []

for random_seed in random_seeds:
	## build attack target
	batch_target, ind_A, ind_B = random_target_AB(dataset=dataset,min_a=min_a,batch_id=batch_id,batch_size=batch_size,
	                                     classifier_name=classifier_name,random_seed=random_seed,mode=mode)

	fun_attackability = attackability_svm(inputshape_x=num_of_feature, inputshape_y=num_of_label,
	                                      labelshape=num_of_label, ind_A=ind_A, ind_B=ind_B, tradeAB=trade_AB, ite_num=item_num, dataset=dataset)
	attackability_AB = fun_attackability([batch_x, batch_yt])
	attackabilitys_AB.append(attackability_AB)

	batch_x_copy = copy.deepcopy(batch_x)
	batch_target_copy = copy.deepcopy(batch_target)

	adv_eps , batch_X_ADV, ind_unsuccess = ML_PGD(batch_x=batch_x_copy,batch_target=batch_target_copy,classifier=classifier,
	                              step_ori=step_ori,eps=eps,max_iter=max_iter,classifier_name=classifier_name,
	                              ind_A=ind_A, batch_size=batch_size, eps_bgt=eps_bgt, trade_AB=trade_AB, ind_B=ind_B, cw_alpha=cw_alpha)

	batch_pred = classifier.predict(batch_X_ADV)

	batch_pred_eps0 = classifier.predict(adv_eps[0])
	batch_pred_eps1 = classifier.predict(adv_eps[1])
	batch_pred_eps2 = classifier.predict(adv_eps[2])
	ratio_suc_eps0 , ratio_hurt_eps0 = ratio_success_hurt(batch_target=batch_target_copy , batch_pred=batch_pred_eps0 ,
	                                                ind_A=ind_A , ind_B=ind_B , svm_cla=svm_cla)
	ratio_suc_eps1 , ratio_hurt_eps1 = ratio_success_hurt(batch_target=batch_target_copy , batch_pred=batch_pred_eps1 ,
	                                                      ind_A=ind_A , ind_B=ind_B , svm_cla=svm_cla)
	ratio_suc_eps2 , ratio_hurt_eps2 = ratio_success_hurt(batch_target=batch_target_copy , batch_pred=batch_pred_eps2 ,
	                                                      ind_A=ind_A , ind_B=ind_B , svm_cla=svm_cla)

	costA_eps0 , loss_B_eps0 = costA_lossB(batch_x=batch_x , batch_x_adv=adv_eps[0] , batch_y=batch_y , batch_pred=batch_pred_eps0 ,
	                            ind_B=ind_B , svm_cla=svm_cla)
	costA_eps1 , loss_B_eps1 = costA_lossB(batch_x=batch_x , batch_x_adv=adv_eps[1] , batch_y=batch_y ,
	                                       batch_pred=batch_pred_eps1 ,
	                                       ind_B=ind_B , svm_cla=svm_cla)
	costA_eps2 , loss_B_eps2 = costA_lossB(batch_x=batch_x , batch_x_adv=adv_eps[2] , batch_y=batch_y ,
	                                       batch_pred=batch_pred_eps2 ,
	                                       ind_B=ind_B , svm_cla=svm_cla)
	ratios_suc_eps0.append(ratio_suc_eps0)
	ratios_suc_eps1.append(ratio_suc_eps1)
	ratios_suc_eps2.append(ratio_suc_eps2)
	ratios_hurt_eps0.append(ratio_hurt_eps0)
	ratios_hurt_eps1.append(ratio_hurt_eps1)
	ratios_hurt_eps2.append(ratio_hurt_eps2)
	losss_B_eps0.append(loss_B_eps0)
	losss_B_eps1.append(loss_B_eps1)
	losss_B_eps2.append(loss_B_eps2)

	ratio_success, ratio_hurt = ratio_success_hurt(batch_target=batch_target_copy, batch_pred=batch_pred, ind_A= ind_A, ind_B=ind_B, svm_cla=svm_cla)
	costA, lossB = costA_lossB(batch_x=batch_x, batch_x_adv=batch_X_ADV, batch_y=batch_y, batch_pred=batch_pred, ind_B=ind_B, svm_cla=svm_cla)
	ratio_successes.append(ratio_success)
	ratio_hurts.append(ratio_hurt)
	costAs.append(costA)
	lossBs.append(lossB)

ratios_suc_eps0_mean = np.mean(np.array(ratios_suc_eps0))
ratios_suc_eps1_mean = np.mean(np.array(ratios_suc_eps1))
ratios_suc_eps2_mean = np.mean(np.array(ratios_suc_eps2))
ratios_hurt_eps0_mean = np.mean(np.array(ratios_hurt_eps0))
ratios_hurt_eps1_mean = np.mean(np.array(ratios_hurt_eps1))
ratios_hurt_eps2_mean = np.mean(np.array(ratios_hurt_eps2))
losss_B_eps0_mean = np.mean(np.array(losss_B_eps0))
losss_B_eps1_mean = np.mean(np.array(losss_B_eps1))
losss_B_eps2_mean = np.mean(np.array(losss_B_eps2))

ratio_success_mean = np.mean(np.array(ratio_successes))
ratio_hurt_mean = np.mean(np.array(ratio_hurts))
costA_mean = np.mean(np.array(costAs))
lossB_mean = np.mean(np.array(lossBs))
attackabilitys_AB_mean = np.mean(np.array(attackabilitys_AB))

with open('AaBb_attackability/' + dataset + mode + 'sizeA'+ str(min_a) + 'tradeAB' + str(trade_AB) + '.txt' ,
          'w') as f:
	suc_eps0 = 'suc_eps0: ' + str(ratios_suc_eps0_mean)
	f.write("%s\n" % suc_eps0)
	suc_eps1 = 'suc_eps1: ' + str(ratios_suc_eps1_mean)
	f.write("%s\n" % suc_eps1)
	suc_eps2 = 'suc_eps2: ' + str(ratios_suc_eps2_mean)
	f.write("%s\n" % suc_eps2)
	suc_final = 'suc_final: ' + str(ratio_success_mean)
	f.write("%s\n" % suc_final)

	hurt_eps0 = 'hurt_eps0: ' + str(ratios_hurt_eps0_mean)
	f.write("%s\n" % hurt_eps0)
	hurt_eps1 = 'hurt_eps1: ' + str(ratios_hurt_eps1_mean)
	f.write("%s\n" % hurt_eps1)
	hurt_eps2 = 'hurt_eps2: ' + str(ratios_hurt_eps2_mean)
	f.write("%s\n" % hurt_eps2)

	lossB_eps0 = 'lossB_eps0: ' + str(losss_B_eps0_mean)
	f.write("%s\n" % lossB_eps0)
	lossB_eps1 = 'lossB_eps1: ' + str(losss_B_eps1_mean)
	f.write("%s\n" % lossB_eps1)
	lossB_eps2 = 'lossB_eps2: ' + str(losss_B_eps2_mean)
	f.write("%s\n" % lossB_eps2)

	attack_norm = 'attack_norm: ' + str(costA_mean)
	f.write("%s\n" % attack_norm)
	attackability = 'attackability: ' + str(attackabilitys_AB_mean)
	f.write("%s\n" % attackability)

f.close()


print()