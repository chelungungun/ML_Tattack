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
import os
from keras.layers import Input , Dense
from keras import Model , regularizers
from art.classifiers import KerasClassifier
import keras
from sklearn.metrics import f1_score

def binary_label(pred_pro):
	predY = np.zeros([np.shape(pred_pro)[0] , np.shape(pred_pro)[1]] , dtype=np.int32)
	for i in range(np.shape(pred_pro)[0]):
		predY[i][pred_pro[i] >= 0.5] = 1
		predY[i][pred_pro[i] < 0.5] = 0
	return predY


def binary_label_svm(pred_pro):
	predY = np.zeros([np.shape(pred_pro)[0] , np.shape(pred_pro)[1]] , dtype=np.int32)
	for i in range(np.shape(pred_pro)[0]):
		predY[i][pred_pro[i] >= 0] = 1
		predY[i][pred_pro[i] < 0] = -1
	return predY


global_seed = 2020
np.random.seed(global_seed)

dataset = 'creepware'
# dataset = 'voc2012'
method = 'nuclear'
loss_lambda = 0.001
batch_id = 0
batch_size = 100
classifier_name = 'svm'
svm_cla = False
tradeAB = 1  #This is to indicate the classifier; candidate: 0, 0.5, 1
trade_AB = 0#This is to indicate the ML-PGD; candidate: 0, 1
cw_alpha = 0
if classifier_name == 'svm':
	svm_cla = True

## Build data
dataset_predcorrect = dataset + "test"
X, Y = load_data_correct(dataset_predcorrect)
num_of_label = Y.shape[1]
num_of_feature = X.shape[1]
batch_x = X[batch_id * batch_size: batch_id * batch_size + batch_size]
batch_y = Y[batch_id * batch_size: batch_id * batch_size + batch_size] # y: 0-1

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
model_name = dataset + method + 'lambda' + str(loss_lambda) + '_tradeab' + str(tradeAB) + '.h5'
model.load_weights('model_regularitors/' + model_name)
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
	eps_bgt = [1.8 , 3 , 4.5]
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
max_iter = 1000

batch_target, ind_A, ind_B = corre_target(dataset=dataset,batch_id=batch_id,batch_size=batch_size,
	                                     classifier_name=classifier_name)
batch_x_copy = copy.deepcopy(batch_x)
batch_target_copy = copy.deepcopy(batch_target)

adv_eps , batch_X_ADV, ind_unsuccess = ML_PGD(batch_x=batch_x_copy,batch_target=batch_target_copy,classifier=classifier,
	                              step_ori=step_ori,eps=eps,max_iter=max_iter,classifier_name=classifier_name,
	                              ind_A=ind_A, batch_size=batch_size, eps_bgt=eps_bgt, trade_AB=trade_AB, ind_B=ind_B, cw_alpha=cw_alpha)

batch_pred_pro = classifier.predict(batch_X_ADV)
# batch_pred_pro = classifier.predict(batch_x)

batch_pred_eps0_pro = classifier.predict(adv_eps[0])
batch_pred_eps1_pro = classifier.predict(adv_eps[1])
batch_pred_eps2_pro = classifier.predict(adv_eps[2])

batch_pred = copy.deepcopy(batch_pred_pro)
batch_pred_eps0 = copy.deepcopy(batch_pred_eps0_pro)
batch_pred_eps1 = copy.deepcopy(batch_pred_eps1_pro)
batch_pred_eps2 = copy.deepcopy(batch_pred_eps2_pro)

ratio_suc_eps0 , ratio_hurt_eps0 = ratio_success_hurt(batch_target=batch_target_copy , batch_pred=batch_pred_eps0 ,
	                                                ind_A=ind_A , ind_B=ind_B , svm_cla=svm_cla)
ratio_suc_eps1 , ratio_hurt_eps1 = ratio_success_hurt(batch_target=batch_target_copy , batch_pred=batch_pred_eps1 ,
	                                                      ind_A=ind_A , ind_B=ind_B , svm_cla=svm_cla)
ratio_suc_eps2 , ratio_hurt_eps2 = ratio_success_hurt(batch_target=batch_target_copy , batch_pred=batch_pred_eps2 ,
	                                                      ind_A=ind_A , ind_B=ind_B , svm_cla=svm_cla)

if classifier == 'svm':
	predY_eps0 = binary_label_svm(pred_pro=batch_pred_eps0)
	predY_eps1 = binary_label_svm(pred_pro=batch_pred_eps1)
	predY_eps2 = binary_label_svm(pred_pro=batch_pred_eps2)
	predY = binary_label_svm(pred_pro=batch_pred)
else:
	predY_eps0 = binary_label(pred_pro=batch_pred_eps0)
	predY_eps1 = binary_label(pred_pro=batch_pred_eps1)
	predY_eps2 = binary_label(pred_pro=batch_pred_eps2)
	predY = binary_label(pred_pro=batch_pred)

if classifier == 'svm':
	predY_eps0 = (predY_eps0 + 1) / 2
	predY_eps1 = (predY_eps1 + 1) / 2
	predY_eps2 = (predY_eps2 + 1) / 2
	predY = (predY + 1) / 2

micro_f1_eps0 = f1_score(batch_y , predY_eps0 , average='micro')
micro_f1_eps1 = f1_score(batch_y , predY_eps1 , average='micro')
micro_f1_eps2 = f1_score(batch_y , predY_eps2 , average='micro')
micro_f1 = f1_score(batch_y , predY , average='micro')

macro_f1_eps0 = f1_score(batch_y , predY_eps0 , average='macro')
macro_f1_eps1 = f1_score(batch_y , predY_eps1 , average='macro')
macro_f1_eps2 = f1_score(batch_y , predY_eps2 , average='macro')
macro_f1 = f1_score(batch_y , predY , average='macro')


costA_eps0 , loss_B_eps0 = costA_lossB(batch_x=batch_x , batch_x_adv=adv_eps[0] , batch_y=batch_y , batch_pred=batch_pred_eps0 ,
	                            ind_B=ind_B , svm_cla=svm_cla)
costA_eps1 , loss_B_eps1 = costA_lossB(batch_x=batch_x , batch_x_adv=adv_eps[1] , batch_y=batch_y ,
                                       batch_pred=batch_pred_eps1 ,
                                       ind_B=ind_B , svm_cla=svm_cla)
costA_eps2 , loss_B_eps2 = costA_lossB(batch_x=batch_x , batch_x_adv=adv_eps[2] , batch_y=batch_y ,
                                       batch_pred=batch_pred_eps2 ,
                                       ind_B=ind_B , svm_cla=svm_cla)

ratio_success, ratio_hurt = ratio_success_hurt(batch_target=batch_target_copy, batch_pred=batch_pred, ind_A= ind_A, ind_B=ind_B, svm_cla=svm_cla)
costA, lossB = costA_lossB(batch_x=batch_x, batch_x_adv=batch_X_ADV, batch_y=batch_y, batch_pred=batch_pred, ind_B=ind_B, svm_cla=svm_cla)

predictions = [batch_y, batch_pred_eps0_pro, batch_pred_eps1_pro, batch_pred_eps2_pro, batch_pred_pro]
f = open('attack_predictions/' + dataset + 'pre' + method + 'lambda' + str(loss_lambda) + "_Ctradeab" + str(tradeAB) +'_Atradeab' + str(trade_AB)+ '.pickle' , 'wb')
pickle.dump(predictions , f)
f.close()

with open('reg_attackability/' + dataset + method + 'lambda' + str(loss_lambda) + "_Ctradeab" + str(tradeAB) +'_Atradeab' + str(trade_AB)+ '.txt' ,
          'w') as f:
	suc_eps0 = 'suc_eps0: ' + str(ratio_suc_eps0)
	f.write("%s\n" % suc_eps0)
	suc_eps1 = 'suc_eps1: ' + str(ratio_suc_eps1)
	f.write("%s\n" % suc_eps1)
	suc_eps2 = 'suc_eps2: ' + str(ratio_suc_eps2)
	f.write("%s\n" % suc_eps2)
	suc_final = 'suc_final: ' + str(ratio_success)
	f.write("%s\n" % suc_final)
	hurt_final = 'hurt_final: ' + str(ratio_hurt)
	f.write("%s\n" % hurt_final)

	hurt_eps0 = 'hurt_eps0: ' + str(ratio_hurt_eps0)
	f.write("%s\n" % hurt_eps0)
	hurt_eps1 = 'hurt_eps1: ' + str(ratio_hurt_eps1)
	f.write("%s\n" % hurt_eps1)
	hurt_eps2 = 'hurt_eps2: ' + str(ratio_hurt_eps2)
	f.write("%s\n" % hurt_eps2)

	lossB_eps0 = 'lossB_eps0: ' + str(loss_B_eps0)
	f.write("%s\n" % lossB_eps0)
	lossB_eps1 = 'lossB_eps1: ' + str(loss_B_eps1)
	f.write("%s\n" % lossB_eps1)
	lossB_eps2 = 'lossB_eps2: ' + str(loss_B_eps2)
	f.write("%s\n" % lossB_eps2)

	attack_norm = 'attack_norm: ' + str(costA)
	f.write("%s\n" % attack_norm)

	mf_eps0 = 'mf_eps0: ' + str(micro_f1_eps0)
	f.write("%s\n" % mf_eps0)
	mf_eps1 = 'mf_eps1: ' + str(micro_f1_eps1)
	f.write("%s\n" % mf_eps1)
	mf_eps2 = 'mf_eps2: ' + str(micro_f1_eps2)
	f.write("%s\n" % mf_eps2)

	MF_eps0 = 'MF_eps0: ' + str(macro_f1_eps0)
	f.write("%s\n" % MF_eps0)
	MF_eps1 = 'MF_eps1: ' + str(macro_f1_eps1)
	f.write("%s\n" % MF_eps1)
	MF_eps2 = 'MF_eps2: ' + str(macro_f1_eps2)
	f.write("%s\n" % MF_eps2)

	mf_final = 'final_mf: ' + str(micro_f1)
	f.write("%s\n" % mf_final)
	MF_final = 'final_MF: ' + str(macro_f1)
	f.write("%s\n" % MF_final)
	# attackability = 'attackability: ' + str(attackabilitys_AB_mean)
	# f.write("%s\n" % attackability)

f.close()


print()