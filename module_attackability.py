"""
This module is to implement the proposed attackability assessment
"""
import os
from keras.layers import Input , Dense , Lambda , Concatenate , Activation
from keras import Model , regularizers
import numpy as np
from keras import optimizers
import tensorflow as tf
import keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input , Dense , GlobalAveragePooling2D

glo_seed = 2020
tf.random.set_seed(glo_seed)

def one_grad(x,i):
	# x[0]: outputs; x[1]:inputs; x[2]:labels
	one_out = K.expand_dims(x[0][:,i], axis=1)
	grad = tf.gradients(one_out , x[1])[0]
	grad_dim = K.shape(grad)[1]
	cp_one_out = K.tile(one_out, (1,grad_dim))
	one_label = K.expand_dims(x[2][:,i], axis=1)
	cp_one_label = K.tile(one_label, (1,grad_dim))

	#The method to calculate normalized gradients
	grad_nl = -1 * grad * cp_one_label / K.log(K.exp(0.01 ) + K.exp(K.abs(cp_one_label * cp_one_out )))
	one_grad = K.expand_dims(grad_nl , axis=(1))

	return one_grad


def regul(x):
	batch_size = K.shape(x)[0]
	loss = K.sum(K.sqrt(K.sum(K.square(K.sum(x, axis=1)), axis=1))) / K.cast(batch_size,tf.float32)

	return loss


def target_grads(x,ind_A,ind_B):
	# returns the grads of labels in A and the sum grads of labels in B
	# x: batch * label * grad_dim

	grads_A = tf.gather(x, axis=1, indices=ind_A)
	grads_B = K.sum(tf.gather(x, axis=1, indices=ind_B), axis=1, keepdims=True)

	return grads_A, grads_B


def Tattackability(x, ite_num, tradeAB):
	# x[0]: batch_size * |A| * grad_dim; x[1]: batch_size * 1 * grad_dim
	batch_size = tf.shape(x[0])[0]
	size_A = K.shape(x[0])[1]
	dim = K.shape(x[0])[2]
	grads_A = x[0]
	grads_B = x[1] * tradeAB

	cgrads_B = K.tile(grads_B, (1 , size_A, 1))

	up_x = grads_A
	remain_record = grads_A
	select_record = []

	for i in range(ite_num):
		norm_B = K.sqrt(K.sum(K.square(cgrads_B), axis=-1, keepdims=True)) #batch * size_A * 1
		norm_A = K.sqrt(K.sum(K.square(up_x), axis=-1, keepdims=True)) #batch * size_A * 1
		cos_ij = K.sum(up_x * cgrads_B, axis=-1, keepdims=True) / norm_A / norm_B #batch * size_A * 1
		max_c1 = norm_A * cos_ij / (-1 * K.abs(cos_ij))
		max_c2 = norm_A * K.sqrt(1 - K.square(cos_ij))
		max_c3 = K.sqrt(K.sum(K.square(up_x - cgrads_B), axis=-1, keepdims=True)) * (norm_A * cos_ij - norm_B) / K.abs(norm_A * cos_ij - norm_B)
		max_cs = K.concatenate((max_c1,max_c2,max_c3), axis=-1) # batch * size_A * 3
		max_c = K.max(max_cs , axis=-1) # batch * size_A

		[values , indices] = tf.math.top_k(max_c)
		cat_idx = tf.stack([tf.range(0 , batch_size) , K.squeeze(indices, axis=-1)] , axis=1)
		x_max_gather = K.expand_dims(tf.gather_nd(up_x, cat_idx), axis=1)
		select_record.append(K.expand_dims(tf.gather_nd(remain_record, cat_idx), axis=1))
		cp_xmax = K.tile(x_max_gather, (1, size_A, 1))

		[values_neg , indices_neg] = tf.math.top_k(-1 * max_c, k=size_A - 1)
		cp_ind = K.tile(K.expand_dims(tf.range(0 , batch_size), axis=1), (1, size_A-1))
		gather_neg_id = K.reshape(K.concatenate([K.expand_dims(cp_ind, axis=2), K.expand_dims(indices_neg, axis=2)], axis=2), shape=(batch_size * (size_A-1), 2))
		x_neg_gather = K.reshape(tf.gather_nd(remain_record, gather_neg_id), shape=(batch_size, size_A-1, dim))

		zo = tf.zeros(shape=(batch_size, 1, dim))
		remain_record = K.concatenate([x_neg_gather, zo], axis=1)
		up_x = cp_xmax + remain_record

	norm_Bo = K.sqrt(K.sum(K.square(cgrads_B) , axis=-1 , keepdims=True))  # batch * size_A * 1
	norm_Ao = K.sqrt(K.sum(K.square(up_x) , axis=-1 , keepdims=True))  # batch * size_A * 1
	cos_ijo = K.sum(up_x * cgrads_B , axis=-1 , keepdims=True) / norm_Ao / norm_Bo  # batch * size_A * 1
	max_c1o = norm_Ao * cos_ijo / (-1 * K.abs(cos_ijo))
	max_c2o = norm_Ao * K.sqrt(1 - K.square(cos_ijo))
	max_c3o = K.sqrt(K.sum(K.square(up_x - cgrads_B) , axis=-1 , keepdims=True)) * (norm_Ao * cos_ijo - norm_Bo) / K.abs(
		norm_Ao * cos_ijo - norm_Bo)
	max_cso = K.concatenate((max_c1o , max_c2o , max_c3o) , axis=-1)  # batch * size_A * 3
	max_co = K.max(max_cso , axis=-1)  # batch * size_A

	[valueso , indiceso] = tf.math.top_k(max_co)
	cat_idxo = tf.stack([tf.range(0 , batch_size) , K.squeeze(indiceso , axis=-1)] , axis=1)
	x_max_gathero = tf.gather_nd(up_x , cat_idxo) # batch * grad_dim

	norm_Bl = K.sqrt(K.sum(K.square(grads_B) , axis=-1 ))  # batch * 1
	norm_Al = K.sqrt(K.sum(K.square(x_max_gathero) , axis=-1 , keepdims=True))  # batch  * 1
	cos_ijl = K.sum(x_max_gathero * K.squeeze(grads_B, axis=1) , axis=-1 , keepdims=True) / norm_Al / norm_Bl  # batch  * 1
	max_c1l = norm_Al * cos_ijl / (-1 * K.abs(cos_ijl))
	max_c2l = norm_Al * K.sqrt(1 - K.square(cos_ijl))
	max_c3l= K.sqrt(K.sum(K.square(x_max_gathero - K.squeeze(grads_B, axis=1)) , axis=-1 , keepdims=True)) * (norm_Al * cos_ijl - norm_Bl) / K.abs(
		norm_Al * cos_ijl - norm_Bl)
	max_csl = K.concatenate((max_c1l , max_c2l , max_c3l) , axis=-1)  # batch  * 3
	max_cl = K.max(max_csl , axis=-1 , keepdims=True)  # batch * 1
	batch_Tattack = K.expand_dims(max_cl, axis=1)

	select_record.append(K.expand_dims(tf.gather_nd(remain_record , cat_idxo) , axis=1))
	select_record_tensor = K.concatenate(select_record, axis=1)
	select_record_sepsum = K.sum(K.sqrt(K.sum(K.square(select_record_tensor), axis=-1)), axis=1, keepdims=True) # batch * 1
	# batch_max = K.expand_dims(K.sqrt(K.sum(K.square(x_max_gathero), axis=1, keepdims=True)), axis=1) # batch * 1 * 1

	return [batch_Tattack, select_record_sepsum, select_record_tensor]



def attackability_svm(inputshape_x , inputshape_y, labelshape , ind_A, ind_B, tradeAB, ite_num, dataset):
	#ind_A: the label indes in attacked label set A
	#loss_lambda: the regularization strength of regularizor
	#tradeAB: the trade-off weight between attack on A and mis-hurt on B
	inputs_x = Input(shape=(inputshape_x ,))
	inputs_y = Input(shape=(inputshape_y ,))
	outputs = Dense(labelshape , activation='linear' , kernel_regularizer=regularizers.l1_l2(l1=0.000000 , l2=0.0000005))(
		inputs_x)
	grads_list = []
	fun_one_grad = Lambda(one_grad)
	for m in range(labelshape):
		fun_one_grad.arguments = {'i': m}
		grad = fun_one_grad([outputs, inputs_x, inputs_y])
		grads_list.append(grad)
	grads = Concatenate(1)(grads_list)
	fun_rel = Lambda(regul)

	fun_ta_grads = Lambda(target_grads)
	fun_ta_grads.arguments = {'ind_A': ind_A, 'ind_B': ind_B}
	grads_A , grads_B = fun_ta_grads(grads)

	fun_unta_grads = Lambda(Tattackability)
	fun_unta_grads.arguments = {'ite_num': ite_num, 'tradeAB': tradeAB}
	[batch_Tattack , select_record_sepsum , select_record_tensor] = fun_unta_grads([grads_A,grads_B])
	loss = fun_rel(batch_Tattack)

	fun_attackability = K.function([inputs_x , inputs_y] , loss)
	model = Model(inputs = [inputs_x, inputs_y] , outputs = outputs )
	model_name = dataset + '_svmbasic' + '.h5'
	model.load_weights('model/' + model_name)
	model.compile(optimizer=optimizers.Adam(lr=0.01) , loss='hinge')

	return fun_attackability


def attackability_logistic(inputshape_x , inputshape_y, labelshape , ind_A, ind_B, tradeAB, ite_num, dataset):
	#ind_A: the label indes in attacked label set A
	#loss_lambda: the regularization strength of regularizor
	#tradeAB: the trade-off weight between attack on A and mis-hurt on B
	inputs_x = Input(shape=inputshape_x)
	inputs_y = Input(shape=(inputshape_y ,))
	base_model = InceptionV3(weights='imagenet' , include_top=False , input_tensor=inputs_x)
	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	outputs = Dense(labelshape , activation='linear')(x)

	grads_list = []
	fun_one_grad = Lambda(one_grad)
	for m in range(labelshape):
		fun_one_grad.arguments = {'i': m}
		grad = fun_one_grad([outputs, inputs_x, inputs_y])
		grads_list.append(grad)
	grads = Concatenate(1)(grads_list)
	fun_rel = Lambda(regul)

	fun_ta_grads = Lambda(target_grads)
	fun_ta_grads.arguments = {'ind_A': ind_A, 'ind_B': ind_B}
	grads_A , grads_B = fun_ta_grads(grads)

	fun_unta_grads = Lambda(Tattackability)
	fun_unta_grads.arguments = {'ite_num': ite_num, 'tradeAB': tradeAB}
	[batch_Tattack , select_record_sepsum , select_record_tensor] = fun_unta_grads([grads_A,grads_B])
	loss = fun_rel(batch_Tattack)

	fun_attackability = K.function([inputs_x , inputs_y] , loss)
	outputs_prob = Activation('sigmoid')(outputs)
	model = Model(inputs=[base_model.input , inputs_y] , outputs=outputs_prob)

	model_name = dataset + 'lambda0' + '.h5'
	model.load_weights('model_basic/' + model_name)
	model.compile(optimizer=optimizers.Adam(lr=0.01) , loss='binary_crossentropy')

	return fun_attackability

