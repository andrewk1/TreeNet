
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, MaxPooling3D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from data import load_train_data, load_test_data
from sklearn.metrics import f1_score, recall_score, precision_score
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 480
img_cols = 480

smooth = 1.


def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	print y_true_f
	intersection = K.sum(y_true_f * y_pred_f)
	print K.sum(y_true_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

# Given the returned .npy test ask and truth mask, returns dice coeff
def dice_loss_test():
	path = 'first_run_results/results/orig/'
	imgs_test = np.load(path + 'imgs_test.npy')
	imgs_id = np.load(path + 'imgs_id_test.npy')
	imgs_mask_truth = np.load(path + 'imgs_mask_test.npy')
	imgs_mask_test = np.load(path + 'result_imgs_mask_test.npy')
	imgs_mask_test = preprocess((imgs_mask_test * 255).astype(np.uint8))
	imgs_mask_truth = preprocess(imgs_mask_truth.astype(np.uint8))
	y_true_f = np.ndarray.flatten(imgs_mask_truth)
	y_pred_f = np.ndarray.flatten(imgs_mask_test)
	print y_true_f
	print y_pred_f
	intersection = np.sum(y_true_f * y_pred_f)
	print 'Precision', precision_score([1 if x > 200 else 0 for x in y_true_f], [1 if x > 200 else 0 for x in y_pred_f], average='binary') 
	print 'Recall', recall_score([1 if x > 200 else 0 for x in y_true_f], [1 if x > 200 else 0 for x in y_pred_f], average='binary') 
	print 'F1', f1_score([1 if x > 200 else 0 for x in y_true_f], [1 if x > 100 else 0 for x in y_pred_f], average='binary')  
	print 'Dice score', (2 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def get_unet():
	inputs = Input((img_rows, img_cols, 1))
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid', data_format="channels_last")(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy', dice_coef])
	model.summary()
	return model


def preprocess(imgs):
	imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 1), dtype=np.uint8)
	for i in range(imgs.shape[0]):
		if imgs[i].shape == (168, 168, 3):
			x = np.sum(imgs[i], axis=2)
		else:
			x = imgs[i]
		imgs_p[i] = resize(x, (img_cols, img_rows, 1), preserve_range=True)

	# mgs_p = imgs_p[..., np.newaxis]
	return imgs_p


def train_and_predict(load=False):
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, imgs_mask_train = load_train_data()

	imgs_train = preprocess(imgs_train)
	imgs_mask_train = preprocess(imgs_mask_train)

	imgs_train = imgs_train.astype('float32')
	# print 'Training images: '
	# print imgs_train
	mean = np.mean(imgs_train)  # mean for data centering
	std = np.std(imgs_train)  # std for data normalization

	imgs_train -= mean
	imgs_train /= std

	imgs_mask_train = imgs_mask_train.astype('float32')
	# print "OG Masks: "
	# print imgs_mask_train
	imgs_mask_train /= 255.  # scale masks to [0, 1]

	# print "Training images: "
	# print imgs_train
	# print "Masks: "
	# print imgs_mask_train
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model = get_unet()

	if not load:
		model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss')

		print('-'*30)
		print('Fitting model...')
		print('-'*30)
		history = model.fit(imgs_train, imgs_mask_train, batch_size=8, nb_epoch=1, verbose=1, shuffle=True,
				  validation_split=0.2,
				  callbacks=[model_checkpoint])
		np.save('history.npy', history.history)

	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, imgs_id_test = load_test_data()
	imgs_test = preprocess(imgs_test)

	imgs_test = imgs_test.astype('float32')
	imgs_test -= mean
	imgs_test /= std

	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	model.load_weights('weights.h5')

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	imgs_mask_test = model.predict(imgs_test, verbose=1)

	np.save('imgs_mask_test.npy', imgs_mask_test)
	print imgs_mask_test.shape
	print imgs_id_test.shape
	print('-' * 30)
	print('Saving predicted masks to files...')
	print('-' * 30)
	pred_dir = 'preds'
	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)
	for image, image_id in zip(imgs_mask_test, imgs_id_test):
		print image_id
		image = (image[:, :, 0] * 255.).astype(np.uint8)
		imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
	train_and_predict(load=False)
	dice_loss_test()
