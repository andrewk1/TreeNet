import os
import numpy as np

from skimage.io import imsave, imread
# import matplotlib.pyplot as plt
# import tifffile as tiff

data_path = 'raw/'

image_rows = 480
image_cols = 480

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images, masks = os.listdir(train_data_path)[0], os.listdir(train_data_path)[1]
    total = len(os.listdir(os.path.join(train_data_path, images))) # Num training examples
    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in os.listdir(os.path.join(train_data_path, images)):
        image_mask_name = image_name
        img = imread(os.path.join(train_data_path + '/' + images, image_name))
        # print img.shape
        img_mask = imread(os.path.join(train_data_path + '/' + masks, image_mask_name))
        # img = np.ndarray(shape = (image_rows, image_cols, 3), buffer = img)
        # img_mask = np.ndarray(shape = (image_rows, image_cols, 3), buffer = img_mask)
        imgs[i] = img * 255
        imgs_mask[i] = img_mask
        # print img_mask

        print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images, masks = os.listdir(train_data_path)[0], os.listdir(train_data_path)[1]
    print images
    total = len(os.listdir(os.path.join(train_data_path, images)))
    print total

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=(np.unicode_, 16))
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in os.listdir(os.path.join(train_data_path, images)):
        print image_name
        img_id = (image_name.split('.')[0])
        img = imread(os.path.join(train_data_path + '/' + images, image_name))
        img_mask = imread(os.path.join(train_data_path + '/' + masks, image_name))
        # img = np.array([img])

        imgs[i] = img * 255
        imgs_mask[i] = img_mask
        imgs_id[i] = img_id

        print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('first_run_results/results/orig/imgs_test.npy', imgs)
    np.save('first_run_results/results/orig/imgs_mask_test.npy', imgs_mask)
    np.save('first_run_results/results/orig/imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load(path + 'imgs_test.npy')
    imgs_id = np.load(path + 'imgs_id_test.npy')
    imgs_mask_test = np.load('')
    return imgs_test, imgs_id

if __name__ == '__main__':
    # create_train_data()
    create_test_data()
