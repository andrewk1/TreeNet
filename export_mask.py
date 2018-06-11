# Adapted from visoft's export mask script https://www.kaggle.com/visoft

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

# The code is for python 2.7. Parts of it are taken from other posts/kernels.
# Good luck!
def _get_image_names(base_path, imageId):
	'''
	Get the names of the tiff files
	'''
	d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
		 'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
		 'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
		 'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
		 }
	return d


def _convert_coordinates_to_raster(coords, img_size, xymax):
	Xmax,Ymax = xymax
	H,W = img_size
	W1 = 1.0*W*W/(W+1)
	H1 = 1.0*H*H/(H+1)
	xf = W1/Xmax
	yf = H1/Ymax
	coords[:,1] *= yf
	coords[:,0] *= xf
	coords_int = np.round(coords).astype(np.int32)
	return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
	xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
	return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
	df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
	multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
	polygonList = None
	if len(multipoly_def) > 0:
		assert len(multipoly_def) == 1
		polygonList = wkt_loads(multipoly_def.values[0])
	return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
	perim_list = []
	interior_list = []
	if polygonList is None:
		return None
	for k in range(len(polygonList)):
		poly = polygonList[k]
		perim = np.array(list(poly.exterior.coords))
		perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
		perim_list.append(perim_c)
		for pi in poly.interiors:
			interior = np.array(list(pi.coords))
			interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
			interior_list.append(interior_c)
	return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value = 1):
	img_mask = np.zeros(raster_img_size,np.uint8)
	if contours is None:
		return img_mask
	perim_list,interior_list = contours
	cv2.fillPoly(img_mask,perim_list,class_value)
	cv2.fillPoly(img_mask,interior_list,0)
	return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
									 wkt_list_pandas):
	xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
	polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
	contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
	mask = _plot_mask_from_contours(raster_size,contours,1)
	return mask

def resize_tiff(file):
	img = tiff.imread(file)
	return cv2.resize(img, (480, 480)) 

if __name__ == '__main__':
	inDir = '../input'
	# read the training data from train_wkt_v4.csv
	df = pd.read_csv('train_wkt_v4.csv')

	# grid size will also be needed later..
	gs = pd.read_csv('grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

	for img in os.listdir('three_band'):
		img_id = img[:-4]
		mask = generate_mask_for_image_and_class((3600, 3600), img_id,5,gs,df)
		cv2.imwrite('480_tiff_img/' + img_id + '.tiff',mask*255)
		cv2.imwrite('480_tiff_mask/' + img_id + '.tiff', resize_tiff('three_band/' + img))


