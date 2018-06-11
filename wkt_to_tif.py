# Code adopted from the10wl's kaggle HTML to WKT kernel found at https://www.kaggle.com/the1owl/html5-wkt-to-svg-3
from shapely import wkt, affinity
import shapely
import pandas as pd
import json, geojson
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tifffile as tiff

class SatelliteDataProcess(object):
	"""docstring for SatelliteDataProcess"""
	def __init__(self):
		self.testImageIds = []

	def tiff_to_png(self, file):
		for infile in os.listdir(file):
			print "file : " + infile
			# print "is tif or bmp"
			outfile = infile[:-3] + "tiff"
			im = Image.open('masks/' + infile)
			print "new filename : " + outfile
			out = im.convert("RGB")
			out.save('tiff-masks/' + outfile, "JPEG", quality=100)
			
	def wkt_to_svg(self):
		cls = ['Buildings','Misc.', 'Road' , 'Track', 'Trees', 'Crops', 'Waterway', 'Standing water', 'Vehicle Large', 'Vehicle Small']
		cls_col = ['white','white', 'white' , 'white', 'black', 'white', 'white', 'white', 'white', 'white']

		# Grid sizes indicate true size of satellite images as reference for WKT
		g = pd.read_csv('grid_sizes.csv', names=['ImageId','Xmax','Ymin'])
		# WKT formatted vector representations of classes
		w = pd.read_csv('train_wkt_v4.csv', dtype={'user_id': object})
		d = {}
		for i in range(len(g)):
			d[str(g['ImageId'][i])+'.tif'] = {'x': g['Xmax'][i], 'y': g['Ymin'][i], 'wkt':['','','','','','','','','','']}
		for i in range(len(w)):
			d[str(w['ImageId'][i])+'.tif']['wkt'][w['ClassType'][i] -1 ] = w['MultipolygonWKT'][i]

		print 'woo'
		X_ = float(3360) * (3360/float(3360+1))
		# For each image in 
		ct = 0
		for img in d:
			if d[img]['wkt'] != ['','','','','','','','','','']:                
				with open('svgs/' + img[:-4] + '.svg', 'w') as f:
					f.write("<svg height='3360' width='3360'>")
					ct += 1
					i = 0
					for poly_ in d[img]['wkt']:
						if len(poly_)>0:
							poly = wkt.loads(poly_)
							poly = affinity.scale(poly, xfact= X_/float(d[img]['x']), yfact= X_/float(d[img]['y']), origin=(0,0,0))
							for p in poly:
								f.write("<polygon points='" + " ".join([str(x[0])+','+str(x[1]) for x in p.exterior.coords]) + "' style='fill:" + cls_col[i] + ";stroke:white;stroke-width:1;' />")
						i+=1
					f.write("</svg>")

if __name__ == '__main__':
	d = SatelliteDataProcess()
	d.wkt_to_svg()
