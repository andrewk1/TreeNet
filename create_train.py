import os, random

if __name__ == '__main__':
	for i in xrange(200):
		img = random.choice(os.listdir('sliced_images')) #change dir name to whatever
		os.rename('sliced_images/' + img, 'sliced_test/images/' + img)
		os.rename('sliced_masks/' + img, 'sliced_test/masks/' + img)