import numpy as np 
from PIL import Image 
import cv2

# define avtivation functions
# derivative 求导

def sigmoid(input, derivative=False):
	res = 1/(1 + np.exp(-input))
	if derivative:
		return res * (1 - res)
	return res

def relu(input, derivative=False):
	res = input
	if derivative:
		return 1.0 * (res > 0)
		# res > 0 : 1 * 1 -> 1
		# res < 0 : 1 * 0 -> 0
	else:
		return res * (res > 0)
		# res > 0 : res * 1 -> res
		# res < 0 : res * 0 -> 0

def lrelu(input, alpha = 0.01, derivative = False):
	res = input
	if derivative:
		dx = np.ones_like(res)
		dx[res < 0] = alpha
		# res < 0, dx[1] = alpha
		# res > 0, dx[0] = alpha
		return dx
	else:
		return np.maximum(input, input * alpha, input)

def tanh(input, derivative=False):
	res = np.tanh(input)
	if derivative:
		return 1.0 - np.tanh(input) ** 2
	return res

def mnist_reader(numbers):
	def one_hot(label, output_dim):
		one_hot = np.zeros((len(label), output_dim))

		for idx in range(0, len(label)):
			one_hot[idx, label[idx]] = 1
		return one_hot

	# Train data
	f = open('./data/train-images.idx3-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32) / 127.54 - 1

	f = open('./data/train-labels.idx1-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)

	newtrainX = []
	for idx in range(0, len(trainX)):
		if trainY[idx] in numbers:
			newtrainX.append(trainX[idx])
	return np.array(newtrainX), trainY, len(trainX)

def img_tile(imgs, path, epoch, step, name, save, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
	if imgs.ndim != 3 and imgs.ndim != 4:
		raise ValueError('imgs has wrong number of dimensions')
		n_imgs = imgs.shape[0]

		tile_shape = None
		img_shape = np.array(img.shape[1:3])
		if tile_shape is None:
			img_aspect_ratio = img_shape[1] / float(img_shape[0])
			aspect_ratio *= img_aspect_ratio
			tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
			tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
			grid_shape = np.array((tile_height, tile_width))
		else:
			assert len(tile_shape) == 2
			grid_shape = np.array(tile_shape)

		 # Tile image shape
		tile_img_shape = np.array(imgs.shape[1:])
		tile_img[:] = border_color
		for i in range(grid_shape[0]):
			for j in range(grid_shape[1]):
				img_idx = j + i * grid_shape[1]
				if img_idx >= n_imgs:
					break
				img = (imgs[img_idx] + 1)/2.0

				yoff = (img_shape[0] + border) * i 
				xoff = (img_shape[1] + border) * j
				tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

		path_name = path + '/epoch_%03d'%(epoch) + '.jpg'

		# change code below if you want to save results using PIL
		tile_img = cv2.resize(tile_img, (256, 256))
		cv2.imshow(name, tile_img)
		cv2.waitKey(1)
		if save:
			cv2.imwrite(path_name, tile_img * 255)