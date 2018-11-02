# gan.py
import numpy as np
import os
import cv2

from utils import *

epsilon = 10e-8

class GAN(object):
	def __init__(self, numbers):
		self.numbers = numbers
		self.epochs = 100
		self.batch_size = 64
		self.learning_rate = 0.0001
		self.decay = 0.001

		self.img_path = './images'		

		if not os.path.exists(self.img_path):
			os.makedirs(self.img_path)

		self.g_W0 = np.random.randn(100, 128).astype(np.float32) * np.sqrt(2.0/ (100))
		self.g_b0 = np.zeros(128).astype(np.float32)

		self.g_W1 = np.random.randn(128, 784).astype(np.float32) * np.sqrt(2.0/ (128))
		self.g_b1 = np.zeros(784).astype(np.float32)

		self.d_W0 = np.random.randn(784, 128).astype(np.float32) * np.sqrt(2.0/ (784))
		self.d_b0 = np.zeros(128).astype(np.float32)

		self.d_W1 = np.random.randn(128, 1).astype(np.float32) * np.sqrt(2.0/ (128))
		self.d_b1 = np.zeros(1).astype(np.float32)

	def generator(self, z):
		self.z = np.reshape(z, (self.batch_size, -1))
		self.g_h0_l = self.z.dot(self.g_W0) + self.g_b0
		self.g_h0_a = lrelu(self.g_h0_l)
		self.g_h1_l = self.g_h0_a.dot(self.g_W1) + self.g_b1
		self.g_h1_a = tanh(self.g_h1_l)
		self.g_out = np.reshape(self.g_h1_a, (self.batch_size, 28, 28))
		return self.g_h1_l, self.g_out

	def discriminator(self,img):
		self.d_input = np.reshape(img, (self.batch_size, -1))

		self.d_h0_l = self.d_input.dot(self.d_W0) + self.d_b0
		self.d_h0_a = lrelu(self.d_h0_l)

		self.d_h1_l = self.d_h0_a.dot(self.d_W1) + self.d_b1
		self.d_h1_a = sigmoid(self.d_h0_l)
		self.d_out = self.d_h1_a
		return self.d_h1_l, self.d_out

	def backprop_gan(self, fake_logit, fake_output, fake_input):
		# fake_logit : logit output from the discriminator D(G(z))
		# fake_output : sigmoid output from the discriminator D(G(z))
		# flatten fake image input
		fake_input = np.reshape(fake_input, (self.batch_size, -1))

		# calculate the derivative(导数) of the loss term -log(D(G(z)))
		g_loss = np.reshape(fake_output, (self.batch_size, -1))
		g_loss = (-1.0)/(g_loss + epsilon)

		# calculate the gradients from the end of the discriminatoe
		# we calculate them but won't update the discriminator weigths
		loss_deriv = g_loss*sigmoid(fake_logit, derivative=True)
		loss_deriv = loss_deriv.dot(self.d_W1.T)

		loss_deriv = loss_deriv * lrelu(self.d_h0_l, derivative=True)
		loss_deriv = loss_deriv.dot(self.d_W0.T)

		# calculate the gradients of generator
		loss_deriv = loss_deriv * tanh(self.g_h1_l, derivative = True)
		prev_layer = np.expand_dims(self.g_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_W1 = np.matmul(prev_layer, loss_deriv_)
		grad_b1 = loss_deriv

		loss_deriv =loss_deriv.dot(self.g_W1.T)

		loss_deriv = loss_deriv * lrelu(self.g_h0_l, derivative = True)
		prev_layer = np.expand_dims(self.z, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_W0 = np.matmul(prev_layer, loss_deriv_)
		grad_b0 = loss_deriv

		# calculate all the gradients in the batch
		for idx in range(self.batch_size):
			self.g_W0 = self.g_W0 - self.learning_rate*grad_W0[idx]
			self.g_b0 = self.g_b0 - self.learning_rate*grad_b0[idx]

			self.g_W1 = self.g_W1 - self.learning_rate*grad_W1[idx]
			self.g_b1 = self.g_b1 - self.learning_rate*grad_b1[idx]

	def backprop_dis(self, real_logit, real_output, real_input, fake_logit, fake_output, fake_input):
		# flatten real image input
		# flatten fake image input
		real_input = np.reshape(real_input, (self.batch_size, -1))
		fake_input = np.reshape(fake_input, (self.batch_size, -1))

		# calculate discriminator loss = -np.mean(log(D(x)) + log(1- D(G(z))))
		d_real_loss = -1.0 / (real_output + epsilon)
		d_fake_loss = -1.0 / (fake_output - 1.0  + epsilon)

		# calculate the gradients from the end of the discriminator
		# ###########################
		#   real input graditents
		#   -log(D(x))
		# ############################
		loss_deriv = d_real_loss * sigmoid(real_logit, derivative = True)
		prev_layer = np.expand_dims(self.d_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_real_W1 = np.matmul(prev_layer, loss_deriv_)
		grad_real_b1 = loss_deriv

		loss_deriv = loss_deriv.dot(self.d_W1)
		# loss_deriv = loss_deriv.dot(self.d_W1.T)

		loss_deriv = loss_deriv * lrelu(self.d_h0_l, derivative = True)
		prev_layer = np.expand_dims(real_input, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_real_W0 = np.matmul(prev_layer, loss_deriv_)
		grad_real_b0 = loss_deriv

		# ###########################
		#   fake input graditents
		#   -log(1 - D(G(z)))
		# ############################

		loss_deriv = d_fake_loss*sigmoid(fake_logit, derivative=True)
		prev_layer = np.expand_dims(self.d_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_fake_W1 = np.matmul(prev_layer, loss_deriv_)
		grad_fake_b1 = loss_deriv

		loss_deriv = loss_deriv.dot(self.d_W1)
		# loss_deriv = loss_deriv.dot(self.d_W1.T)

		loss_deriv = loss_deriv * lrelu(self.d_h0_l, derivative = True)
		prev_layer = np.expand_dims(fake_input, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_fake_W0 = np.matmul(prev_layer, loss_deriv_)
		grad_fake_b0 = loss_deriv

		# combine two gradients(real + fake)

		grad_W1 = grad_real_W1 + grad_fake_W1
		grad_b1 = grad_real_b1 + grad_fake_b1
		grad_W0 = grad_real_W0 + grad_fake_W0
		grad_b0 = grad_real_b0 + grad_fake_b0


		# calculate all the gradients in the batch
		for idx in range(self.batch_size):
			self.d_W0 = self.d_W0 - self.learning_rate*grad_W0[idx]
			self.d_b0 = self.d_b0 - self.learning_rate*grad_b0[idx]

			self.d_W1 = self.d_W1 - self.learning_rate*grad_W1[idx]
			self.d_b1 = self.d_b1 - self.learning_rate*grad_b1[idx]

	def train(self):
		trainX, _ , train_size = mnist_reader(self.numbers)
		np.random.shuffle(trainX)

		batch_idx = train_size // self.batch_size
		for epoch in range(self.epochs):
			for idx in range(batch_idx):
				train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]

				if train_batch.shape[0] != self.batch_size:
					break

				z = np.random.uniform(-1, 1, [self.batch_size, 100])

				g_logits, fake_img = self.generator(z)

				d_real_logits, d_real_output = self.discriminator(train_batch)
				d_fake_logits, d_fake_output = self.discriminator(fake_img)

				# cross entropy loss using sigmoid output
				# add epsilon to avoid overflow
				# maximize discriminator loss = -np.mean(log(D(x)) + log(1- D(G(z))))
				d_loss = -np.log(d_real_output + epsilon) - np.log(1 - d_fake_output + epsilon)

				# generator loss
				# ver1: minimize log(1 - D(G(z)))
				# ver2: maximize -log(D(G(z)))
				g_loss = -np.log(d_fake_output + epsilon)

				# backward pass
				self.backprop_dis(d_real_logits, d_real_output, train_batch, d_fake_logits, d_fake_output, fake_img)
				# generator backward pass
				self.backprop_gen(d_fake_logits, d_fake_output, fake_img)
				# if you want to train generator twice
				# g_logits, fake_img = self.generator(z)
				# d_fake_logits, d_fake_output = self.discriminator(fake_img)
				# self.backprop_gen(d_fake_logits, d_fake_output, fake_img)
				
				# show res images 
				img_tile(np.array(fake_img), self.img_path, epoch, idx, "res", False)
				self.img = fake_img

				print("Epoch [%d] step [%d] G Loss:%.4f D Loss:%.4f, Real Ave:%.4f Fake Ave:%.4f, lr:%.4f" % 
					(epoch, idx, np.mean(g_loss), np.mean(d_loss), np.mean(d_real_output), np.mean(d_fake_output), self.learning_rate))

				self.learning_rate = self.learning_rate * (1.0/(1.0 + self.decay * epoch))

				img_tile(np.array(self.img), self.img_path, epoch, idx, "res", True)

numbers = [2]

gan = GAN(numbers)
gan.train()