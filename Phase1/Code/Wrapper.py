#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import sqrt, pi, reshape, sin, cos
from sklearn.cluster import KMeans
from typing import List, Tuple


def display_and_save_filters(masks: List[List[List]], rows, cols, file_name) -> None:
	"""
	Displays generated filters together and save as an image.
	"""
	for i in range(len(masks)):
		plt.subplot(rows, cols, i+1)
		plt.axis('off')
		plt.imshow(masks[i], cmap='gray')

	plt.savefig(file_name)
	plt.close()

def convert_pngs_to_jpgs(png_folder: str, jpg_folder_name: str) -> str:
	folder = f"./{jpg_folder_name}/"
	if not os.path.exists(folder):
		os.mkdir(jpg_folder_name)

		for file in os.listdir(png_folder):
			image = cv2.imread(png_folder + file)
			cv2.imwrite(
				f'{jpg_folder_name}/{file.split(".")[0]}.jpg',
				image,
				[int(cv2.IMWRITE_JPEG_QUALITY), 100]
			)


def get_2d_gaussian(
		grid: [List[List[float]]], sigma, elong_factor: float = 1
) -> [List[List[float]]]:
	"""
	Calculates 2d gaussian using function defn. at each point (x, y) in grid
	"""
	x, y = grid[0], grid[1]

	sigma_y = sigma
	sigma_x = elong_factor * sigma_y

	num = np.exp(-0.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2))
	denom = np.sqrt(2 * pi * sigma_x * sigma_y)

	return num / denom

def get_dog_filter_bank(
		size: int = 7, sigma_scales: Tuple = (1, np.sqrt(2)), angles: int = 16
) ->  List[List[List[float]]]:
	"""
	Generates the DoG filter bank using sobel operator
	"""
	sobel_x = np.array([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]])

	bound = (size - 1) / 2
	spread = np.linspace(-bound, bound, size)
	x, y = np.meshgrid(spread, spread)
	pts = [x.flatten(), y.flatten()]

	center = (int(size / 2), int(size / 2))
	filter_shape = (size, size)

	DoGs = []
	for sigma in sigma_scales:
		g = get_2d_gaussian(pts, sigma)
		g_2d = reshape(g, filter_shape)
		g_convolved = cv2.filter2D(src=g_2d, ddepth=-1, kernel=sobel_x)

		for i in range(angles):
			angle = i * 360 / angles
			r_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
			dog_filter = cv2.warpAffine(src=g_convolved, M=r_mat, dsize=filter_shape)
			DoGs.append(dog_filter)

	return DoGs

def get_gabor_filter_bank(
		sigma_scales: List, angles, Lambda: int = 30, psi: int = 30, gamma: int = 1
) -> [List[List[float]]]:
	"""
	Generates gabor filter bank using function defn. at each point (x, y) in grid.
	Source: Wikipedia
	"""
	def _gabor(sigma, theta, Lambda, psi, gamma):
		sigma_x = sigma
		sigma_y = float(sigma) / gamma

		nstds = 3  # Number of standard deviation sigma
		xmax = max(abs(nstds * sigma_x * np.cos(theta)),
				   abs(nstds * sigma_y * np.sin(theta)))
		xmax = np.ceil(max(1, xmax))

		ymax = max(abs(nstds * sigma_x * np.sin(theta)),
				   abs(nstds * sigma_y * np.cos(theta)))
		ymax = np.ceil(max(1, ymax))

		xmin = -xmax
		ymin = -ymax

		(x, y) = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))

		# Rotation
		x_theta = x * np.cos(theta) + y * np.sin(theta)
		y_theta = -x * np.sin(theta) + y * np.cos(theta)

		return np.exp(
			-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)
		) * np.cos(2 * np.pi / Lambda * x_theta + psi)

	gabor_bank = []
	for sigma in sigma_scales:
		for i in range(angles):
			theta = i * np.pi / angles
			gb = _gabor(sigma, theta, Lambda, psi, gamma)
			gabor_bank.append(gb)

	return gabor_bank

def gaussian1d(sigma, mean, x, ord):
	x = np.array(x)
	x_ = x - mean
	var = sigma**2

	# Gaussian Function
	g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

	if ord == 0:
		g = g1
		return g
	elif ord == 1:
		g = -g1*(x_/var)
		return g
	else:
		g = g1*(((x_*x_) - var)/(var**2))
		return g

def gaussian2d(size, scales):
	var = scales * scales
	shape = (size, size)
	n,m = [(i - 1)/2 for i in shape]
	x,y = np.ogrid[-m:m+1,-n:n+1]
	g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
	return g

def log2d(size, scales):
	var = scales * scales
	shape = (size, size)
	n,m = [(i - 1)/2 for i in shape]
	x,y = np.ogrid[-m:m+1,-n:n+1]
	g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
	h = g*((x*x + y*y) - var)/(var**2)
	return h

def make_filter(scale, phasex, phasey, pts, size):
	gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
	gy = gaussian1d(scale,   0, pts[1,...], phasey)

	return np.reshape(gx * gy, (size, size))

def get_lm_filter_bank(sigma_scales: List, size: int = 49, angles: int = 6):
	"""
	Get LM filter bank.
	Source: https://www.robots.ox.ac.uk/~vgg/research/texclass/code/makeLMfilters.m
	"""
	first_ord_deriv = sec_ord_deriv = len(sigma_scales) * angles
	gauss_and_log = 12

	total_filters = first_ord_deriv + sec_ord_deriv + gauss_and_log
	F = np.zeros([size, size, total_filters])

	bound  = (size - 1)/2
	x = [np.arange(-bound, bound + 1)]
	y = [np.arange(-bound, bound + 1)]

	[x,y] = np.meshgrid(x,y)
	orgpts = np.array([x.flatten(), y.flatten()])

	count = 0
	for scale in range(len(sigma_scales)):
		for i in range(angles):
			angle = (np.pi * i) / angles
			rotpts = np.array([[cos(angle),-sin(angle)],[sin(angle),cos(angle)]])
			rotpts = np.dot(rotpts,orgpts)
			F[:, :, count] = make_filter(sigma_scales[scale], 0, 1, rotpts, size)
			F[:, :, count + sec_ord_deriv] = make_filter(sigma_scales[scale], 0, 2, rotpts, size)
			count += 1

	count = first_ord_deriv + sec_ord_deriv
	scales = np.sqrt(2) * np.array([1,2,3,4])
	for i in range(len(scales)):
		F[:,:,count] = gaussian2d(size, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:,:,count] = log2d(size, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:,:,count] = log2d(size, 3*scales[i])
		count = count + 1

	LM_bank = []
	for i in range(48):  # Length of filter bank is constant
		LM_bank.append(F[:, :, i])

	return LM_bank


def get_half_disk_masks(sizes: List[int], angles: int) -> [List[List[float]]]:
	"""
	Generates half disk masks at given scales and angles
	"""
	half_disks = []
	for size in sizes:
		bound = (size - 1) / 2
		spread = np.linspace(-bound, bound, size)
		x, y = np.meshgrid(spread, spread)
		pts = [x.flatten(), y.flatten()]

		radius = size / 2
		for i in range(angles):
			angle = (2 * np.pi * i) / angles
			rot = np.array([[np.cos(angle), -np.sin(angle)],
							[np.sin(angle), np.cos(angle)]])
			rot_pts = np.dot(rot, pts)
			x, y = rot_pts[0], rot_pts[1]
			cond1 = (x ** 2 + y ** 2) < radius ** 2
			cond2 = y > 0
			mask = np.logical_and(cond1, cond2)
			mask = mask.reshape(size, size)
			half_disks.append(mask)
	return half_disks


def normalise_img(img: List[List]) -> List[List]:
	"""
	Normalises image in the range 0 - 255.
	"""
	old_min, old_max = np.min(img), np.max(img)
	old_range = old_max - old_min

	old_range = old_range if old_range else 1
	img = 255 * (img - old_min) / old_range
	return img


def get_gradient_map(img_map: List[List], bins, half_disc_masks: List[List[List]]) -> List[List]:
	"""
	Gets gradient map using chi-squared distance calculations for given map of image.
	"""
	l_masks = half_disc_masks[0]
	r_masks = half_disc_masks[1]

	chi_sqr_distances = []
	for i in range(len(l_masks)):
		chi_sqr_dist = np.zeros(img_map.shape)
		for val in range(bins):
			bin_img = (img_map == val)
			bin_img = np.float32(bin_img)

			gi = cv2.filter2D(src=bin_img, ddepth=-1, kernel=np.float32(l_masks[i]))
			hi = cv2.filter2D(src=bin_img, ddepth=-1, kernel=np.float32(r_masks[i]))

			chi_sqr_dist += ((gi - hi) ** 2 )/2 * (gi + hi)
		chi_sqr_distances.append(chi_sqr_dist)

	chi_sqr_distances = np.array(chi_sqr_distances)
	return np.mean(chi_sqr_distances, axis=0)


def main():

	images_folder = "./BSDS500/Images/"
	image_files = os.listdir(images_folder)

	results_folder = "./Results/"
	if not os.path.exists(results_folder):
		os.mkdir("Results")

	sobel_bl_jpg_folder = "Sobel_jpgs"
	canny_bl_jpg_folder = "Canny_jpgs"

	convert_pngs_to_jpgs("./BSDS500/SobelBaseline/", sobel_bl_jpg_folder)
	convert_pngs_to_jpgs("./BSDS500/CannyBaseline/", canny_bl_jpg_folder)

	texton_bins = 64
	brightness_bins = 16
	color_bins = 16

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	D = get_dog_filter_bank()
	# display_and_save_filters(D, 2, 16, results_folder + "DoG.png")

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	scales_small = np.sqrt(2) * np.array([1,2,3])
	scales_large = np.sqrt(2) * np.array([1,2,3,4])
	L = get_lm_filter_bank(scales_small, 49, 6)
	L.extend(get_lm_filter_bank(scales_large, 49, 6))

	display_and_save_filters(L, 16, 6, results_folder + "LM.png")

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	G = get_gabor_filter_bank([18, 24, 30, 36, 42], 8)
	# display_and_save_filters(G, 5, 8, results_folder + "Gabor.png")

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	HD = get_half_disk_masks([10, 26, 50], 16)
	left_hd_masks = [*HD[0:8], *HD[16:24], *HD[32:40]]
	right_hd_masks = [*HD[8:16], *HD[24:32], *HD[40:48]]
	hd_masks = [left_hd_masks, right_hd_masks]
	# display_and_save_filters(HD, 6, 8, results_folder + "HDMasks.png")

	# Loop for all images
	for img_name in image_files:
		img_path = images_folder + img_name
		img = cv2.imread(img_path)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		img_channels = []
		for filt in [*D, *L, *G]:
			img_conv = cv2.filter2D(src=img_gray, ddepth=-1, kernel=filt)
			img_channels.append(normalise_img(img_conv))

		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		img_channels = np.array(img_channels)

		d, w, h = img_channels.shape
		pixels = img_channels.reshape(d, w * h).transpose()

		km = KMeans(n_clusters=64, n_init=2)
		labels = km.fit_predict(pixels)
		texton_map = labels.reshape([w, h])

		plt.imsave(results_folder + "TextonMap_" + img_name, texton_map)

		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Tg = get_gradient_map(texton_map, texton_bins, hd_masks)
		# plt.imshow(Tg)
		plt.imsave(results_folder + "Tg_" + img_name, Tg)

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		img_gray = np.array(img_gray)
		w, h = img_gray.shape
		pixels = img_gray.reshape(w*h, 1)

		km = KMeans(n_clusters=16, n_init=2)
		labels = km.fit_predict(pixels)
		brightness_map = labels.reshape([w, h])

		plt.imsave(results_folder + "BrightnessMap_" + img_name, brightness_map)

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Bg = get_gradient_map(brightness_map, brightness_bins, hd_masks)
		# plt.imshow(Bg)
		plt.imsave(results_folder + "Bg_" + img_name, Bg)

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		w, h, d = img.shape
		pixels = img.reshape(w * h, d)

		km = KMeans(n_clusters=16, n_init=2)
		labels = km.fit_predict(pixels)
		color_map = labels.reshape([w, h])

		plt.imsave(results_folder + "ColorMap_" + img_name, color_map)

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Cg = get_gradient_map(color_map, color_bins, hd_masks)
		# plt.imshow(Cg)
		plt.imsave(results_folder + "Cg_" + img_name, Cg)

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobel_pb = cv2.imread("./" + sobel_bl_jpg_folder + "/" + img_name)
		sobel_pb = cv2.cvtColor(sobel_pb, cv2.COLOR_BGR2GRAY)

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		canny_pb = cv2.imread("./" + canny_bl_jpg_folder + "/" + img_name)
		canny_pb = cv2.cvtColor(canny_pb, cv2.COLOR_BGR2GRAY)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		pb_edges = np.array((Tg + Bg + Cg)/3) * np.array(0.5 * sobel_pb + 0.5 * canny_pb)
		plt.imsave(results_folder + "PbLite_" + img_name, pb_edges, cmap='gray')


if __name__ == '__main__':
	main()
