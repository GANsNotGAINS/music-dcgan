import glob
import matplotlib.pyplot as plt
import os

data_dir = "poly/"
output_dir = "poly24/"
files = glob.glob(data_dir + "*.png")
length = 24
keep_first_row = False

for file in files:
	img = plt.imread(file)
	rows, cols, z = img.shape
	for i in range(cols // length):
		start = i * length 
		if keep_first_row:
			cropped = img[:, start: start+length]
		else:
			cropped = img[1:, start: start+length]

		print(cropped.shape)

		fname = file.split("/")[-1]
		fname = os.path.splitext(fname)[0]
		fname = "{0}{1}-{2}.png".format(output_dir, fname, i)

		plt.imsave(fname, cropped) 
