from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.ndim
train_images.dtype
train_images.shape


digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show

my_slice = train_images[10:100]
my_slice.shape

my_slice = train_images[10:100, :, :]
my_slice.shape

my_slice = train_images[10:100, 0:28, 0:28]
my_slice.shape
