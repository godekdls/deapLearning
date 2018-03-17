import sys, os
sys.path.append(os.pardir)
from mnist.mnist import load_mnist

(image_train, label_train), (image_test, label_test) = load_mnist(flatten=True, normalize=False)

print(image_train.shape)
print(label_train.shape)
print(image_test.shape)
print(label_test.shape)