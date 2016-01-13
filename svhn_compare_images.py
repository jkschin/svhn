import os
import binascii
import numpy as np
from PIL import Image

#Expect 2 for SVHN and 6 for CIFAR

cifar_dir = "check_images/cifar-10-batches-bin"
svhn_dir = "check_images/svhn-batches-bin"

cifar = open(cifar_dir+"/data_batch_1.bin","rb") 
svhn = open(svhn_dir+"/data_batch_0.bin","rb")


i = 0
svhn_label = svhn.read(1)
svhn_image = svhn.read(3072)
svhn_image = np.frombuffer(svhn_image,dtype=np.uint8)
svhn_image = svhn_image.reshape([3,1024]).T.reshape([32,32,3])
svhn = Image.fromarray(svhn_image,mode="RGB")
print (np.frombuffer(svhn_label,dtype=np.uint8))
svhn.save(open("svhn.jpg","w"))

cifar_label = cifar.read(1)
cifar_image = cifar.read(3072)
cifar_image = np.frombuffer(cifar_image,dtype=np.uint8)
cifar_image = cifar_image.reshape([3,1024]).T.reshape([32,32,3])
cifar = Image.fromarray(cifar_image,mode="RGB")
print (np.frombuffer(cifar_label,dtype=np.uint8))
cifar.save(open("cifar.jpg","w"))

# a = np.array([1,2,3],np.uint8)
# a.tofile(open("test.bin","wb"))
# f = open("test.bin", "rb")
# byte = f.read(1)
# print (binascii.hexlify(byte))