import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh
import seaborn as sn

classes = 10

# path to data
p_train_images = 'E:\\IIITD\\Semester 2\\SML\\Assignments\\Assignment 2\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
p_train_labels = 'E:\\IIITD\\Semester 2\\SML\\Assignments\\Assignment 2\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'


# LOADING DATASET - reading idx and converting to numpy
images = idx2numpy.convert_from_file(p_train_images)
labels = idx2numpy.convert_from_file(p_train_labels)

print("Number of samples:", len(images))
# normalizing data
images = images.astype('float')/255


# showing first N images
N = 5
for i in range(N):
    img = images[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
    
# Generate Gaussian noise

# for img in images:
#     noise = np.random.randint(5, size = (28, 28), dtype = 'uint8')
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             img[i][j]+=noise[i][j]

# Adding Noise
noise_factor = 0.1
for k in range(N):
    for i in range(len(images[k])):
        for j in range(len(images[k][i])):
            images[k][i][j]+=noise_factor*np.random.normal(loc=0.0, scale=1.0, size=1) 

# showing first N Noisy images
for i in range(N):
    img = images[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
    
# vectorizing data
images = images.reshape(images.shape[0], -1)
print("Vectorized Data:", images.shape)
# for x in images[0]:
#     print(x, ",")


# mean calculation with data frame
mean = []
image_df = pd.DataFrame(images)
for i in image_df.columns:
    value = np.array(image_df[i])
    mean.append(value.mean())
mean = np.asarray(mean)
print("Mean of the data: ",mean)

# standard deviation
sigma = images.std()
print("Standard Deviation:", sigma)

# images['label'] = labels
# print(image_df)
# print(images)

# mean_vectors = []
# cl = np.unique(labels)
# # Compute class wise mean
# for c in cl: 
#     mean_vectors.append(np.mean(images[labels==c], axis= 0))
# print("Class wise Means computed: ", len(mean_vector))

covariance = np.matmul(images.T , images)
print("Covariance Matrix", covariance)


# # Data Centralization -> (x-mean)/sigma
# for i in range(len(images)):
#     images[i] = np.divide(np.subtract(images[i], mean), sigma)


last = (len(images))
eigenvalues, eigenvectors = eigh(covariance)
eigenvalues = eigenvalues[-200]
eigenvectors = eigenvectors[:,-200:]

# PCA
reduced_data = np.matmul(eigenvectors.T, images.T)

print(eigenvalues.shape, eigenvectors.shape, images.shape, reduced_data.shape)

# retreiving images after noise reduction
res_images = np.matmul(reduced_data.T, eigenvectors.T)
#res_images = res_images.T

print(res_images.shape)

# =============================================================================
# res_images = res_images.reshape((60000, 28, 28))
# print(res_images.shape)
# 
# =============================================================================
# showing first N Noise reduced images
for i in range(N):
    img = res_images[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()

