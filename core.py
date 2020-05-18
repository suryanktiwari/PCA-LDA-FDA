import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh
import seaborn as sn
import random

# path to data
p_train_images = 'E:\\IIITD\\Semester 2\\SML\\Assignments\\Assignment 2\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
p_train_labels = 'E:\\IIITD\\Semester 2\\SML\\Assignments\\Assignment 2\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
p_test_images = 'E:\\IIITD\\Semester 2\\SML\\Assignments\\Assignment 2\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
p_test_labels = 'E:\\IIITD\\Semester 2\\SML\\Assignments\\Assignment 2\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'

# LOADING DATASET - reading idx and converting to numpy
images = idx2numpy.convert_from_file(p_train_images)
labels = idx2numpy.convert_from_file(p_train_labels)
test_images = idx2numpy.convert_from_file(p_test_images)
test_labels = idx2numpy.convert_from_file(p_test_labels)
print("Number of train samples:", len(images))
print("Number of test samples:", len(test_images))

# normalizing data
images = images.astype('float')/255
test_images = test_images.astype('float')/255

# showing sample images
for i in range(3):
    img = images[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()

# vectorizing data
images = images.reshape(images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
#print("Vectorized Data:", images[0])
# for x in images[0]:
#     print(x, ",")


# mean calculation with data frame
def getMean(images):
    mean = []
    image_df = pd.DataFrame(images)
    for i in image_df.columns:
        value = np.array(image_df[i])
        mean.append(value.mean())
    mean = np.asarray(mean)
    return mean

# Compute class wise mean
def getClassWiseMean(images, labels):
    mean_vectors = []
    cl = np.unique(labels)
    for c in cl: 
        mean_vectors.append(np.mean(images[labels==c], axis= 0))
    return mean_vectors
    print("Class wise Means computed")

# Covariance Matrix
def getCovariance(images):
    covariance = np.matmul(images.T , images)
    # to avoid determinant zero, adding some values to diagonals
    np.fill_diagonal(covariance, covariance.diagonal()+0.00001)
    return covariance
print("Covariance Matrix", getCovariance(images))

def PCA(images, labels, top, draw=False):
    covariance = getCovariance(images)
    eigenvalues, eigenvectors = eigh(covariance)
    eigenvalues = eigenvalues[-top:]
    eigenvectors = eigenvectors[:,-top:]
    if draw==True:
        for i in range(7):
            vec = eigenvectors.T[i*100]
            plt.figure()
            fig=plt.imshow(np.asarray(vec).reshape(28,28),origin='upper')
            fig.set_cmap('gray_r')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()
            plt.close()
    # print(eigenvalues.shape, eigenvectors.shape)
    reduced_data = np.matmul(eigenvectors .T, images.T)
    return reduced_data

def FDA(images, labels, top):
    mean = getMean(images)
    #print("Mean Shape: ", mean.shape)
    c_means = getClassWiseMean(images, labels)
    #print("c means Shape: ", len(c_means), c_means[0].shape)
    sb = np.zeros((images.shape[1], images.shape[1]))
    
    for cl, cm in enumerate(c_means):
        samples = images[labels==cl]
        size = samples.shape[0]
        
        cm = cm.reshape(1, images.shape[1])
        centr = cm-mean
        sb += size*np.dot(centr.T, centr)
   
    sw = []
    for cl, cm in enumerate(c_means):
        si = np.zeros((images.shape[1], images.shape[1]))
        samples = images[labels==cl]
        for s in samples:
            temp = s-mean
            temp = temp.reshape(1, images.shape[1])
            si += np.dot(temp.T, temp)
        sw.append(si)
        
    S = np.zeros((images.shape[1], images.shape[1]))
    for si in sw:
        S += si
    
    np.fill_diagonal(S, S.diagonal()+0.0001)
    SI = np.linalg.inv(S)
    W = SI.dot(sb)
    
    eigen_values, eigen_vectors = np.linalg.eigh(W)

    eigen_values = eigen_values[-top:]
    eigen_vectors = eigen_vectors[:,-top:]  
    fda = images.dot(eigen_vectors)
    return fda

def LDA(images, labels):
    mean = getMean(images)
    #print("Mean Shape: ", mean.shape)
    icov = np.linalg.inv(getCovariance(images))
    #print("icov Shape: ", icov.shape)
    images = pd.DataFrame(images)
    c_means = getClassWiseMean(images, labels)
    #print("c means Shape: ", len(c_means), c_means[0].shape)
    classes = np.unique(labels)
    results = []
    discriminant = [0]*len(classes)
    for sample in images:
        for i in range(len(classes)):
            t1 = np.subtract(sample, c_means[i])
            t2 = t1.T
            t2 = np.matmul(t2, icov)
            discriminant[i] = -0.5 * np.matmul(t2, t1)
        results.append(discriminant.index(max(discriminant)))
    count = 0
    for i in range(len(results)):
        if(labels[i]==results[i]):
            count+=1
    print("Accuracy: ", float(count/len(labels)) * 100)
    

mean = getMean(images)
print("Mean of the data:", mean)

# standard deviation
sigma = images.std()
print("Standard Deviation:", sigma)

covariance = getCovariance(images)
print("Covariance Matrix:", covariance)

# =============================================================================
# # Data Centralization -> (x-mean)/sigma
# for i in range(len(images)):
#     images[i] = np.divide(np.subtract(images[i], mean), sigma)    
# =============================================================================

print("\nPCA and then FDA:")
reduced_data = PCA(images, labels, 2)
fda_data = reduced_data.T
reduced_data = np.vstack((reduced_data, labels)).T
dataframe = pd.DataFrame(data=reduced_data, columns=('Principal Axis 1', 'Principal Axis 2', 'label'))
sn.FacetGrid(dataframe, hue='label', height=6).map(plt.scatter, 'Principal Axis 1', 'Principal Axis 2').add_legend()
plt.show()
fda = FDA(fda_data, labels, 2)
plt.scatter(fda[:,0], fda[:,1], c=labels, cmap=plt.cm.Set1)
plt.show()


print("\nPerforming LDA", end=" ")
LDA(test_images, test_labels)

# eigen energy calculation
mean = getMean(test_images)
cur = 0
# =============================================================================
# std = test_images.std()
# for i in range(len(test_images)):
#     test_images[i] = np.divide(np.subtract(test_images[i], mean), std)
# covariance = np.matmul(test_images.T , test_images)
# =============================================================================
eigenvalues, eigenvectors = eigh(covariance)
eigen_sum = sum(eigenvalues)
energies = [0.7, 0.9, 0.95, 0.99]
for energy in energies:
    j = 0
    for i in range(len(eigenvalues)):
        cur += eigenvalues[len(eigenvalues)-i-1]
        if cur/eigen_sum > energy:
            j = i
            break
    result = []
    if energy == 0.99:
        result = PCA(test_images, test_labels, j, draw=True)
    else:
        result = PCA(test_images, test_labels, j)
    print("Accuracy for ", energy, ':', end=" ")
    LDA(result.T, test_labels)
    


print("\nFDA then LDA")
res = FDA(test_images, test_labels, 500)
LDA(res, test_labels)

print("\nPCA to FDA and then LDA")
res = PCA(test_images, test_labels, 500)
fda = FDA(res.T, test_labels, 500)
LDA(fda, test_labels)
