from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

def rbf_kernel_pca(X,gamma,n_components=2):
    '''
    RBF kernel PCA implementation
    
    X: shpe=[n_samples, n_features]
    gamma: float,Turning parameter of RBF kernel
    
    n_components: int.Number of pricipal components to return
    '''
    #Calculate pariwise squared Euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')
    
    #covert pairwise distance into a sequare matrix
    mat_sq_dists = squareform(sq_dists)
    
    #Compute the symmetric kernel martirx
    K=exp(-gamma * mat_sq_dists)
    
    #Center the kernel matrix
    N=K.shape[0]
    one_n = np.ones((N,N))/N
    K=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
    
    gigvals,eigvecs=eigh(K)
    #Collect the top k eigenvectors(projected samples)
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    
    return X_pc

def rbf_kernel_pca1(X,gamma,n_components=2):
    '''
    RBF kernel PCA implementation
    
    X: shpe=[n_samples, n_features]
    gamma: float,Turning parameter of RBF kernel
    
    n_components: int.Number of pricipal components to return
    '''
    #Calculate pariwise squared Euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')
    
    #covert pairwise distance into a sequare matrix
    mat_sq_dists = squareform(sq_dists)
    
    #Compute the symmetric kernel martirx
    K=exp(-gamma * mat_sq_dists)
    
    #Center the kernel matrix
    N=K.shape[0]
    one_n = np.ones((N,N))/N
    K=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
    
    eigvals,eigvecs=eigh(K)
    #Collect the top k eigenvectors(projected samples)
    alphas= np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    
    #Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1,n_components+1)]
    
    return alphas,lambdas


def project_x(x_new,X,gamma,alphas,lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k=np.exp(-gamma*pair_dist)
    
    return k.dot(alphas/lambdas)

plt.figure(1)
X,y = make_moons(n_samples=100, random_state =123)

plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5)
print 'number of X=0 is:%d'%(len(X[y==0]))
#plt.show()

scikit_pca = PCA(n_components=1)
X_spca = scikit_pca.fit_transform(X)


fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X[y==0,0], X[y==0,1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y==0],np.zeros((50))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1],np.zeros((50))-0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')

ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
#plt.show()


X_kpca= rbf_kernel_pca(X,gamma=15,n_components=2)

flg,ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0,1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1,1], color='blue', marker='o', alpha=0.5)


ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
#plt.show()


plt.figure(4)
X,y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1,0], X[y==1,1],color='blue', marker='o', alpha=0.5)
#plt.show()


scikit_pca=PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig1,ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scatter(X_spca[y==0,0],X_spca[y==0,1],color='red',marker='^',alpha=0.5)
ax[0].scatter(X_spca[y==1,0],X_spca[y==1,1],color='blue',marker='o',alpha=0.5)

ax[1].scatter(X_spca[y==0,0],np.zeros((500,1))+0.02,color='red',marker='^',alpha=0.5)
ax[1].scatter(X_spca[y==1,0],np.zeros((500,1))-0.02,color='blue',marker='o',alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
#plt.show()


X_kpca = rbf_kernel_pca(X,gamma=15,n_components=15)
flg1,ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))

ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0,1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1,1], color='blue', marker='o', alpha=0.5)


ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
#plt.show()

X,y = make_moons(n_samples=100, random_state=123)
alphas,lambdas=rbf_kernel_pca1(X,gamma=15,n_components=1)
x_new=X[25]
x_new

x_proj = alphas[25]
print 'The value of new data is:%s'%x_proj

x_reproj = project_x(x_new, X, gamma=15,alphas=alphas,lambdas=lambdas)
print 'The transform of new data is:%s' %x_reproj



X,y = make_moons(n_samples=100,random_state=123)
scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma=15)

plt.figure(7)
X_skernpca = scikit_kpca.fit_transform(X)
plt.scatter(X_skernpca[y==0,0],X_skernpca[y==0,1],color='red',marker='^',alpha=0.5)
plt.scatter(X_skernpca[y==1,0],X_skernpca[y==1,1],color='blue',marker='^',alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

print 'done'