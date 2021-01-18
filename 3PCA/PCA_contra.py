import numpy as np
from sklearn.decomposition import PCA
import sys
from sklearn.metrics import roc_auc_score

#returns choosing how many main factors
def index_lst(lst, component=0, rate=0):
    #component: numbers of main factors
    #rate: rate of sum(main factors)/sum(all factors)
    #rate range suggest: (0.8,1)
    #if you choose rate parameter, return index = 0 or less than len(lst)
    if component and rate:
        print('Component and rate must choose only one!')
        sys.exit(0)
    if not component and not rate:
        print('Invalid parameter for numbers of components!')
        sys.exit(0)
    elif component:
        print('Choosing by component, components are %s......'%component)
        return component
    else:
        print('Choosing by rate, rate is %s ......'%rate)
        for i in range(1, len(lst)):
            if sum(lst[:i])/sum(lst) >= rate:
                return i
        return 0

def main():
    # test data
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    from pyod.utils.data import generate_data
    X_train, y_train, X_test, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=20,
                      contamination=0.1,
                      random_state=42)

    mat = X_test

    # simple transform of test data
    Mat = np.array(mat, dtype='float64')
    # print('Before PCA transforMation, data is:\n', Mat)
    print('\nMethod 1: PCA by original algorithm:')
    p,n = np.shape(Mat) # shape of Mat
    t = np.mean(Mat, 0) # mean of each column

    # substract the mean of each column
    for i in range(p):
        for j in range(n):
            Mat[i,j] = float(Mat[i,j]-t[j])

    # covariance Matrix
    cov_Mat = np.dot(Mat.T, Mat)/(p-1)

    # PCA by original algorithm
    # eigvalues and eigenvectors of covariance Matrix with eigvalues descending
    U,V = np.linalg.eigh(cov_Mat)
    # Rearrange the eigenvectors and eigenvalues
    U = U[::-1]
    for i in range(n):
        V[i,:] = V[i,:][::-1]
    # choose eigenvalue by component or rate, not both of them euqal to 0
    Index = index_lst(U, component=2)  # choose how many main factors
    if Index:
        v = V[:,:Index]  # subset of Unitary matrix
    else:  # improper rate choice may return Index=0
        print('Invalid rate choice.\nPlease adjust the rate.')
        print('Rate distribute follows:')
        print([sum(U[:i])/sum(U) for i in range(1, len(U)+1)])
        sys.exit(0)
    # data transformation
    T1 = np.dot(Mat, v)
    # print the transformed data
    # print('We choose %d main factors.'%Index)
    print('After PCA transformation, data becomes:\n',T1)

    # PCA by original algorithm using SVD
    print('\nMethod 2: PCA by original algorithm using SVD:')
    # u: Unitary matrix,  eigenvectors in columns
    # d: list of the singular values, sorted in descending order
    u,d,v = np.linalg.svd(cov_Mat)
    Index = index_lst(d, rate=0.95)  # choose how many main factors
    T2 = np.dot(Mat, u[:,:Index])  # transformed data
    print('We choose %d main factors.'%Index)
    print('After PCA transformation, data becomes:\n',T2)

    # PCA by Scikit-learn
    pca = PCA(n_components=2) # n_components can be integer or float in (0,1)
    pca.fit(mat)  # fit the model
    print('\nMethod 3: PCA by Scikit-learn:')
    print('After PCA transformation, data becomes:')
    print(pca.fit_transform(mat))  # transformed data

main()