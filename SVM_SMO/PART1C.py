from libsvm.svmutil import *
import numpy as np
import random
import math
import argparse 

def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--method", default="pinv", help = "type of solver")  
    parser.add_argument("--batch_size", default=75, type=int, help = "batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "regularization constant")
    parser.add_argument("--polynomial", default=10, type=int, help = "degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()

def predict(w,point, b,y):
    f = np.dot(w,point) + b
    if((f > 0 and y>0) or (f < 0 and y<0)):
        return 1
    else:
        return 0

def kernel(x1,x2,kt,gamma,degree):
    if(kt==0):
        return np.dot(x1,x2)[0][0]
    elif(kt==1):
        return pow(gamma*np.dot(x1,x2)[0][0],degree)
    elif(kt==2):
        gamma*=-1
        return math.exp(gamma*(np.dot(abs(x1-x2.T),abs(x1.T-x2))[0][0]**2))#exp(-gamma*|u-v|^2)
    else:
        gamma*=-1
        return 1.0 / (1.0 + math.exp(gamma*np.dot(x1,x2)[0][0]))

def takeStep(i,j,X,Y,alpha,E,beta,w,kt,gamma,eps,tol,C,degree):
    if(i==j):
        return 0
    alphaYj = np.multiply(alpha,Y).transpose()
    XjX = np.dot(X,X[j:j+1,:].transpose())
    fXj = np.dot(alphaYj,XjX) + beta
    if(fXj[0][0]>0):
        E[j][0] = 1 - Y[j][0]
    else:
        E[j][0] = -1 - Y[j][0]#(check in error cache)
    
    if(Y[i][0]!=Y[j][0]):
        L = max(0, alpha[j][0] - alpha[i][0])
        H = min(C, C + alpha[j][0] - alpha[i][0])
    else:
        L = max(0, alpha[i][0] + alpha[j][0] - C)
        H = min(C, alpha[i][0] + alpha[j][0])
        
    xixi = kernel(X[i:i+1,:],X[i:i+1,:].transpose(),kt,gamma,degree)
    xixj = kernel(X[i:i+1,:],X[j:j+1,:].transpose(),kt,gamma,degree)
    xjxj = kernel(X[j:j+1,:],X[j:j+1,:].transpose(),kt,gamma,degree)
    old_alpha_i = alpha[i][0] # i = 1,j = 2
    old_alpha_j = alpha[j][0]
    
    eta = xixi+xjxj-2*xixj
    
    if(eta > 0):
        alpha[j][0] +=  (Y[j][0]*(E[i][0]-E[j][0])*1.0)/eta
        #print(alpha[i][0],E[j][0],E[i][0])
        alpha[j][0] = max(alpha[j][0],L)
        alpha[j][0] = min(alpha[j][0],H)
    else:
        F1 = Y[i][0]*(E[i][0] + beta) - alpha[i][0]*xixi - alpha[j][0]*Y[i][0]*Y[j][0]*xixj
        F2 = Y[j][0]*(E[j][0] + beta) - Y[i][0]*Y[j][0]*alpha[i][0]*xixj - alpha[j][0]*xjxj
        L1 = alpha[i][0] + Y[i][0]*Y[j][0](alpha[j][0] - L)
        H1 = alpha[i][0] + Y[i][0]*Y[j][0](alpha[j][0] - H)
        Lobj = LI*F1 + L*F2 + 0.5*L1*L1*xixi + 0.5*L*L*xjxj + Y[i][0]*Y[j][0]*L*L1*xixj
        Hobj = HI*F1 + H*F2 + 0.5*H1*H1*xixi + 0.5*H*H*xjxj + Y[i][0]*Y[j][0]*H*H1*xixj
        if (Lobj < Hobj-tol):
            alpha[j][0] = L
        elif (Lobj > Hobj+tol):
            alpha[j][0] = H
        else:
            alpha[j][0] = old_alpha_j
        
    if (abs(alpha[j][0]-old_alpha_j) < eps*(alpha[j][0]+old_alpha_j+eps)):
        return 0
    
    alpha[i][0] += ((Y[i][0]*Y[j][0])*(old_alpha_j - alpha[j][0]))

    #Update threshold to reflect change in Lagrange multipliers
    
    beta1 = beta + E[i][0] + Y[i][0]*(alpha[i][0] - old_alpha_i)*xixi + Y[j][0]*(alpha[j][0] - old_alpha_j)*xixj
    beta2 = beta + E[j][0] + Y[i][0]*(alpha[i][0] - old_alpha_i)*xixj + Y[j][0]*(alpha[j][0] - old_alpha_j)*xjxj
    
    if(0<alpha[i][0] and alpha[i][0] <C):
        beta = beta1
    elif(0<alpha[j][0] and alpha[j][0] <C):
        beta = beta2
    else:
        beta = (beta1 + beta2)/2.0
        
    #Update weight vector to reflect change in a1 & a2, if SVM is linear
    if(kt==0):
        w += (np.multiply(Y[i][0]*(alpha[i][0] - old_alpha_i),X[i:i+1,:]) + np.multiply(Y[j][0]*(alpha[j][0] - old_alpha_j),X[j:j+1,:]))
    #Update error cache using new Lagrange multipliers
    alphaYj = np.multiply(alpha,Y).transpose()
    XjX = np.dot(X,X[j:j+1,:].transpose())
    fXj = np.dot(alphaYj,XjX) + beta
    if(fXj[0][0]>0):
        E[j][0] = 1 - Y[j][0]
    else:
        E[j][0] = -1 - Y[j][0]#(check in error cache)
        
    alphaYi = np.multiply(alpha,Y).transpose()
    XiX = np.dot(X,X[i:i+1,:].transpose())
    fXi = np.dot(alphaYi,XiX) + beta
    if(fXi[0][0]>0):
        E[i][0] = 1 - Y[i][0]
    else:
        E[i][0] = -1 - Y[i][0]#(check in error cache)
          
    return 1    

def examineExample(i,X,Y,alpha,beta,w,kt,gamma,E,eps,tol,C,degree):
    alphaYi = np.multiply(alpha,Y).transpose()
    XiX = np.dot(X,X[i:i+1,:].transpose())
    fXi = np.dot(alphaYi,XiX) + beta
    if(fXi[0][0]>0):
        E[i][0] = 1 - Y[i][0]
    else:
        E[i][0] = -1 - Y[i][0]#(check in error cache)
        
    if ((Y[i][0]*E[i][0] < -tol and alpha[i][0] < C) or (Y[i][0]*E[i][0] > tol and alpha[i][0] > 0)):
        if (alpha[i][0]!=0 and alpha[i][0]!=C and (alpha[i][0]> 1)):
        
            #i1 = result of second choice heuristic (section 2.2)
            #SMO keeps a cached error value E for every non-bound example in the training set 
            #and then chooses an error to approximately maximize the step size. 
            #If E1 is positive, SMO chooses an example with minimum error E2. If E1 is negative, SMO chooses
            #an example with maximum error E2
            k = 0 
            if(E[i][0]>0):
                error_min = 1000000000000000
                for j in range(len(X)):
                    if(E[j][0]<error_min):
                        k = j
                        error_min = E[j][0]
            else:
                error_min = -100000000000000
                for j in range(len(X)):
                    if(E[j][0]>error_min):
                        k = j
                        error_min = E[j][0]
                
            if takeStep(i,k,X,Y,alpha,E,beta,w,kt,gamma,eps,tol,C,degree):
                return 1
        
        
        #loop over all non-zero and non-C alpha, starting at a random point
        for j in range(len(X)):
            if(alpha[j][0]!=0 and alpha[j][0]!=C):
                if takeStep(i,j,X,Y,alpha,E,beta,w,kt,gamma,eps,tol,C,degree):
                    return 1
        
        #loop over all possible i1, starting at a random point
        for j in range(len(X)):
            if takeStep(i,j,X,Y,alpha,E,beta,w,kt,gamma,eps,tol,C,degree):
                return 1
     
    return 0
  
def fast_SMO(X,Y,m,eps,tol,gamma,kt,C,degree):
    numChanged = 0
    examineAll = 1
    alpha = np.zeros((m,1))
    E = np.zeros((m,1))
    beta = 0
    w = np.zeros((1,len(X[0])))
    while(numChanged > 0 or examineAll):
        numChanged = 0
        if (examineAll):
            #loop I over all training examples
            for I in range(m):
                numChanged += examineExample(I,X,Y,alpha,beta,w,kt,gamma,E,eps,tol,C,degree)
        else:
            #loop I over examples where alpha is not 0 & not C
            for I in range(m):
                if(alpha[I][0]!=0 and alpha[I][0]!=C):
                    numChanged += examineExample(I,X,Y,alpha,beta,w,kt,gamma,E,eps,tol,C,degree)
        if (examineAll == 1):
            examineAll = 0
        elif (numChanged == 0):
            examineAll = 1
    
    if(kt!=0):
        ia = np.multiply(alpha,Y).transpose()
        w = np.dot(ia,X)
    return alpha,beta,w
  
def demo(args):
    path_of_training_data = input("Enter the path of the training data: ")
    
    train_data = np.genfromtxt(path_of_training_data,dtype = float,delimiter = ",")

    a,b = map(int,input("Enter two Classes you want to compare: ").split())
    C = float(input("Enter the Regularization Parameter: "))
    tol = float(input("Numerical Tolerance: "))
    gamma = float(input("Enter the gamma: "))
    eps = float(input("Enter the epsilon: "))
    kt = int(input("Enter the kernel_type: "))
    degree = 0
    if(kt==1):
        degree = int(input("Enter the degree of Polynomial: "))
    
    
    f = int(input("Enter the features: "))
    
    list1 = []
    list2 = []
    for i in range(len(train_data)):
        if(train_data[i:i+1,25].astype(np.float32)[0]==a or train_data[i:i+1,25].astype(np.float32)[0]==b):
            list1.append(train_data[i:i+1,0:f].astype(np.float32).flatten())
            if(a>b):
                if(train_data[i:i+1,25:26][0][0] == a):
                    list2.append([1])
                else:
                    list2.append([-1])
            else:
                if(train_data[i:i+1,25:26][0][0] == b):
                    list2.append([1])
                else:
                    list2.append([-1])

    X = np.array(list1)
    Y = np.array(list2)
    
    m = int(len(X)*0.75)
    X_train = X[0:m,:]
    X_test = X[m:,:]
    Y_train = Y[0:m,:]
    Y_test = Y[m:,:]
    
    alpha,beta,weight = fast_SMO(X_train,Y_train,m,eps,tol,gamma,kt,C,degree)
    
    
    accuracy = 0
    for i in range(len(X_test)):
        accuracy+=predict(weight,X_test[i:i+1,:].transpose(),beta,Y_test[i][0])

    print(f"Accuracy = {accuracy*100.0/len(X_test)}%")

if __name__ == '__main__':
    args = setup()
    demo(args)