from libsvm.svmutil import *
import numpy as np
import random
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

def SMO(max_passes,tol,C,X,Y,m,f):
    alpha = np.zeros((m,1))
    beta = 0
    passes = 0
    while(passes<max_passes):
        num_changed_alphas = 0
        for i in range(m):
            alphaYi = np.multiply(alpha,Y).transpose()
            XiX = np.dot(X,X[i:i+1,:].transpose())
            fXi = np.dot(alphaYi,XiX) + beta
            if(fXi[0][0]>0):
                Ei = 1 - Y[i][0]
            else:
                Ei = -1 - Y[i][0]
            if((Y[i][0]*Ei < -tol and alpha[i][0] < C) or (Y[i][0]*Ei > tol and alpha[i][0] > 0)):
                new_list = [el for el in range(m) if el != i]
                j = random.choices(new_list)[0]
                alphaYj = np.multiply(alpha,Y).transpose()                
                XjX = np.dot(X,X[j:j+1,:].transpose())
                fXj = np.dot(alphaYj,XjX) + beta
                if(fXj[0][0]>0):
                    Ej = 1 - Y[j][0]
                else:
                    Ej = -1 - Y[j][0]
                
                xixi = np.dot(X[i:i+1,:],X[i:i+1,:].transpose())[0][0]
                xixj = np.dot(X[i:i+1,:],X[j:j+1,:].transpose())[0][0]
                xjxj = np.dot(X[j:j+1,:],X[j:j+1,:].transpose())[0][0]
                old_alpha_i = alpha[i][0]
                old_alpha_j = alpha[j][0]
                
                
                if(Y[i][0]!=Y[j][0]):
                    L = max(0, alpha[j][0] - alpha[i][0])
                    H = min(C, C + alpha[j][0] - alpha[i][0])
                else:
                    L = max(0, alpha[i][0] + alpha[j][0] - C)
                    H = min(C, alpha[j][0] + alpha[i][0])
                
                if(L==H):
                    continue
                
                eta = (2*xixj - xixi - xjxj)
               
                
                if(eta>=0):
                    continue
                
                alpha[j][0] -= (Y[j][0]*1.0*(Ei-Ej))/eta
                alpha[j][0] = max(alpha[j][0],L)
                alpha[j][0] = min(alpha[j][0],H)
                
                if(abs(alpha[j][0] - old_alpha_j)<0.00001):
                    continue
                
                alpha[i][0] += ((Y[i][0]*Y[j][0])*(old_alpha_j - alpha[j][0]))
                
                beta1 = beta - Ei - Y[i][0]*(alpha[i][0] - old_alpha_i)*xixi - Y[j][0]*(alpha[j][0] - old_alpha_j)*xixj
                beta2 = beta - Ej - Y[i][0]*(alpha[i][0] - old_alpha_i)*xixj - Y[j][0]*(alpha[j][0] - old_alpha_j)*xjxj
                
                if(0<alpha[i][0] and alpha[i][0] <C):
                    beta = beta1
                elif(0<alpha[j][0] and alpha[j][0] <C):
                    beta = beta2
                else:
                    beta = (beta1 + beta2)/2
                    
                num_changed_alphas+=1
        if(num_changed_alphas == 0):
            passes += 1
        else:
            passes = 0

    w = np.zeros((1,f))
    ia = np.multiply(alpha,Y).transpose()
    w = np.dot(ia,X)
    return alpha,beta,w
    
def demo(args):
    path_of_training_data = input("Enter the path of the training data: ")
    
    train_data = np.genfromtxt(path_of_training_data,dtype = float,delimiter = ",")

    a,b = map(int,input("Enter two Classes you want to compare: ").split())
    C = float(input("Enter the Regularization Parameter: "))
    tol = float(input("Numerical Tolerance: "))
    max_passes = int(input("Max # of times to iterate over α’s without changing: "))
    f = int(input("Enter Number of features you want to take: ")) 
    
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
    
    al,be,w = SMO(max_passes,tol,C,X_train,Y_train,m,f)
    accuracy = 0
    for i in range(len(X_test)):
        accuracy+=predict(w,X_test[i:i+1,:].transpose(),be,Y_test[i][0])

    print(f"Accuracy = {accuracy*100.0/len(X_test)}%")
    


if __name__ == '__main__':
    args = setup()
    demo(args)