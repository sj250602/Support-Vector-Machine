from libsvm.svmutil import *
import numpy as np
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

def demo(args):
    path_of_training_data = input("Enter the path of the training data : ")
    
    train_data = np.genfromtxt(path_of_training_data,dtype = float,delimiter = ",")
    x = input("Enter B for Binary Class and M for Multi Class : ")
    f = int(input("Enter the Number of features you take for comapare : "))
    
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    if(x=='B'):
        a,b = map(int,input("Enter two Classes you want to compare : ").split())

        for i in range(len(train_data)):
            if(train_data[i:i+1,25].astype(np.float32)[0]==a or train_data[i:i+1,25].astype(np.float32)[0]==b):
                list1.append(train_data[i:i+1,0:f].astype(np.float32).flatten())
                list2.append(train_data[i:i+1,25:26].astype(np.float32).flatten())
            else:
                list3.append(train_data[i:i+1,0:f].astype(np.float32).flatten())
                list4.append(train_data[i:i+1,25:26].astype(np.float32).flatten())
    else:
        for i in range(len(train_data)):
            list1.append(train_data[i:i+1,0:f].astype(np.float32).flatten())
            list2.append(train_data[i:i+1,25:26].astype(np.float32).flatten())
    
    X = np.array(list1)
    Y = np.array(list2)
    
    m = int(len(X)*0.75)
    X_train = X[0:m,:]
    X_test = X[m:,:]
    Y_train = Y[0:m,:]
    Y_test = Y[m:,:]
    
    #X_val = np.array(list3)
    #Y_val = np.array(list4)
    
    # parameters = gamma,kernel type,degree
    problem = svm_problem(Y.flatten(),X)
    
    param = svm_parameter("-q") # quiet!
    
    #print(param)
    
    param.gamma = float(input("Enter the gamma parameter : "))
    
    param.kernel_type = int(input("Enter the Kernel Type : "))
    if(param.kernel_type==1):
        param.degree = int(input("Enter the degree of polynomial : "))
    
    param.cross_validation=1
    param.nr_fold=5
    
    model = svm_train(problem,param)
    
    param.cross_validation = 0
    
    problem = svm_problem(Y_train.flatten(), X_train)
    model = svm_train(problem,param)
    
    p_lbl, p_acc, p_prob = svm_predict(Y_test.flatten(),X_test, model)
    
    


if __name__ == '__main__':
    args = setup()
    demo(args)
