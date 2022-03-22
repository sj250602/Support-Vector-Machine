from libsvm.svmutil import *
import numpy as np
import csv
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
    train_path = input("Enter the path of training data : ")
    test_path = input("Enter the path of testing data : ")
    train_data = np.genfromtxt(train_path,dtype = float,delimiter = ",")
    test_data = np.genfromtxt(test_path,dtype = float,delimiter = ",")
    
    f = int(input("Eter the features you want to take : "))
    
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    
    for i in range(len(train_data)):
        list1.append(train_data[i:i+1,0:f].flatten())
        list2.append(train_data[i:i+1,25:26].flatten())
    
    for i in range(len(test_data)):
        list3.append(test_data[i:i+1,0:f].flatten())
        list4.append(test_data[i:i+1,25:26].flatten())
    
    X = np.array(list1)
    Y = np.array(list2)
    
    
    X_test_fin = np.array(list3)
    Y_test_fin = np.zeros((len(test_data),1))
    
    # parameters = gamma,kernel type,degree
    problem = svm_problem(Y.flatten(),X)
    
    param = svm_parameter("-q") # quiet!
    
    param.gamma = float(input("Enter the gamma parameter: "))
    
    param.kernel_type = int(input("Enter the Kernel Type: "))
    if(param.kernel_type==1):
        param.degree = int(input("Enter the degree of polynomial: "))
    
    param.cross_validation=1
    param.nr_fold=5
    
    model = svm_train(problem,param)
    
    param.cross_validation = 0
    
    problem = svm_problem(Y.flatten(), X)
    model = svm_train(problem,param)
    
    p_lbl, p_acc, p_prob = svm_predict(Y_test_fin.flatten(),X_test_fin, model)
    
    
    header = ['Id','Class']
    with open('predicted.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(header)
        for i in range(len(p_lbl)):
            str1 = str(i+1)
            if(len(str1)>3):
                str1 = str1[0]+','+str1[1:]             
            data = [str1,p_lbl[i]]
            writer.writerow(data)


if __name__ == '__main__':
    args = setup()
    demo(args)
