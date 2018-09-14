__author__ = 'user'
import math
import sys
import numpy as np
import csv

class Node:

    def __init__(self, data):
        self.left = None
        self.right = None
        self.attribute_name = data
        self.leafNode=False
        self.leftVal=""
        self.rightVal=""
        self.predictVal=""

def entropy(data):
    a=data[:,np.size(data,1)-1]
    dist_label=np.unique(a)
    if (len(dist_label)==0):
        return 0
    elif (len(dist_label)==1):
        c1=(data[:,np.size(data,1)-1]==dist_label[0]).sum()
        c2=0
    else:
        c1=(data[:,np.size(data,1)-1]==dist_label[0]).sum()
        c2=(data[:,np.size(data,1)-1]==dist_label[1]).sum()

    probab1=(float)(c1)/(c1+c2)
    if(probab1==0):
        e1=0.0
    else:
        e1=-1*probab1*(math.log(probab1,2))

    probab2=(float)(c2)/(c1+c2)
    if(probab2==0):
        e2=0.0
    else:
        e2=-1*probab2*(math.log(probab2,2))

    return (e1+e2)

def entropy_attr(data,attr,colNumb):
    attr_uniq_val=np.unique(data[:,colNumb])
    if (len(attr_uniq_val)==0):
        return 0
    elif(len(attr_uniq_val)==1):
        c1=(data[:,colNumb]==attr_uniq_val[0]).sum()
        c2=0
        df_uniq0=data[data[:,colNumb] == attr_uniq_val[0]]
        e_attr_0=entropy(df_uniq0)*((float)(c1)/(c1+c2))
        e_attr_1=0;
    else:
        c1=(data[:,colNumb]==attr_uniq_val[0]).sum()
        c2=(data[:,colNumb]==attr_uniq_val[1]).sum()
        df_uniq0=data[data[:,colNumb] == attr_uniq_val[0]]
        df_uniq1=data[data[:,colNumb] == attr_uniq_val[1]]
        e_attr_0=entropy(df_uniq0)*((float)(c1)/(c1+c2))
        e_attr_1=entropy(df_uniq1)*((float)(c2)/(c1+c2))

    return (e_attr_0+e_attr_1)

def info_gain(data,attr,colNumb):
    h_wo_split=entropy(data)
    h_w_split=entropy_attr(data,attr,colNumb)

    return (h_wo_split-h_w_split)

def find_majority(data):
    a=data[:,np.size(data,1)-1]
    dist_label=np.unique(a)
    num_label=len(dist_label)
    if(num_label==1):
        return dist_label[0]
    else:
        c1=(data[:,np.size(data,1)-1]==dist_label[0]).sum()
        c2=(data[:,np.size(data,1)-1]==dist_label[1]).sum()
        if(c1>=c2):
            return dist_label[0]
        else:
            return dist_label[1]

def printLevelInfo(data,target_label,attrName,attrVal,level):
    a=data[:,np.size(data,1)-1]
    dist_label=np.unique(a)
    num_label=len(dist_label)
    if(num_label==1):
        c1=(data[:,np.size(data,1)-1]==dist_label[0]).sum()
        print("Level_{} {} = {}: [{} {}/0 {}]".format(level,attrName,attrVal,c1,dist_label[0],target_label[target_label!=dist_label[0]][0]))
    else:
        c1=(data[:,np.size(data,1)-1]==dist_label[0]).sum()
        c2=(data[:,np.size(data,1)-1]==dist_label[1]).sum()
        print("Level_{} {} = {}: [{} {}/{} {}]".format(level,attrName,attrVal,c1,dist_label[0],c2,dist_label[1]))

def getLabel(root,attr_arr,sample):
    if root.leafNode==True:
        return root.predictVal
    if sample[np.where(attr_arr==root.attribute_name)] == root.leftVal:
        return getLabel(root.left,attr_arr,sample)
    else:
        return getLabel(root.right,attr_arr,sample)

def test(data,attr_arr,root,filename):
    fw=open(filename,"w")
    error_count=0

    for row in data:
        predictVal=getLabel(root,attr_arr,row)
        fw.write(predictVal+'\n')

        #print("{} {}".format(row,predictVal))
        if predictVal!=row[len(attr_arr)-1]:
            error_count=error_count+1

    fw.close()
    error_value=(float)(error_count)/data.shape[0]
    #print("error({}): {}".format(type_dataset,error_value))
    return error_value

def train(data,attr_arr,target_label,depth,level):
    entr_data=entropy(data)

    if entr_data==0:
        a=data[:,np.size(data,1)-1]
        dist_label=np.unique(a)
        root=Node(dist_label[0])
        root.leafNode=True
        root.predictVal=dist_label[0]
        return root

    if depth==0:
        maj_lab=find_majority(data)
        root=Node(maj_lab)
        root.leafNode=True
        root.predictVal=maj_lab
        return root

    all_col=attr_arr
    i = 0
    maxIG=0
    index=0
    attrName=""
    while i < (len(all_col)-1):
        IG=info_gain(data,all_col[i],i)
        if(maxIG<IG):
            attrName=all_col[i]
            index=i
            maxIG=IG
        i+=1

    if maxIG==0:
        maj_lab=find_majority(data)
        root=Node(maj_lab)
        root.leafNode=True
        root.predictVal=maj_lab
        return root

    root=Node(attrName)
    dist_label=np.unique(data[:,index])
    root.leftVal=dist_label[0]
    root.rightVal=dist_label[1]

    left_data=data[data[:,index]==dist_label[0]]
    printLevelInfo(left_data,target_label,attrName,dist_label[0],level)
    left_next_level_data=np.delete(left_data,index,axis=1)
    next_level_attr=np.delete(attr_arr,index)
    root.left=train(left_next_level_data,next_level_attr,target_label,depth-1,level+1)

    right_data=data[data[:,index]==dist_label[1]]
    printLevelInfo(right_data,target_label,attrName,dist_label[1],level)
    right_next_level_data=np.delete(right_data,index,axis=1)
    root.right=train(right_next_level_data,next_level_attr,target_label,depth-1,level+1)

    return root


def printStartInfo(data,target_label):
    num_label=len(target_label)
    if(num_label==1):
        c1=(data[:,np.size(data,1)-1] == target_label[0]).sum()
        print("[{} {}]".format(c1,target_label[0]))
    else:
        c1=(data[:,np.size(train_data,1)-1] == target_label[0]).sum()
        c2=(data[:,np.size(train_data,1)-1] == target_label[1]).sum()
        print("[{} {}/{} {}]".format(c1,target_label[0],c2,target_label[1]))
    return


if __name__== "__main__":
    with open(sys.argv[1], 'rb') as f:
        reader = csv.reader(f)
        train_data = np.asarray(list(reader))

    train_attr_arr=train_data[0,:]
    train_data=np.delete(train_data,0,axis=0)
    a=train_data[:,np.size(train_data,1)-1]
    target_label=np.unique(a)
    printStartInfo(train_data,target_label)
    root = train(train_data,train_attr_arr,target_label,int(sys.argv[3]),level=1)

    train_error = test(train_data,train_attr_arr,root,sys.argv[4])

    with open(sys.argv[2], 'rb') as f:
        reader = csv.reader(f)
        test_data = np.asarray(list(reader))

    test_data=np.delete(test_data,0,axis=0)
    test_error = test(test_data,train_attr_arr,root,sys.argv[5])

    fwm=open(sys.argv[6],"w")
    fwm.write("error(train): {}\n".format((train_error)))
    fwm.write("error(test): {}\n".format((test_error)))
    fwm.close()
