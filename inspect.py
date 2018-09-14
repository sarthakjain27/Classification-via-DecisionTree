__author__ = 'user'
import math
import sys
import numpy as np
import csv

with open(sys.argv[1], 'rb') as f:
    reader = csv.reader(f)
    data = np.asarray(list(reader))

fw=open(sys.argv[2],"w")

a=data[:,np.size(data,1)-1]
dist_label=np.unique(a)
dist_label=np.delete(dist_label,0)

if (len(dist_label)==0):
    error_rate=0.0
    fw.write("error: {}\n".format(error_rate))
    fw.close()
    sys.exit(0)
elif (len(dist_label)==1):
    c1=(data[:,np.size(data,1)-1]==dist_label[0]).sum()
    c2=0;
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

print("entropy: {}\n".format(e1+e2))
fw.write("entropy: {}\n".format(e1+e2))

if c1>c2:
    error_rate=(float)(c2)/(c1+c2)
else:
    error_rate=(float)(c1)/(c1+c2)
fw.write("error: {}\n".format(error_rate))
print("error: {}\n".format(error_rate))
fw.close()




