import numpy as np
a = np.array([[1],[5],[-3],[2]])
b = np.array([[8],[2],[4],[7]])
print('a=',a,'b=',b,'a-b=',a,sep='\n')
print('aT*b=\n',np.dot(np.transpose(a),b))
M = np.array([[1,2,4],[-2,5,-1]])
v = np.array([[3],[0],[-5]])
w = np.array([[2],[-3]])
print('M=',M,'v=',v,'w=',w,'M*v=',np.dot(M,v),'wT*M=',np.dot(np.transpose(w),M),sep='\n')
print('v*wT=',np.dot(v,np.transpose(w)),'w*vT=',np.dot(w,np.transpose(v)),sep='\n')

def normalize(v):
    norm = np.linalg.norm(v)
    return v/norm

arrBc = [[170,65],[174,80],[179,90],[166,72]]

def calc(arr):
    return np.dot(np.transpose(arr),np.ones((4,1)))/4

print(calc(arrBc))

def signed_dist(x,th,th0):
    return (np.dot(np.transpose(th),x)+th0)/np.linalg.norm(th)

def positive(x,th,th0):
    if signed_dist(x,th,th0)>0: return 1
    elif signed_dist(x,th,th0)==0: return 0
    else: return -1

def score(data, labels, th, th0):
    #if (positive(data(i), th, th0)==1 and labels[i]>0)
    data_dist = (np.dot(np.transpose(data), th)+th0)*1/np.linalg.norm(th)
    return len(np.where(data_dist*np.transpose(labels) > 0)[0])

data = np.array([[-3,1,3,-1,-1,3,-1,4,-4,2,-0.5,2,2],[3,4,2,-2,-4,-3,4,1,-1,-1,1,1,-3]])
labels = np.array([[1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1]])
th1 = np.array([[1],[2]])
th01 = 0
th2 = np.array([[-2],[2]])
th02 = -4
print(score(data, labels, th1, th01))
print(score(data, labels, th2, th02))

def best_separator(data, labels, ths, th0s):
    return 0