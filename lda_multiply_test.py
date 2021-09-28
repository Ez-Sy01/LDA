import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

cls1_data = np.array([[2.93,6.634],[2.53,7.79],[3.57,5.65],[3.16,5.47]])
cls2_data = np.array([[2.58,4.44],[2.16,6.22],[3.27,3.52]])

E_cls1 = np.mean(cls1_data,axis = 0)
E_cls2 = np.mean(cls2_data,axis = 0)
E_all = (np.sum(cls1_data,axis = 0) + np.sum(cls2_data, axis = 0)) / (np.size(cls1_data, 0)  +np.size(cls2_data,0))

plt.figure('data - variance - draw')
plt.axis([0,10,0,10])
plt.plot(cls1_data[:,0],cls1_data[:,1], '*r')
plt.plot(cls2_data[:,0],cls2_data[:,1], '*b')

# between_class scatter matrix
x1 = E_cls1 - E_all
x2 = E_cls2 - E_all
x_all_size = np.size(cls1_data,0) + np.size(cls2_data,0)
Sb = (np.size(cls1_data,0) / x_all_size) * np.transpose(x1) * x1 + (np.size(cls2_data,0) / x_all_size) * np.transpose(x2) * x2

# within_class scatter matrix
y1 = 0
for i in range(np.size(cls1_data,0)):
    y1 = y1 + np.transpose([cls1_data[i,:] - E_cls1]) * (cls1_data[i,:] - E_cls1)

y2 = 0
for i in range(np.size(cls2_data,0)):
    y2 = y2 + np.transpose([cls2_data[i,:]] - E_cls2) * (cls2_data[i,:] - E_cls2)

y_all_size = np.size(cls1_data,0) + np.size(cls2_data,0)
Sw = (np.size(cls1_data,0) / y_all_size) * y1 + (np.size(cls2_data,0) / y_all_size) * y2

# eigen values & eigne vectors
eig_val,eig_vec = linalg.eig(linalg.inv(Sw) * Sb)
value_index = np.argmax(eig_val)
vector = eig_vec[:,value_index]

# this vector space projection
new_cls1_data = cls1_data * vector
new_cls2_data = cls2_data * vector

# draw vector on graph
a = vector[1] / vector[0]
plt.plot([0,10],[0,10 * a], '-g')

for i in range(np.size(cls1_data,0)):
    new_x = (cls1_data[i,0] + a * cls1_data[i,1]) / (a ** 2 + 1)
    new_y = a * new_x
    plt.plot(new_x,new_y,'*r')
    plt.plot([cls1_data[i,0],new_x], [cls1_data[i,1],new_y],'--k')

for i in range(np.size(cls2_data, 0)):
    new_x = (cls2_data[i,0] + a * cls2_data[i,1]) / (a ** 2 + 1)
    new_y = a * new_x
    plt.plot(new_x,new_y, '*b')
    plt.plot([cls2_data[i,0],new_x], [cls2_data[i,1],new_y], '--k')

# testing
test_data = np.array([4.81,3.46])
plt.plot(test_data[0],test_data[1],' *g')
result = test_data * vector # this vector projection test_data

projected_test_data_x = (test_data[0] + a * test_data[1]) / (a ** 2 + 1)
projected_test_data_y = a * projected_test_data_x
plt.plot(projected_test_data_x, projected_test_data_y,'*g')
plt.plot([test_data[0], projected_test_data_x],[test_data[1], projected_test_data_y], '--k')

# classification -> average in class
temp1 = new_cls1_data - result
temp2 = new_cls2_data - result

distance_result1 = list()
for i in range(np.size(temp1,0)):
    for j in range(i + 1,np.size(temp1,0)):
        distance_result1.append(np.linalg.norm(temp1[i] - temp1[j]))
distance_result1 = np.mean(distance_result1)
print(distance_result1)

distance_result2 = list()
for i in range(np.size(temp2,0)):
    for j in range(i + 1,np.size(temp2,0)):
        distance_result2.append(np.linalg.norm(temp2[i] - temp2[j]))
distance_result2 = np.mean(distance_result2)
print(distance_result2)

if distance_result1 < distance_result2:
    print('class -> red')
else:
    print('class -> blue')
plt.gca().set_title('experiment -> 2-D LDA Algorithm')
plt.show()
