import numpy as np
import time

number = 30000
p1, p2, p3 = 0.2, 0.5, 0.4
number1, number2, number3 = int(number * 0.2), int(number * 0.5), number - int(number * 0.2) - int(number * 0.5)

data1 = np.random.binomial(1, p1, size=number1)
data2 = np.random.binomial(1, p2, size=number2)
data3 = np.random.binomial(1, p3, size=number3)
data0 = []
data0.extend(data1)
data0.extend(data2)
data0.extend(data3)
np.random.shuffle(data0)
data_arr = np.array(data0)


def E_step(dataset, alpha1, alpha2, alpha3, theta1, theta2, theta3):
    gamma1 = alpha1 ** dataset * (1 - alpha1) ** (1 - dataset) * theta1
    gamma2 = alpha2 ** dataset * (1 - alpha2) ** (1 - dataset) * theta2
    gamma3 = alpha3 ** dataset * (1 - alpha3) ** (1 - dataset) * theta3
    sum0 = gamma1 + gamma2 + gamma3
    gamma1 = gamma1 / sum0
    gamma2 = gamma2 / sum0
    gamma3 = gamma3 / sum0
    return gamma1, gamma2, gamma3


def M_step(dataset, gamma1, gamma2, gamma3):
    alpha1_new = np.sum(gamma1) / number
    alpha2_new = np.sum(gamma2) / number
    alpha3_new = np.sum(gamma3) / number
    theta1_new = np.dot(gamma1, dataset) / np.sum(gamma1)
    theta2_new = np.dot(gamma2, dataset) / np.sum(gamma2)
    theta3_new = np.dot(gamma3, dataset) / np.sum(gamma3)
    return alpha1_new, alpha2_new, alpha3_new, theta1_new, theta2_new, theta3_new


if __name__ == '__main__':
    start = time.time()
    step = 0
    iter_num = 500
    alpha01, alpha02, alpha03 = 0.1, 0.4, 0.5
    theta01, theta02, theta03 = 0.6, 0.7, 0.4
    while step < iter_num:
        step = step + 1
        gamma01, gamma02, gamma03 = E_step(data_arr, alpha01, alpha02, alpha03, theta01, theta02, theta03)
        alpha01, alpha02, alpha03, theta01, theta02, theta03 = M_step(data_arr, gamma01, gamma02, gamma03)

    print('---------------------------')
    print('the Parameters set is:')
    print('alpha1:%.1f, alpha2:%.1f, alpha3:%.1f, theta1:%.1f, theta2:%.1f, theta3:%.1f' % (
        number1/number, number2/number, number3/number,p1, p2, p3, ))

    print('----------------------------')
    print('the Parameters predict is:')
    print('alpha1:%.2f, alpha2:%.2f, alpha3:%.2f, theta1:%.2f, theta2:%.2f, theta3:%.2f' % (
        alpha01, alpha02, alpha03, theta01, theta02, theta03))

    print('----------------------------')
    print('time span:', time.time() - start)
