# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 12:20:30 2016

@author: Hongwei
"""

import matplotlib.pyplot as plt

t = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

a6g5 = [34, 42, 71, 82, 83, 95, 94, 96, 95, 95]

a8g5 = [29, 52, 66, 66, 80, 85, 90, 92, 94, 98]

a4g5 = [32, 53, 59, 73, 89, 90, 90, 95, 92, 97]

a6g2 = [21, 50, 79, 80, 93, 99, 98, 98, 97, 99]

a6g8 = [31, 54, 62, 69, 78, 78, 89, 88, 96, 92]





plt.plot(t, a6g5, 'red', t, a8g5, 'blue', t, a4g5, 'black', t, a6g2, 'green', t, a6g8, 'pink')

plt.ylabel('Success Trials per 100')
plt.xlabel('Numer of Trials')
plt.show()



a6g5 = [666, 463, 304, 204, 150, 111, 78, 64, 62, 51]

a8g5 = [628, 436, 297, 253, 203, 145, 86, 79, 53, 45]

a4g5 = [617, 383, 298, 216, 108, 108, 84, 62, 59, 48]

a6g2 = [636, 447, 277, 212, 122, 90, 89, 59, 60, 35]

a6g8 = [624, 462, 337, 266, 175, 143, 121, 136, 68, 50]

plt.plot(t, a6g5, 'red', t, a8g5, 'blue', t, a4g5, 'black', t, a6g2, 'green', t, a6g8, 'pink')

plt.ylabel('Number of Invalid Move per 100 Trials')
plt.xlabel('Numer of Trials')
plt.show()


a6g5 = [1.499381, 1.399729, 0.983359, 0.786376, 0.739905, 0.546380, 0.507797, 0.508761, 0.513697, 0.485262]

a8g5 = [1.590487, 1.250403, 1.041635, 1.003517, 0.810340, 0.711136, 0.615363, 0.605920, 0.517956, 0.516350]

a4g5 = [1.554174, 1.212364, 1.139493, 0.919689, 0.698329, 0.644856, 0.617790, 0.574399, 0.596553, 0.514878]

a6g2 = [1.684033, 1.268496, 0.830752, 0.814963, 0.613656, 0.484067, 0.482990, 0.457665, 0.492662, 0.414971]

a6g8 = [1.53421, 1.249454, 1.092473, 0.982574, 0.847400, 0.885221, 0.669927, 0.697964, 0.578363, 0.608837]

plt.plot(t, a6g5, 'red', t, a8g5, 'blue', t, a4g5, 'black', t, a6g2, 'green', t, a6g8, 'pink')

plt.ylabel('Average Time Usage per 100 Trials')
plt.xlabel('Numer of Trials')
plt.show()