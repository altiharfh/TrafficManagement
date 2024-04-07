
#required
#python -m pip install numpy scikit-learn statsmodels

#implementation based on
#https://realpython.com/linear-regression-in-python/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#see example data from tex file
cars = np.array([1, 1, 1, 1, 2,2,2,2,3,3,3,3,4,4,4,4])
trips = np.array([1,1,3,3,2,3,4,5,5,4,5,7,7,5,8,8])

#load linear regression & fit
linregress_model = LinearRegression().fit(cars.reshape((-1, 1)), trips)

#R^2 performance
r_sq = linregress_model.score(cars.reshape((-1, 1)), trips)

car_scale = np.arange(0, 6, 1).reshape((-1, 1))
trips_pred = linregress_model.predict(car_scale)

#plot
plt.plot(cars,trips,'*',label='orig data')
plt.plot(car_scale,trips_pred,'-',label='predicted data ' + '{:.2f}'.format(r_sq))
plt.legend()
plt.title('linear regression trips=' + '{:.2f}'.format(linregress_model.intercept_) + 
          ' + ' + '{:.2f}'.format(linregress_model.coef_[0]) + ' cars')
plt.xlabel('number of cars')
plt.ylabel('number of trips')
plt.show()