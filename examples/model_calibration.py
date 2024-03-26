
import numpy as np
import matplotlib.pyplot as plt
#using linear regression for Greenshield
from sklearn.linear_model import LinearRegression
#see https://realpython.com/linear-regression-in-python/

from sklearn.preprocessing import PolynomialFeatures

#using logarithmic regression for greenberg
from scipy.optimize import curve_fit

#data from field measurements
u = np.array([71, 62, 41, 13,  22,  31, 49])
k = np.array([13, 22, 45, 96, 75, 58, 33])

#flow
q = u * k

#compare value
k_pred = np.arange(0, 120, 1).reshape((-1, 1))
u_pred = np.arange(0, 120, 1).reshape((-1, 1))
#x_pred = np.linspace(0.0, 120.0, num=120).reshape((-1, 1))

#load linear regression & fit
greenshield = LinearRegression().fit(k.reshape((-1, 1)), u)
#r_sq = model.score(k, u)
#get parameters
print(greenshield.coef_)

#predict greenberg
# define a function for fitting
def greenbergfit(a,x,b):
    return a * np.log10(x) + b

init_vals = [50, 0]
# fit your data and getting fit parameters
popt, pcov = curve_fit(greenbergfit, k, u, p0=init_vals, bounds=([0, 0], [100, 100]))
print(popt)

#predict values
u_pred_greenshield = greenshield.predict(k_pred)
u_pred_greenberg = greenbergfit(k_pred, *popt)

#plots
fig, sub_plots = plt.subplots(1, 3)

cur_sub_plot_id = 0
sub_plots[cur_sub_plot_id].plot(k_pred, u_pred_greenshield, '-',label='predicted Greenshield')
sub_plots[cur_sub_plot_id].plot(k_pred, u_pred_greenberg, '--',label='predicted Greenberg')
sub_plots[cur_sub_plot_id].plot(k, u, 'x',label='orig')
sub_plots[cur_sub_plot_id].legend()
sub_plots[cur_sub_plot_id].set_title('speed over density')
sub_plots[cur_sub_plot_id].set(xlabel='density [veh/km]')
sub_plots[cur_sub_plot_id].set(ylabel='speed [km/h]')

cur_sub_plot_id = 1
sub_plots[cur_sub_plot_id].plot(q, u, 'x',label='orig')
sub_plots[cur_sub_plot_id].set_title('speed over flow')
sub_plots[cur_sub_plot_id].set(xlabel='density [veh/km]')
sub_plots[cur_sub_plot_id].set(ylabel='speed [km/h]')

cur_sub_plot_id = 2
sub_plots[cur_sub_plot_id].plot(q, k, 'x',label='orig')
sub_plots[cur_sub_plot_id].set_title('flow over density')
sub_plots[cur_sub_plot_id].set(xlabel='density [veh/km]')
sub_plots[cur_sub_plot_id].set(ylabel='speed [km/h]')

plt.show()