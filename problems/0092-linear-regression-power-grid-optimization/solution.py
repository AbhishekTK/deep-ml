import math
# import numpy as np
PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	# pass
	detrended_data = [ consumption_data[i] - 10*math.sin(2*PI*(i+1)/10) for i in range(10)] 

	n =  len(consumption_data)
	# m = 1
	# b = 0
	X = list(range(1,n+1))
	y = detrended_data
	ys = sum(y)
	m,b =0,0
	xs = sum(X)
	xss = sum(d**2 for d in X)
	xys = sum(x*y for x,y in zip(X,detrended_data))
	# for _ in range(n):
	m = (n*xys- xs*ys)/(n*xss-xs**2)
	b = (ys - m*xs)/n
	
	f = m*15 +b
	fluc =10*math.sin(2*PI*15/10)
	f +=fluc
	return math.ceil((f*1.05))
