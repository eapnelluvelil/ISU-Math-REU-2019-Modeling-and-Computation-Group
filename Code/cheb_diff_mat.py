import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from spec_diff import spectral_diff

# Function name: cheb_diff_mat
# Inputs: cheb_pts - (N+1) by 1 vector containing the Chebyshev points
#					 on a pre-specified mesh, where N is the number of 
#					 intervals
# Outputs: D     - The Chebyshev differentiation matrix
# def cheb_diff_mat(cheb_pts):
# 	# Dimensions of matrix
# 	N = len(cheb_pts) - 1
# 	D = np.zeros((N + 1, N + 1))

# 	# Populate the (0, 0) and (N, N) entries
# 	D[0, 0] = (2 * pow(N, 2) + 1) / 6
# 	D[N, N] = -D[0, 0]

# 	# Populate the off-diagonal entries
# 	for i in range(N + 1):
# 		for j in range(N + 1):
# 			ci = 2 if (i == 0 or i == N) else 1
# 			cj = 2 if (j == 0 or j == N) else 1

# 			if i != j:
# 				D[i, j] = (ci / cj) * ((-1) ** (i + j) / (cheb_pts[i] - cheb_pts[j]))

# 	# Populate the diagonal entries
# 	for i in range(1, N):
# 		# Create array to sum over (ignoring the ith entry)
# 		# a = np.ma.array(D[i], mask=False)
# 		# a.mask[i] = True
# 		D[i, i] = (-1) * D[i].sum()

# 	return D

# Function name: cheb_diff_mat
# Inputs:
#	N - Number of intervals 
#	c - Left endpoint of interval
#	d - Right endpoint of interval
# Outputs:
#	D - The Chebyshev differentiation matrix
#		corresponding to (N + 1) Chebyshev
#		points on the interval [c, d]
#	cheb_pts -	Vector of (N + 1) Chebyshev points
#				on the interval [c, d]
def cheb_diff_mat(N, c=-1, d=1):
	cheb_pts = np.cos((np.pi / N) * np.array(range(N + 1)))
	# Rescale the points to lie on [c, d]
	cheb_pts = c + ((d - c) / 2) * (cheb_pts + 1)

	D = np.zeros((N + 1, N + 1))

	# Populate the off-diagonal entries
	for i in range(N + 1):
		for j in range(N + 1):
			ci = 2 if (i == 0 or i == N) else 1
			cj = 2 if (j == 0 or j == N) else 1

			if i != j:
				D[i, j] = (-1) ** (i + j) * (ci / cj) * (1 / (cheb_pts[i] - cheb_pts[j]))

	# Populate the diagonal entries
	for i in range(N + 1):
		D[i, i] = 0
		D[i, i] = (-1) * D[i].sum()

	return cheb_pts, D

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Time-stepping using Chebyshev differentiation matrices
# Number of intervals
N = 5

# Desired interval
c = -5
d = 5

# Create Chebyshev points and scale them to lie in [c, d]
# cheb_pts = np.cos((np.pi / N) * np.array(range(N + 1)))
# cheb_pts = c + ((d - c) / 2) * (cheb_pts + 1)

# Create differentiation matrix
# D = cheb_diff_mat(cheb_pts)
cheb_pts, D = cheb_diff_mat(N, c, d)
# s, D2 = spectral_diff(N + 1, c, d)
# print(np.allclose(D, D2))
# print(D)

# Create vector of initial conditions
f = lambda x: 1 / (1 + 25 * (x ** 2))
# f = lambda x: np.exp(-(x-2)**2)

# Initial conditions
f0 = f(cheb_pts)

# Plot the initial conditions and derivative
plt.plot(cheb_pts, f0, 'r')
plt.plot(cheb_pts, D @ f0, 'b')
plt.show()

# Time-stepping
# Time step size
dt = 0.1

# Current time
t = 0

# Final time
tf = 10

op = np.linalg.inv(np.eye(N + 1) + dt*D)

# while t < tf:
# 	# Clear the current figure
# 	plt.clf()

# 	# Plot the current solution
# 	plt.plot(cheb_pts, f0)
# 	plt.ylim((0, 1))

# 	# Update time
# 	t = t + dt
# 	# Find solution at next time
# 	f0 = op @ f0
# 	plt.pause(0.1)

# plt.show()
