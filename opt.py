import cvxpy as cp
import numpy as np

def findPath(x):
    airport = 0
    airports = ['LAX', 'ATL', 'SFO', 'JFK', 'DEN', 'DFW', 'ORD', 'DTW', 'MCO', 'LAS', 'SEA', 'PHX', 'MSP', 'BOS']
    str = airports[airport]
    for i in range(len(x[airport])):
        if x[airport, i] == 1:
            airport = i
            str = str + " -> " + airports[airport]
            break
    while airport != 0:
        for i in range(len(x[airport])):
            if x[airport, i] == 1:
                airport = i
                str = str + " -> " + airports[airport]
                break
    return(str)

distances = np.array([
   [0, 1946, 337, 2475, 862, 1231, 1745, 1979, 2214, 236, 954, 370, 1533, 2611], 
   [1946, 0, 2139, 760, 1199, 732, 606, 594, 403, 1747, 2182, 1587, 906, 946],    
   [337, 2139, 0, 2586, 967, 1464, 1846, 2079, 2444, 414, 679, 651, 1589, 2704], 
   [2475, 760, 2586, 0, 1620, 1391, 740, 502, 944, 2247, 2422, 2154, 1028, 187], 
   [862, 1199, 967, 1620, 0, 641, 888, 1120, 1547, 628, 1021, 602, 680, 1778],  
   [1231, 732, 1464, 1391, 641, 0, 802, 987, 980, 1221, 1667, 868, 853, 1564],  
   [1745, 606, 1846, 740, 888, 802, 0, 237, 989, 1520, 1733, 1413, 333, 847],     
   [1979, 594, 2079, 502, 1120, 987, 237, 0, 957, 1778, 1931, 1670, 529, 632],   
   [2214, 403, 2444, 944, 1547, 980, 989, 957, 0, 2043, 2553, 1842, 1310, 1183], 
   [236, 1747, 414, 2247, 628, 1221, 1520, 1778, 2043, 0, 869, 255, 1284, 2368], 
   [954, 2182, 679, 2422, 1021, 1667, 1733, 1931, 2553, 869, 0, 1107, 1398, 2484],
   [370, 1587, 651, 2154, 602, 868, 1413, 1670, 1842, 255, 1107, 0, 1277, 2298],  
   [1533, 906, 1589, 1028, 680, 853, 333, 529, 1310, 1284, 1398, 1277, 0, 1104],  
   [2611, 946, 2704, 187, 1778, 1564, 847, 632, 1183, 2368, 2484, 2298, 1104, 0]
])

n = 14

w = cp.Variable((n, n), boolean=True)
x = cp.Variable((n, n), boolean=True)
y = cp.Variable((n, n), boolean=True)
z = cp.Variable((n, n), boolean=True)
t = cp.Variable(n, nonneg = True)

obj_func = (
    cp.sum(cp.multiply(distances, w)) + 
    cp.sum(cp.multiply(distances, x)) +
    cp.sum(cp.multiply(distances, y)) +
    cp.sum(cp.multiply(distances, z))
    )

constraints = []
for i in range(1, n):
    constraints.append(cp.sum(w[:, i]) + cp.sum(x[:, i]) + cp.sum(y[:, i]) + cp.sum(z[:, i]) == 1)
    constraints.append(cp.sum(w[i, :]) + cp.sum(x[i, :]) + cp.sum(y[i, :]) + cp.sum(z[i, :]) == 1)

for i in range(n):
    constraints.append(w[i, i] == 0)
    constraints.append(x[i, i] == 0)
    constraints.append(y[i, i] == 0)
    constraints.append(z[i, i] == 0)

for i in range(n):
    constraints.append(cp.sum(w[:, i]) == cp.sum(w[i, :]))
    constraints.append(cp.sum(x[:, i]) == cp.sum(x[i, :]))
    constraints.append(cp.sum(y[:, i]) == cp.sum(y[i, :]))
    constraints.append(cp.sum(z[:, i]) == cp.sum(z[i, :]))

constraints.append(cp.sum(w)<=5)
constraints.append(cp.sum(w)>=4)
constraints.append(cp.sum(x)<=5)
constraints.append(cp.sum(x)>=4)
constraints.append(cp.sum(y)<=5)
constraints.append(cp.sum(y)>=4)
constraints.append(cp.sum(z)<=5)
constraints.append(cp.sum(z)>=4)

constraints.append(cp.sum(w[0,:]) == 1)
constraints.append(cp.sum(x[0,:]) == 1)
constraints.append(cp.sum(y[0,:]) == 1)
constraints.append(cp.sum(z[0,:]) == 1)

for i in range(1, n):
    for j in range(1, n):
        if i != j:
            constraints.append(t[i] - t[j] + (n - 1) * w[i, j] <= n - 2)
            constraints.append(t[i] - t[j] + (n - 1) * x[i, j] <= n - 2)
            constraints.append(t[i] - t[j] + (n - 1) * y[i, j] <= n - 2)
            constraints.append(t[i] - t[j] + (n - 1) * z[i, j] <= n - 2)


problem = cp.Problem(cp.Minimize(obj_func), constraints)
problem.solve(solver=cp.GUROBI)
print(f"Solver status: {problem.status}")
print("obj_func =")
print(obj_func.value)
print("w =")
print(w.value)
print("x =")
print(x.value)
print("y =")
print(y.value)
print("z =")
print(z.value)
print("path of plane 1:")
print(findPath(w.value))
print("path of plane 2:")
print(findPath(x.value))
print("path of plane 3:")
print(findPath(y.value))
print("path of plane 4:")
print(findPath(z.value))
