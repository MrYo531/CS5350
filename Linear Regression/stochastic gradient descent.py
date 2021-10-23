import math

# Stochastic Gradient Descent
data = [[1, -1, 2],
        [1, 1, 3],
        [-1, 1, 0],
        [1, 2, -4],
        [3, -1, -1]]
        
y = [1, 4, -1, -2, 0]

w = [0, 0, 0]
b = 0
lr = 0.1

ss = [0, 0, 0]
max_steps = 5
for step in range(0, max_steps):
        gradient = [0, 0, 0]
        b_slope = 0
        for j in range(0, len(data[0])):
                #for i, x in enumerate(data):
                i = step
                x = data[i]
                gradient[j] += -(y[i] - (w[0]*x[0] + w[1]*x[1] + w[2]*x[2]) - b) * x[j]
        for i, x in enumerate(data):
                b_slope += -(y[i] - (w[0]*x[0] + w[1]*x[1] + w[2]*x[2]) - b)
        print(step)
        print(gradient)
        #print(b_slope)

        # new weight & bias
        ss = [lr * g for g in gradient]
        w = [w[0] - ss[0], w[1] - ss[1], w[2] - ss[2]]
        b -= b_slope * lr
        print(w, b)
        
        #print(ss)
        if abs(ss[0]) < lr and abs(ss[1]) < lr and abs(ss[2]) < lr:
                print(step)
                break
