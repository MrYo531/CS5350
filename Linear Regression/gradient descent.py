import math, os
# Gradient Descent 

# dot product of two 1D vectors:
# multiplies the values of the two vectors together and returns the sum
def dot(a, b):
        result = 0
        for i in range(len(a)):
                result += a[i] * b[i]
        return result

# subtracts one 1D vector from the other
def subtract(a, b):
        result = []
        for i in range(len(a)):
                result.append(a[i] - b[i])
        return result

# calculates the length of a vector
def norm(a):
        squared_sum = 0
        for val in a:
                squared_sum += val ** 2
        return math.sqrt(squared_sum)

# Gradient Descent algorithm
def gradient_descent(data, w, b, lr):
        fsize = len(data[0]) - 1 # feature size
        max_steps = 100
        for step in range(max_steps):
                # Calc gradient
                gradient = [0] * fsize
                for j in range(fsize):
                        for x in data:
                                y = x[-1]
                                wx = dot(w, x)
                                gradient[j] += -(y - wx - b) * x[j]
                
                # Calc b slope
                b_slope = 0
                for x in data:
                        y = x[-1]
                        wx = dot(w, x)
                        b_slope += -(y - wx - b)

                # new weight & bias
                ss = [lr * g for g in gradient] # step size
                w = subtract(w, ss)
                b -= b_slope * lr
                print(w, b)

                # Stop converging when the step size (length) is
                # smaller than the learning rate
                if norm(ss) < lr:
                        print(step)
                        break

# reads the data from the given csv file
def read_file(CSV_file):
        data = []
        with open(CSV_file, 'r') as f:
                for line in f:
                        values = list(map(int, line.strip().split(',')))
                        data.append(values)
        f.close()
        return data

def main():
        sample_file = os.path.join("sample.csv")
        #data = [[1, -1, 2, 1],
        #[1, 1, 3, 4],
        #[-1, 1, 0, -1],
        #[1, 2, -4, -2],
        #[3, -1, -1, 0]]
        #data = read_file(sample_file)

        train_file = os.path.join("concrete", "train.csv")
        test_file = os.path.join("concrete", "test.csv")

        data = read_file(sample_file)

        w = [-1, 1, -1]
        b = -1
        lr = 0.01 # learning rate

        gradient_descent(data, w, b, lr)


if __name__ == "__main__":
    main()