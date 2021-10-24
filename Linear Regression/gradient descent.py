import math, os, decimal
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
        squared_sum = decimal.Decimal(0)
        for val in a:
                val = decimal.Decimal(val)
                squared_sum += val ** 2
        return math.sqrt(squared_sum)

# Gradient Descent algorithm
def gradient_descent(data, w, b, lr):
        fsize = len(data[0]) - 1 # feature size
        max_steps = 10000
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
                prev_w = w
                w = subtract(w, ss)
                b -= b_slope * lr
                #print(w, b)

                # Stop iterating when the length of the weight difference
                # (from the prev iteration) is less than the tolerance level
                w_diff = subtract(prev_w, w)
                t = 10e-6
                #print(step)
                if norm(w_diff) < t:
                        return step

                # Stop iterating when the step size (length) is
                # smaller than the learning rate
                #if norm(ss) < lr:
                #        print(step)
                #        break
        #return step

# reads the data from the given csv file
def read_file(CSV_file):
        data = []
        with open(CSV_file, 'r') as f:
                for line in f:
                        values = list(map(float, line.strip().split(',')))
                        data.append(values)
        f.close()
        return data

def main():
        #sample_file = os.path.join("sample.csv")
        #data = [[1, -1, 2, 1],
        #[1, 1, 3, 4],
        #[-1, 1, 0, -1],
        #[1, 2, -4, -2],
        #[3, -1, -1, 0]]
        #data = read_file(sample_file)

        train_file = os.path.join("concrete", "train.csv")
        test_file = os.path.join("concrete", "test.csv")

        data = read_file(train_file)
        fsize = len(data[0]) - 1

        w = [0] * fsize
        b = 0
        lr = 0.0078125 # learning rate

        step = gradient_descent(data, w, b, lr)
        print(step)


if __name__ == "__main__":
    main()