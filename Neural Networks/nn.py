import math

def nn():
    NotImplemented

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def dot(x, y):
    return sum([x * y for x, y in zip(x, y)])

def main():

    # what each index represents
    node_layer = 0
    node_to = 1
    node_from = 2
    weight_value = 3

    # define all the weights
    # three layers
    num_of_layers = 3
    weights = []
    # layer 1
    weights.append( [ 1, 0, 1, -1] )
    weights.append( [ 1, 0, 2, 1 ] )
    weights.append( [ 1, 1, 1, -2 ] )
    weights.append( [ 1, 1, 2, 2 ] )
    weights.append( [ 1, 2, 1, -3 ] )
    weights.append( [ 1, 2, 2, 3 ] )
    # layer 2
    weights.append( [ 2, 0, 1, -1] )
    weights.append( [ 2, 0, 2, 1 ] )
    weights.append( [ 2, 1, 1, -2 ] )
    weights.append( [ 2, 1, 2, 2 ] )
    weights.append( [ 2, 2, 1, -3 ] )
    weights.append( [ 2, 2, 2, 3 ] )
    # layer 3 - output
    weights.append( [ 3, 0, 1, -1 ] )
    weights.append( [ 3, 1, 1, 2 ] )
    weights.append( [ 3, 2, 1, -1.5 ] )

    #print(weights)


    # forward pass (calculate all the z values)
    nn_width = 2
    z_values = []
    x = [1, 1, 1]
    input_width = len(x)
    z_values.append(x)
    for layer in range(1, num_of_layers):
        z_values.append([1]) # the first element is the bias (1)
        for node_index in range(nn_width):
            prev_layer = layer - 1
            input = z_values[prev_layer]
            node_weight = []
            for i in range(input_width):
                index = (prev_layer * nn_width * input_width) + (nn_width * i) + node_index
                node_weight.append(weights[index][weight_value])
           
            z = sigmoid(dot(input, node_weight))
            z_values[layer].append(z)
        
    print(z_values)

    # y value/prediction (last calc)
    prev_layer = num_of_layers - 1
    input = z_values[prev_layer]
    node_weight = []
    for i in range(input_width):
        index = (prev_layer * nn_width * input_width) + i
        node_weight.append(weights[index][weight_value])
    
    y = dot(input, node_weight)
    print(y)


    # backpropagation
    label = 1
    partial_l_y = y - label
    top_layer = num_of_layers - 1 # (2)
    partial_derivatives = []
    for layer in reversed(range(1, num_of_layers + 1)):
        # top layer (3)
        if layer == num_of_layers: 
            for i in range(input_width):
                #index = (prev_layer * nn_width * input_width) + i
                #print(weights[index])
                input = z_values[top_layer]
                partial_y_w = input[i]
                partial_derivatives.append(partial_l_y * partial_y_w)

            continue
        
        # layer 2
        if layer == 2:
            for node_index in range(nn_width):
                next_layer = layer - 1
                for i in range(input_width):
                    index = (top_layer * nn_width * input_width) + node_index + 1
                    #print(weights[index])
                    partial_y_z = weights[index][weight_value]
                    node_to = z_values[top_layer][node_index + 1]
                    node_from = z_values[next_layer][i]
                    partial_z_w = node_to * (1 - node_to) * node_from
                    partial_derivatives.append(partial_l_y * partial_y_z * partial_z_w)
            
            continue
        

        # layer 1
        if layer == 1:
            for node_index in range(nn_width):
                prev_layer = layer - 1
                for i in range(input_width):
                    sum = 0
                    for node_path_index in range(nn_width):
                        #print(node_index, i, node_path_index)
                        index = (top_layer * nn_width * input_width) + node_path_index + 1 
                        #print(weights[index])
                        partial_y_z = weights[index][weight_value]
                        #index = (layer * nn_width * input_width) + node_index + node_path_index + 1 
                        index = (layer * nn_width * input_width) + (nn_width * (i + 1)) + node_index + node_path_index
                        partial_z_z = weights[index][weight_value]
                        node_to = z_values[layer][node_index + 1]
                        node_from = z_values[prev_layer][i]
                        partial_z_w = node_to * (1 - node_to) * node_from
                        sum += partial_l_y * partial_y_z * partial_z_z * partial_z_w
                    partial_derivatives.append(sum)

        print(partial_derivatives)
        #print(len(partial_derivatives))



if __name__ == "__main__":
    main()