import math

def transpose(some_list):
    new_list = [[0 for i in range(len(some_list))] for j in range(len(some_list[0]))]
    for i in range(len(some_list)):
        for j in range(len(some_list[0])):
            new_list[j][i] = some_list[i][j]
    return new_list


def sigm(x):
    return 1/(1+math.exp(-x))

def output_value_calculation(input_list: list, weight_matrix: list) -> list: 
    value_input = []
    for element in weight_matrix:
        sum = 0
        for index in range(len(element)):
            sum += element[index]*input_list[index]
        value_input.append(sum)
    

    value_output = [sigm(x) for x in value_input]
        
    return value_output

def training_calculation(errors, weight_matrix, learning_rate=1):
    
    previous_layer_errors = []
    
    for i in range(len(weight_matrix[0])):
        value = 0
        for j in range(len(errors)):
            value += weight_matrix[j][i] * errors[j]
        previous_layer_errors.append(value)
    
    for i in range(len(errors)):
        sum_of_elements = sum(weight_matrix[i])
        for y in range(len(weight_matrix[i])):
            adjustment = (weight_matrix[i][y]/(sum_of_elements+0.01))*errors[i]*learning_rate
            
            weight_matrix[i][y] += adjustment

    new_weight_matrix = weight_matrix
    return new_weight_matrix, previous_layer_errors


input_list = [0.7,0.4]
weight_matrix_input_hidden = [[0.2, 0.4], [0.3, 0.1]]
weight_matrix_hidden_output = [[0.3, 0.1]]
targets = [0.6]


hidden_output = output_value_calculation(input_list, weight_matrix_input_hidden)
outputs = output_value_calculation(hidden_output, weight_matrix_hidden_output)
errors = [targets - outputs for targets, outputs in zip(targets, outputs)]
print(outputs, errors, sep='\n')

new_hidden_output_weight, hidden_output_errors = training_calculation(errors, weight_matrix_hidden_output)

new_input_hidden_weight = training_calculation(hidden_output_errors, weight_matrix_input_hidden)[0]


hidden_output = output_value_calculation(input_list, new_input_hidden_weight)
outputs = output_value_calculation(hidden_output, new_hidden_output_weight)
errors = [targets - outputs for targets, outputs in zip(targets, outputs)]
print(outputs, errors, sep='\n')


