from math import exp

input_signal = [0.1, 0.2]
weight_input_hidden = [0.2, 0.3, 0.4, 0.5]
weight_hidden_output = [0.35, 0.25]

def act(x):
    return 1/(1+exp(-x))
    
def forward_propagation():
    i1, i2 = input_signal
    w11_1, w21_1, w12_1, w22_1 = weight_input_hidden
    w11_2, w21_2 = weight_hidden_output

    hidden_input = [w11_1*i1 + w21_1*i2, w12_1*i1 + w22_1*i2]
    hidden_output = [act(x) for x in hidden_input]

    o1, o2 = hidden_output

    output_input = [o1*w11_2 + o2*w21_2]
    output_output = [act(output_input[0])]
    return output_output


def back_propagation(output):
    target = 0.6
    learning_rate = 0.01
    w11_1, w21_1, w12_1, w22_1 = weight_input_hidden
    w11_2, w21_2 = weight_hidden_output

    error = target - output[0]

    hidden_error = [(w11_2/(w11_2+w21_2)) * error, (w21_2/(w11_2+w21_2)) * error]
    new_hidden_weight = [w11_2 + (w11_2/(w11_2+w21_2)) * error * learning_rate, (w21_2/(w11_2+w21_2)) * error* learning_rate]

    
    new_input_weight = [w11_1 + ((w11_1/(w11_1+w21_1))*hidden_error[0])*learning_rate, w21_1 + ((w21_1/(w11_1+w21_1))*hidden_error[0])*learning_rate, w12_1 + ((w12_1/(w12_1+w22_1))*hidden_error[1])*learning_rate, w22_1 + ((w22_1/(w12_1+w22_1))*hidden_error[1])*learning_rate]

    print(new_input_weight)

output = forward_propagation()
back_propagation(output)