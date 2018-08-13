import numpy as np


def relu(in_num):
    l1 = []
    in_num = in_num.reshape((in_num.shape[1], 1))
    for each in in_num:
        if each < 0:
            output = 0
        else:
            output = each.tolist()
            output = output[0]
        l1.append(output)
    outputs = np.array(l1)[np.newaxis, :]
    return outputs


def tanh(in_num):
    outputs = np.tanh(in_num)
    return outputs


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = np.random.random((in_size, out_size)) * 2 - 1
    biases = np.zeros((1, out_size)) + 0.1
    wx_plus_b = np.dot(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


# example of hidden_list:[16,8,4, ...], means the first hidden layer's size is 16, next layer's size is 8, and so on.
def net_work(inputs, hidden_list, funcs):
    if len(hidden_list) == len(funcs):
        for i, s in enumerate(hidden_list):
            locals()['l%s' % (i+1)] = add_layer(inputs, inputs.shape[1], s, funcs[i])
            inputs = locals()['l%s' % (i+1)]
        outputs = locals()['l%s' % (i+1)]
        return outputs
    else:
        print("The amount of hidden layer didn't fit with the amount of activation function!")


if __name__ == '__main__':
    x = np.linspace(-5, 5, 20).reshape((1, 20))
    y = net_work(x, [16, 8, 4], [relu, tanh, None])
    print(y)


