import numpy as np


# 只有两种数据类型， Variable 类型， numpy.ndarray 类型

# 定义一个变量类
class Variable:
    def __init__(self, input_data):
        # 输入数据类型 numpy.ndarray
        if not isinstance(input_data, np.ndarray):
            raise TypeError("input_data must be a numpy.ndarray, input_data type: {}".format(type(input_data)))

        self.input_data = input_data
        self.grad = None  # 梯度, 初始值为 None
        self.creator = None  # 指向创建该变量的函数, 初始值为 None

    def set_grad(self, grad):
        self.grad = grad

    # 递归地通过变量和其创建函数，计算整个计算图中所有变量的梯度
    def backward(self):
        # 初始化梯度，如果没有的话。
        if self.grad is None:
            self.grad = np.ones_like(self.input_data)

        func = self.creator
        if func is not None:
            # 输出变量的梯度是一个数组
            output_grads = [temp3.grad for temp3 in func.output_variable]
            grad_list = func.backward(*output_grads)  # * 的作用是逆操作，作为多个独立的参数传给 backward 方法

            # 统一成元组
            if not isinstance(grad_list, tuple):
                grad_list = (grad_list,)

            # 对应每个输入变量, 赋值梯度
            for i, temp2 in enumerate(func.input_variable):
                if temp2.grad is None:
                    temp2.grad = grad_list[i]
                else:
                    temp2.grad = temp2.grad + grad_list[i]  # 为什么不能写成 temp.grad += grad_list[i] ?

            # 递归地计算所有输入变量的梯度
            for x in func.input_variable:
                x.backward()


# 定义一个函数类, 作为所有函数的基类
class Function:
    def forward(self, *input_data):  # 入参和出参 是 np.ndarray 类型, 而非 Variable 类型
        raise NotImplementedError

    # 反向传播, 入参是非 Variable 类型
    # backward 方法返回的数量必须和 forward 方法的输入参数的数量一致
    def backward(self, input_dy):
        raise NotImplementedError

    def __call__(self, *input_variable: Variable):
        self.input_variable = input_variable  # 保存输入变量, 用于反向传播时计算梯度

        xs = [temp2.input_data for temp2 in input_variable]

        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        output_variable_list = [Variable(as_array(temp2)) for temp2 in ys]

        self.output_variable = output_variable_list

        for output_variable in output_variable_list:
            output_variable.creator = self  # 保存创建该变量的函数

        # 规定，如果返回值列表中只有一个元素，那么则返回第一个元素
        # 单元素， 或者 多元素列表
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]


class Square(Function):
    def forward(self, input_data):
        return np.square(input_data)

    # 反向传播, 入参是非 Variable 类型
    def backward(self, input_dy):
        (temp,) = self.input_variable
        return (2 * temp.input_data) * input_dy


def square(input_variable: Variable):
    return Square()(input_variable)


class Exp(Function):
    def forward(self, input_data):
        return np.exp(input_data)

    # 反向传播, 入参是非 Variable 类型
    def backward(self, input_dy):
        (temp,) = self.input_variable
        return np.exp(temp.input_data) * input_dy


def exp(input_variable: Variable):  # Variable
    return Exp()(input_variable)


class Add(Function):
    def forward(self, input1_data, input2_data):  # 入参和出参 是 np.ndarray 类型, 而非 Variable 类型
        return input1_data + input2_data

    # backward 方法返回的数量必须和 forward 方法的输入参数的数量一致
    def backward(self, input_dy):
        return input_dy, input_dy


def add(input_variable: Variable, output_variable: Variable):
    return Add()(input_variable, output_variable)


# 数值微分函数, input_data 是 Variable 类型
def numerical_gradient(func: Function, input_variable: Variable, eps=1e-4):
    y0 = func(Variable(input_variable.input_data + eps))
    y1 = func(Variable(input_variable.input_data - eps))
    return Variable((y0.input_data - y1.input_data) / (2 * eps))


# 复合函数
class CompositeFunction(Function):
    def __init__(self):
        self.A = Square()
        self.B = Exp()
        self.C = Square()

    def forward(self, input_value):
        temp_var = Variable(input_value)
        # 通过__call__方法调用子函数，确保input_variable被正确保存
        a_output = self.A(temp_var)
        b_output = self.B(a_output)
        c_output = self.C(b_output)

        return c_output.input_value

    def backward(self, input_dy):
        # 反向传播，计算梯度
        dx = self.C.backward(input_dy)
        dx = self.B.backward(dx)
        dx = self.A.backward(dx)
        # 将计算得到的梯度保存到输入变量的grad属性中
        self.input_variable.grad = dx
        return dx


# 处理标量和数组的差异，兼容使用 numpy 数组进行计算
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


if __name__ == "__main__":
    x = Variable(np.array(1.0))
    a = square(x)
    b = exp(a)
    c = exp(a)
    y = add(b, c)
    y.backward()
    print(x.grad)  # 16.30969097075427  正确的梯度应该是 10.873127495050205

    # x = Variable(as_array(100))
    #
    # y = add(x, x)
    # print(y.input_data)
    # y.backward()
    # print(x.grad)
    #
    # # 隔离
    # x.set_grad(None)
    #
    # z = add(x, x)
    # print(z.input_data)
    # z.backward()
    # print(x.grad)

    # x1 = Variable(as_array(1))
    # x2 = Variable(as_array(2))
    # # y = (x1 + x2) ^ 2
    # y = square(add(x1, x2))
    #
    # print(y.input_data)
    #
    # y.grad = 1
    # y.backward()
    # print(x1.grad)
    # print(x2.grad)

    # y = Square()(x)
    # y.backward()
    # print(x.grad)
    # a = square(x)
    # b = exp(a)
    # y = square(b)
    # print("最终输出结果: ", y.input_data)

    # print(y.creator)
    # print(y.creator.input_variable.input_data)
    # print(y.creator.input_variable.creator.input_variable)
    # print(y.creator.input_variable.creator.input_variable.input_data)

    # 通过这种方式手搓反向传播, 迭代
    # y.grad = 1
    # y.backward()
    # print(x.grad)
