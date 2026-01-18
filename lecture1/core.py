import numpy as np


# 定义一个变量类
class Variable:
    def __init__(self, input_data):
        self.input_data = input_data
        self.grad = None  # 梯度, 初始值为 None


# 定义一个函数类, 作为所有函数的基类
class Function:
    def forward(self, input_data):  # 入参和出参 是非 Variable 类型
        raise NotImplementedError

    # 反向传播, 入参是非 Variable 类型
    def backward(self, input_dy):
        raise NotImplementedError

    def __call__(self, input_variable: Variable):
        self.input_variable = input_variable  # 保存输入变量, 用于反向传播时计算梯度
        x = input_variable.input_data
        y = self.forward(x)
        output_variable = Variable(y)
        return output_variable


class Square(Function):
    def forward(self, input_data):
        return np.square(input_data)

    # 反向传播, 入参是非 Variable 类型
    def backward(self, input_dy):
        return (2 * self.input_variable.input_data) * input_dy


class Exp(Function):
    def forward(self, input_data):
        return np.exp(input_data)

    # 反向传播, 入参是非 Variable 类型
    def backward(self, input_dy):
        return np.exp(self.input_variable.input_data) * input_dy


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


A = Square()
B = Exp()
C = Square()

x = Variable(-1)
a = A(x)
b = B(a)
y = C(b)
print("最终输出结果: ", y.input_data)

dy = 1.0
db = C.backward(dy)
print("y 对 b 的梯度: ", db)
da = B.backward(db)
print("y 对 a 的梯度: ", da)
dx = A.backward(da)
print("y 对 x 的梯度: ", dx)


def composeFunc(x_):
    a_ = A(x_)
    b_ = B(a_)
    c_ = C(b_)
    return c_


temp = numerical_gradient(composeFunc, x)
print("数值微分结果: ", temp.input_data)
