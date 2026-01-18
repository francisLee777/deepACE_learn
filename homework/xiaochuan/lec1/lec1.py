import numpy as np


# 1. Variable 变量的梯度字段呢？
# 2. Neg 的反向传播函数是否可以简化
# 3. 作业3回答正确

# 定义变量类
class Variable:
    def __init__(self, input_data):
        self.data = input_data
        self.grad = None  # 梯度, 初始值为 None


# 定义一个函数基类
class Function:
    def __call__(self, input_variable: Variable):
        self.input_variable = input_variable
        x = input_variable.data
        y = self.forward(x)
        output_variable = Variable(y)
        return output_variable

    def forward(self, input_x):
        raise NotImplementedError()

    def backward(self, input_y):
        raise NotImplementedError()


# 求平方函数
class Square(Function):
    def forward(self, input_x):  # 重写forward方法
        return input_x ** 2

    def backward(self, input_y):
        return input_y * 2 * self.input_variable.data


# exp函数
class Exp(Function):
    def forward(self, input_x):
        return np.exp(input_x)

    def backward(self, input_y):
        return input_y * np.exp(self.input_variable.data)


class Sin(Function):
    def forward(self, input_x):
        return np.sin(input_x)

    def backward(self, input_y):
        return input_y * np.cos(self.input_variable.data)


class Cos(Function):
    def forward(self, input_x):
        return np.cos(input_x)

    def backward(self, input_y):
        return input_y * (-1) * np.sin(self.input_variable.data)


class Abs(Function):
    def forward(self, input_x):
        return np.abs(input_x)

    def backward(self, input_y):
        x = self.input_variable.data
        if x > 0:
            return input_y
        elif x < 0:
            return -input_y
        else:
            return 0


class Neg(Function):
    def forward(self, input_x):
        return np.negative(input_x)

    def backward(self, input_y):
        x = self.input_variable.data
        if x < 0:
            return input_y
        elif x > 0:
            return -input_y
        else:
            return 0


class Pow(Function):
    def forward(self, input_x):
        return input_x ** 3

    def backward(self, input_y):
        return input_y * 3 * self.input_variable.data ** 2


class CompositeFunction(Function):
    def __init__(self):
        self.A = Square()
        self.B = Exp()
        self.C = Square()

    def forward(self, input_x):
        x = Variable(input_x)
        return self.C(self.B(self.A(x))).data

    def backward(self, input_y):
        db = self.C.backward(input_y)
        da = self.B.backward(db)
        dx = self.A.backward(da)
        return dx


def numerical_differentiation(func, input_x, eps=1e-4):
    x0 = Variable(input_x.data + eps)
    x1 = Variable(input_x.data - eps)
    y0 = func(x0)
    y1 = func(x1)
    return (y0.data - y1.data) / (2 * eps)


if __name__ == '__main__':
    f_sin = Sin()
    f_cos = Cos()
    x = Variable(np.pi / 2)
    result_sin = f_sin(x).data
    result_cos = f_cos(x).data
    print(f"result_sin={result_sin}")
    print(f"result_cos={result_cos}")
    dx_sin = f_sin.backward(1.0)
    dx_cos = f_cos.backward(1.0)
    print(f"dx_sin={dx_sin}")
    print(f"dx_cos={dx_cos}")

    f_abs = Abs()
    f_neg = Neg()
    f_pow = Pow()
    x = Variable(2)
    result_abs = f_abs(x).data
    result_neg = f_neg(x).data
    result_pow = f_pow(x).data
    print(f"result_abs={result_abs}")
    print(f"result_neg={result_neg}")
    print(f"result_pow={result_pow}")
    dx_abs = f_abs.backward(1.0)
    dx_neg = f_neg.backward(1.0)
    dx_pow = f_pow.backward(1.0)
    print(f"dx_abs={dx_abs}")
    print(f"dx_neg={dx_neg}")
    print(f"dx_pow={dx_pow}")
"""
第三题思路：重写函数基类
n元需要n个变量
对于forward方法：输入n个变量进行计算
对于backward方法：分别对n个变量求偏导

"""
