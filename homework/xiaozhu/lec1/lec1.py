import numpy as np


# 1. 定义变量类的时候不要用 temp 表示输入字段
# 2. Function 基类的 forward 方法没有定义
# 3. Pow 函数中的 backward 方法是否需要这么多 if?

class Variable:
    def __init__(self, in_data):
        self.temp = in_data
        self.grad = None


class Function:
    def __call__(self, in_variable: Variable):
        x = in_variable.temp
        y = self.forward(x)
        self.in_variable = in_variable  # 用于反向传播
        out_variable = Variable(y)
        return out_variable

    def backward(self, in_dy):
        raise NotImplementedError()  # 用于补充


class Square(Function):
    def forward(self, in_x):
        return in_x ** 2

    def backward(self, in_dy):
        return (2 * self.in_variable.temp) * in_dy


class Exp(Function):
    def forward(self, in_x):
        return np.exp(in_x)

    def backward(self, in_dy):
        return np.exp(self.in_variable.temp) * in_dy


class Sin(Function):
    def forward(self, in_x):
        return np.sin(in_x)

    def backward(self, in_y):
        return np.cos(self.in_variable.temp) * in_y


# 余弦函数
class Cos(Function):
    def forward(self, in_x):
        return np.cos(in_x)

    def backward(self, in_y):
        return -np.sin(self.in_variable.temp) * in_y


# 绝对值函数
class Abs(Function):
    def forward(self, in_x):
        return np.abs(in_x)

    def backward(self, in_y):
        return in_y * np.sign(self.in_variable.temp)


# 负号函数
class Neg(Function):
    def forward(self, in_x):
        return -in_x

    def backward(self, in_y):
        return -in_y


# 指数函数
class Pow(Function):
    def __init__(self, power=2.0):
        self.power = power  # 幂指数作为类属性

    def forward(self, in_x):
        return np.power(in_x, self.power)  # 使用 self.power

    def backward(self, in_y):
        x = self.in_variable.temp
        if self.power == 2.0:  # 平方
            grad = 2.0 * x
        elif self.power == 1.0:  # 一次方
            grad = 1.0
        elif self.power == 0.0:  # 零次方
            grad = 0.0
        else:
            grad = self.power * np.power(x, self.power - 1.0)
        return grad * in_y

    # 数值微分, 传入函数和变量, 返回函数在这个变量上的微分
    def numerical_diff(func, in_var, eps=1e-4):
        x0 = Variable(in_var.temp - eps)
        x1 = Variable(in_var.temp + eps)
        y0 = func(x0)
        y1 = func(x1)
        return (y1.temp - y0.temp) / (2 * eps)


if __name__ == '__main__':
    class CompositeFunction(Function):
        def __init__(self):
            self.A = Square()
            self.B = Exp()
            self.C = Square()

        def forward(self, in_var):
            te = Variable(in_var)
            a_out = self.A(te)
            b_out = self.B(a_out)
            c_out = self.C(b_out)
            return c_out.temp

        def backward(self, in_dy):
            db = self.C.backward(in_dy)
            da = self.B.backward(db)
            dx = self.A.backward(da)
            return dx


    comp = CompositeFunction()
    y = comp(Variable(1))
    print("前向传播结果：", y.temp)
    print("反向传播结果：", comp.backward(1.0))

# 作业三
# 数据层面：从标量到张量
