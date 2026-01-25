import numpy as np


# 定义变量类
class Variable:
    def __init__(self, input_data):
        if input_data is not None and not isinstance(input_data, np.ndarray):
            raise TypeError(
                'Variable类型的数据类型不正确, 请输入 np.ndarray 类型的数据, 而不是 {} 类型'.format(type(input_data)))
        self.data = input_data
        self.grad = None
        self.func = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        func = self.func
        if func is not None:
            output_grads = [y.grad for y in func.output_variable]
            gradList = func.backward(*output_grads)
            if not isinstance(gradList, tuple):
                gradList = (gradList,)
            for i, x in enumerate(func.input_variable):
                if x.grad is None:
                    x.grad = gradList[i]
                else:
                    x.grad = x.grad + gradList[i]
            for x in func.input_variable:
                x.backward()


# 定义一个函数基类
class Function:
    def __call__(self, *input_variable: Variable):
        xs = [x.data for x in input_variable]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        output_variable_list = [Variable(as_array(y)) for y in ys]
        for output in output_variable_list:
            output.func = self
        self.input_variable = input_variable
        self.output_variable = output_variable_list
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]

    def forward(self, input_x):
        raise NotImplementedError()

    def backward(self, input_y):
        raise NotImplementedError()


# 求平方函数
class Square(Function):
    def forward(self, input_x):  # 重写forward方法
        return input_x ** 2

    def backward(self, input_y):
        (x,) = self.input_variable
        return input_y * 2 * x.data


# exp函数
class Exp(Function):
    def forward(self, input_x):
        return np.exp(input_x)

    def backward(self, input_y):
        (x,) = self.input_variable
        return input_y * np.exp(input_y)


class Sin(Function):
    def forward(self, input_x):
        return np.sin(input_x)

    def backward(self, input_y):
        (x,) = self.input_variable
        return input_y * np.cos(x.data)


class Cos(Function):
    def forward(self, input_x):
        return np.cos(input_x)

    def backward(self, input_y):
        (x,) = self.input_variable
        return input_y * (-1) * np.sin(x.data)


class Abs(Function):
    def forward(self, input_x):
        return np.abs(input_x)

    def backward(self, input_y):
        (xx,) = self.input_variable
        x = xx.data
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
        return -input_y


class Pow(Function):
    def forward(self, input_x):
        return input_x ** 3

    def backward(self, input_y):
        (x,) = self.input_variable
        return input_y * 3 * x.data ** 2


class Add(Function):
    def forward(self, input_x1, input_x2):
        return input_x1 + input_x2

    def backward(self, input_y):
        return input_y, input_y


class Sub(Function):
    def forward(self, input_x1, input_x2):
        return input_x1 - input_x2

    def backward(self, input_y):
        return input_y, -input_y


class Mul(Function):
    def forward(self, input_x1, input_x2):
        return input_x1 * input_x2

    def backward(self, input_y):
        (x1, x2) = self.input_variable
        return input_y * x2.data, input_y * x1.data


class Div(Function):
    def forward(self, input_x1, input_x2):
        return input_x1 / input_x2

    def backward(self, input_y):
        (x1, x2) = self.input_variable
        return input_y / x2.data, -input_y * x1.data * x2.data ** (-2)


class CompositeFunction(Function):
    def __init__(self):
        self.A = Square()
        self.B = Exp()
        self.C = Square()

    def forward(self, input_x):
        x = Variable(input_x)
        return self.C(self.B(self.A(x))).data

    def backward(self, input_y):
        dx = self.A.backward(self.B.backward(self.C.backward(input_y)))
        return dx


def numerical_differentiation(func, input_x, eps=1e-4):
    x0 = Variable(input_x.data + eps)
    x1 = Variable(input_x.data - eps)
    y0 = func(x0)
    y1 = func(x1)
    return (y0.data - y1.data) / (2 * eps)


def pow(x):
    return Pow()(x)


def square(input_variable: Variable):
    return Square()(input_variable)


def exp(input_variable: Variable):
    return Exp()(input_variable)


def add(x0, x1):
    return Add()(x0, x1)


def sub(x0, x1):
    return Sub()(x0, x1)


def mul(x0, x1):
    return Mul()(x0, x1)


def div(x0, x1):
    return Div()(x0, x1)


def neg(input_variable: Variable):
    return Neg()(input_variable)


def abs(input_variable: Variable):
    return Abs()(input_variable)


def as_array(input_data):  # 将输入数据转换为 numpy 数组
    if np.isscalar(input_data):
        return np.array(input_data)
    return input_data


if __name__ == '__main__':
    # 测试sub
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(2.0))
    z = sub(x0, x1)
    z.backward()
    print("测试sub")
    print(z.data)
    print(x0.grad)
    print(x1.grad)
    # 测试mul
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(2.0))
    z = mul(x0, x1)
    z.backward()
    print("测试mul")
    print(z.data)
    print(x0.grad)
    print(x1.grad)
    # 测试pow
    x0 = Variable(np.array(1.0))
    z = pow(x0)
    z.backward()
    print("测试pow")
    print(z.data)
    print(x0.grad)
    # 测试div
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(2.0))
    z = div(x0, x1)
    z.backward()
    print("测试div")
    print(z.data)
    print(x0.grad)
    print(x1.grad)
    # 测试neg
    x0 = Variable(np.array(1.0))
    z = neg(x0)
    z.backward()
    print("测试neg")
    print(z.data)
    print(x0.grad)
    # 测试abs
    x0 = Variable(np.array(-1.0))
    z = abs(x0)
    z.backward()
    print("测试abs")
    print(z.data)
    print(x0.grad)

    print("z=(x^2+y^2)")
    x = Variable(np.array(1.0))
    y = Variable(np.array(2.0))
    z = add(square(x), square(y))  # (x^2+y^2)
    z.backward()
    print(z.data)
    print(x.grad)
    print(y.grad)

    x.grad = 0
    y.grad = 0

    print("z=0.26(x^2+y^2)-0.48xy")
    z1 = mul(Variable(np.array(0.26)), add(square(x), square(y)))  # 0.26(x^2+y^2)
    z2 = mul(Variable(np.array(0.48)), mul(x, y))  # 0.48xy
    z = sub(z1, z2)
    z.backward()
    print(z.data)
    print(x.grad)
    print(y.grad)
