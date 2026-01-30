import weakref

import numpy as np


# 定义变量类
class Variable:
    __array_priority__ = 999

    def __init__(self, input_data):
        if input_data is not None and not isinstance(input_data, np.ndarray):
            raise TypeError(
                'Variable类型的数据类型不正确, 请输入 np.ndarray 类型的数据, 而不是 {} 类型'.format(type(input_data)))
        self.data = input_data
        self.grad = None
        self.func = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return pow(other, self)

    def __neg__(self):
        return neg(self)

    def __abs__(self):
        return abs(self)

    def backward(self):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
            funcs = []
            visited = set()

            def add_func(temp_func):
                if temp_func not in visited:
                    visited.add(temp_func)
                    for temp_x in temp_func.input_variable:
                        if temp_x.func is not None:
                            add_func(temp_x.func)
                    funcs.append(temp_func)

            if self.func is not None:
                add_func(self.func)
            for f in funcs[::-1]:
                output_grads = [y().grad for y in f.output_variable]
                grads = f.backward(*output_grads)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for i, x in enumerate(f.input_variable):
                    if x.grad is None:
                        x.grad = grads[i]
                    else:
                        x.grad = x.grad + grads[i]


# 定义一个函数基类
class Function:
    def __call__(self, *inputs: Variable):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        # inputs = [as_variable(x) for x in input_variable]
        # xs=[x.data for x in inputs]

        # 将input_variable换成inputs就会报错？

        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        output_variable_list = [Variable(as_array(y)) for y in ys]
        for output in output_variable_list:
            output.func = self
        self.input_variable = inputs
        self.output_variable = [weakref.ref(out) for out in output_variable_list]
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]

    def forward(self, *input_x):
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
        return input_y * self.output_variable[0].data


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
        grad = np.where(x > 0, 1, np.where(x < 0, -1, 0))
        return input_y * grad


class Neg(Function):
    def forward(self, input_x):
        return np.negative(input_x)

    def backward(self, input_y):
        return -input_y


class Pow(Function):
    def __init__(self, power):
        self.power = power

    def forward(self, input_x):
        return input_x ** self.power

    def backward(self, input_y):
        (x,) = self.input_variable
        return input_y * self.power * x.data ** (self.power - 1)


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
        # return input_y / x2.data, -input_y * x1.data * x2.data ** (-2)  # 除法测试这种会报错
        return input_y / x2.data, -input_y * x1.data / x2.data ** (2)


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


def pow(x: Variable, power):
    return Pow(power)(x)


def square(input_variable: Variable):
    return Square()(input_variable)


def exp(input_variable: Variable):
    return Exp()(input_variable)


def add(x0: Variable, x1: Variable):
    x1 = as_array(x1)  # 不写这句也能正常运行Variable + scalar
    return Add()(x0, x1)


def sub(x0: Variable, x1: Variable):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def mul(x0: Variable, x1: Variable):
    return Mul()(x0, x1)


def div(x0: Variable, x1: Variable):
    return Div()(x0, x1)


def neg(input_variable: Variable):
    return Neg()(input_variable)


def abs(input_variable: Variable):
    return Abs()(input_variable)


def as_array(input_data):  # 将输入数据转换为 numpy 数组
    if np.isscalar(input_data):
        return np.array(input_data)
    return input_data


def as_variable(input_data):
    if isinstance(input_data, Variable):
        return input_data
    return Variable(as_array(input_data))


if __name__ == '__main__':
    # 第三课可视化运行，发现跑出的图是错的，将弱引用去掉之后，结果正确

    # 运算符重载测试
    print("\n" + "=" * 60)
    print("\n运算符重载测试")
    print("-" * 40)

    # 测试加法运算符
    print("\n1. 加法运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = x + y
    print(f"x + y = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}, ∂z/∂y = {y.grad.data}")

    # 测试反向加法运算符
    x = as_variable(as_array(2))
    z = 3.0 + x
    print(f"3.0 + x = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}")

    # 测试乘法运算符
    print("\n2. 乘法运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = x * y
    print(f"x * y = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}, ∂z/∂y = {y.grad.data}")

    # 测试反向乘法运算符
    x = as_variable(as_array(2))
    z = 3.0 * x
    print(f"3.0 * x = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}")

    # 测试减法运算符
    print("\n3. 减法运算符测试")
    x = as_variable(as_array(5))
    y = as_variable(as_array(3))
    z = x - y
    print(f"x - y = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}, ∂z/∂y = {y.grad.data}")

    # 测试反向减法运算符
    x = as_variable(as_array(3))
    z = 5.0 - x
    print(f"5.0 - x = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}")

    # 测试除法运算符
    print("\n4. 除法运算符测试")
    x = as_variable(as_array(6))
    y = as_variable(as_array(3))
    z = x / y
    print(f"x / y = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}, ∂z/∂y = {y.grad.data}")

    # 测试反向除法运算符
    x = as_variable(as_array(2))
    z = 6.0 / x
    print(f"6.0 / x = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}")

    # 测试幂运算符
    print("\n5. 幂运算符测试")
    x = as_variable(as_array(2))
    z = x ** 3
    print(f"x ** 3 = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}")

    # 注意，不实现反向幂运算

    # 测试取负运算符
    print("\n6. 取负运算符测试")
    x = as_variable(as_array(5))
    z = -x
    print(f"-x = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}")

    # 测试绝对值运算符
    print("\n7. 绝对值运算符测试")
    x = as_variable(as_array(-5))
    z = abs(x)
    print(f"abs(x) = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}")

    # 测试组合运算符
    print("\n8. 组合运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = (x + y) * (x - y)
    print(f"(x + y) * (x - y) = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}, ∂z/∂y = {y.grad.data}")

    # 测试平方函数
    print("\n9. 平方函数测试")
    x = as_variable(as_array(5))
    z = square(x)
    print(f"x² = {z.data}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.data}")

    print("\n" + "=" * 60)
    print("计算两个函数在 x=1, y=2 处的梯度")
    print("\n函数1: z = x² + y²")
    x1 = as_variable(as_array(1))
    y1 = as_variable(as_array(2))
    z1 = add(square(x1), square(y1))
    print(f"输入: x = {x1.data}, y = {y1.data} 结果: z = {z1.data}")
    z1.backward()
    print(f"梯度: ∂z/∂x = {x1.grad.data} ∂z/∂y = {y1.grad.data}\n")
    print("-" * 40)
    print("函数2: z = 0.26(x² + y²) - 0.48xy")
    x2 = as_variable(as_array(1))
    y2 = as_variable(as_array(2))
    z2 = sub(mul(Variable(as_array(0.26)), add(square(x2), square(y2))),
             mul(Variable(as_array(0.48)), mul(x2, y2)))
    print(f"输入: x = {x2.data}, y = {y2.data} 结果: z = {z2.data}")
    z2.backward()
    print(f"\n梯度: ∂z/∂x =  {x2.grad.data} ∂z/∂y = {y2.grad.data}")
