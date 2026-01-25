import weakref  # 引入弱引用模块，用于处理循环引用

import numpy as np


class Variable:
    def __init__(self, in_data):
        if in_data is not None and not isinstance(in_data, np.ndarray):  # 变量类型校验
            raise TypeError("Variable() only accepts numpy.ndarray , in_data type: {}".format(type(in_data)))

        self.value = in_data
        self.grad = None  # 梯度 默认为 None
        self.creator = None  # 本变量是由哪个函数创建的。 默认为 None

    def set_grad(self, grad):
        self.grad = grad

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.value))

        # 创建一个列表来存储需要处理的函数
        funcs = []
        visited = set()  # 用于跟踪已访问的函数，避免重复处理

        # 后序遍历收集所有函数
        def add_func(temp_func):
            if temp_func not in visited:
                # 先添加输入变量的创建函数
                visited.add(temp_func)
                # 把输入变量的所有创建函数也添加到列表中
                for temp_x in temp_func.input_variable:
                    if temp_x.creator is not None:
                        add_func(temp_x.creator)
                # 再添加当前函数
                funcs.append(temp_func)

        # 如果当前变量有创建函数，开始收集
        if self.creator is not None:
            add_func(self.creator)

        # 按照后序遍历的逆序（从输出到输入）处理每个函数
        for f in funcs[::-1]:
            # 计算当前函数的梯度
            output_grads = [y().grad for y in f.output_variable]
            grads = f.backward(*output_grads)
            if not isinstance(grads, tuple):
                grads = (grads,)

            # 将梯度传递给输入变量
            for i, temp2 in enumerate(f.input_variable):
                if temp2.grad is None:
                    temp2.grad = grads[i]
                else:
                    temp2.grad = temp2.grad + grads[i]  # 继续递归调用，计算连接图中所有变量的梯度

            for temp2 in funcs[::-1]:
                # 如果不需要保留梯度，则把中间变量的梯度都置为None
                if not retain_grad:
                    for y in temp2.output_variable:
                        y().grad = None  # 弱引用使用 ()

    # ============ 运算符重载 ============
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):  # 右加：other + self
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):  # 右乘：other * self
        return mul(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):  # 右减：other - self
        return sub(other, self)

    def __truediv__(self, other):  # 真除法：/
        return div(self, other)

    def __rtruediv__(self, other):  # 右真除法：other / self
        return div(other, self)


class Function:
    def __call__(self, *input_variable: Variable):
        # x = in_variable.value
        # y = self.forward(x)
        # self.input_variable = in_variable  # 用于反向传播
        # out_variable = Variable(y)
        # self.out_variable = out_variable
        # self.out_variable.creator = self #保存创建该变量的函数
        # return out_variable
        input_variable = [as_array(x) for x in input_variable]

        xs = [temp2.value for temp2 in input_variable]  # 从变量元组中取出所有变量的实际值。这一步的作用是 计算图层（Variable） → 数值计算层（ndarray）
        ys = self.forward(*xs)  # 是逆操作，把列表 / 元组拆开，作为多个独立参数传给函数。所以下面的 forward 方法做同步改造
        # 当 ys 不是元组时，用元组包裹起来，不然后面的迭代处理会出错
        if not isinstance(ys, tuple):
            ys = (ys,)
        output_variable_list = [Variable(as_array(temp2)) for temp2 in ys]  # 将计算结果封装成变量对象并返回
        for output in output_variable_list:
            output.creator = self  # 保存创建函数，用于反向传播时计算梯度

        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        self.output_variable = [weakref.ref(out) for out in output_variable_list]  # 保存输出变量，用于反向传播时计算梯度
        # 如果返回值列表中只有一个元素，则返回第 1 个元素
        # 切记，函数的 call 方法返回的是 单元素或者多元素列表
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]

    def forward(self, *in_x):
        raise NotImplementedError()  # 用于补充

    # 反向传播，入参是非 Variable 类型
    # backward 方法返回的数值必须和 forward 方法的输入参数的数量一致
    def backward(self, in_dy):
        raise NotImplementedError()  # 用于补充


class Add(Function):
    def forward(self, input1_data, input2_data):
        return input1_data + input2_data

    def backward(self, input_dy):
        return input_dy, input_dy


def add(input1_variable: Variable, input2_variable: Variable):
    return Add()(input1_variable, input2_variable)


# ==================== 减法 ====================
class Sub(Function):
    def forward(self, input1_data, input2_data):
        return input1_data - input2_data

    def backward(self, input_dy):
        return input_dy, -input_dy


def sub(input1_variable: Variable, input2_variable: Variable):
    return Sub()(input1_variable, input2_variable)


# ==================== 乘法 ====================
class Mul(Function):
    def forward(self, input1_data, input2_data):
        return input1_data * input2_data

    def backward(self, input_dy):
        x1 = self.input_variable[0].value
        x2 = self.input_variable[1].value
        return input_dy * x2, input_dy * x1


def mul(input1_variable: Variable, input2_variable: Variable):
    return Mul()(input1_variable, input2_variable)


# ==================== 除法 ====================
class Div(Function):
    def forward(self, input1_data, input2_data):
        return input1_data / input2_data

    def backward(self, input_dy):
        x1 = self.input_variable[0].value
        x2 = self.input_variable[1].value
        gx1 = input_dy / x2
        gx2 = -input_dy * x1 / (x2 ** 2)
        return gx1, gx2


def div(input1_variable: Variable, input2_variable: Variable):
    return Div()(input1_variable, input2_variable)


class Square(Function):
    def forward(self, in_x: np.ndarray):
        return in_x ** 2

    def backward(self, in_dy):
        if in_dy is None:
            in_dy = 1.0  # 默认梯度为1
        # (x, ) 把一个只包含一个元素的元组解包（unpack）成变量 x
        (temp2,) = self.input_variable
        return 2 * temp2.value * in_dy  # 在 func.backward 之后会被封装成元组
        # return (2 * self.input_variable.value) * in_dy


def square(x):
    """支持标量和Variable的平方"""
    if not isinstance(x, Variable):
        x = Variable(np.array(float(x)))
    return Square()(x)


class Exp(Function):
    def forward(self, in_x):
        return np.exp(in_x)

    def backward(self, input_dy):
        (temp2,) = self.input_variable
        return temp2.value * input_dy


def exp(input_variable: Variable):
    return Exp()(input_variable)


class Sin(Function):
    def forward(self, in_x):
        return np.sin(in_x)

    def backward(self, in_y):
        (temp2,) = self.input_variable
        return np.cos(temp2.value) * in_y


# 余弦函数
class Cos(Function):
    def forward(self, in_x):
        return np.cos(in_x)

    def backward(self, in_y):
        (temp2,) = self.input_variable
        return -np.sin(temp2.value) * in_y


# 绝对值函数
class Abs(Function):
    def forward(self, in_x):
        return np.abs(in_x)

    def backward(self, in_y):
        (temp2,) = self.input_variable
        return in_y * np.sign(temp2.value)


# 负号函数
class Neg(Function):
    def forward(self, in_x):
        return np.negative(in_x)

    def backward(self, in_y):
        return -in_y


# 指数函数
class Pow(Function):
    def __init__(self, power=2.0):
        self.power = power  # 幂指数作为类属性

    def forward(self, in_x):
        return np.power(in_x, self.power)  # 使用 self.power

    def backward(self, in_y):
        (temp2,) = self.input_variable
        x = temp2.value
        return self.power * np.power(x, self.power - 1.0) * in_y

    # 数值微分, 传入函数和变量, 返回函数在这个变量上的微分
    def numerical_diff(func, in_var, eps=1e-4):
        x0 = Variable(in_var.value - eps)
        x1 = Variable(in_var.value + eps)
        y0 = func(x0)
        y1 = func(x1)
        return (y1.value - y0.value) / (2 * eps)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)  # 转换成 np.array 类型
    return x


if __name__ == '__main__':
    print("计算两个函数在 x=1, y=2 处的梯度")

    # 函数1: z = x² + y²
    print("\n函数1: z = x² + y²")
    print("-" * 40)

    x1 = Variable(np.array(1.0))
    y1 = Variable(np.array(2.0))
    z1 = add(square(x1), square(y1))

    print(f"输入: x = {x1.value}, y = {y1.value}")
    print(f"结果: z = {z1.value}")

    z1.backward()
    print(f"\n梯度:")
    print(f"∂z/∂x = {x1.grad}")
    print(f"∂z/∂y = {y1.grad}")

    # 函数2: z = 0.26(x² + y²) - 0.48xy
    print("\n" + "=" * 60)
    print("\n函数2: z = 0.26(x² + y²) - 0.48xy")
    print("-" * 40)

    x2 = Variable(np.array(1.0))
    y2 = Variable(np.array(2.0))
    z2 = sub(mul(Variable(as_array(0.26)), add(square(x2), square(y2))),
             mul(Variable(as_array(0.48)), mul(x2, y2)))

    print(f"输入: x = {x2.value}, y = {y2.value}")
    print(f"结果: z = {z2.value}")

    z2.backward()
    print(f"\n梯度:")
    print(f"∂z/∂x = {x2.grad}")
    print(f"∂z/∂y = {y2.grad}")
