import weakref

import numpy as np


class Function:
    # 使用 * 写法把所有input_variable参数收集起来，打包成一个元组 args，这样则支持了任意数量输入变量，而不仅仅是一个
    def __call__(self, *input_variable: Variable):
        # 入参可能是非 Variable 类型，需要先转换成 Variable 类型
        input_variable = [as_variable(temp_x) for temp_x in input_variable]
        xs = [temp_x.value for temp_x in input_variable]  # 从元组中取出所有变量对象，取出实际值，放到列表 xs 中
        ys = self.forward(*xs)  # 解包元组中的元素
        # 有些函数只返回一个输出（比如 ReLU），有些返回多个输出（比如 split），为了让后面逻辑统一处理成可迭代对象，这里强制转成 tuple。
        if not isinstance(ys, tuple):
            ys = (ys,)
        output_variable_list = [Variable(as_array(temp_y)) for temp_y in ys]  # 将计算结果封装成变量对象并返回
        for output in output_variable_list:
            output.creator = self  # 保存创建函数，这样在反向传播时，可以沿着 output.creator 反查梯度来源

        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        self.output_variable = [weakref.ref(out) for out in output_variable_list]  # 保存输出变量，用于反向传播时计算梯度
        # 如果返回值列表中只有一个元素，则返回第 1 个元素。
        # 这种处理方式的优点是符合人类直觉，但缺点是返回值类型不固定，需要调用者根据实际情况决定如何取值，y, = Square(x) 单输出时加逗号解包  y1, y2 = split(x) # 多输出时正常解包
        # 作为教学项目比较合理，但工业级框架一般固定为返回一个 tuple/tensor, 这样可以统一处理单输出和多输出的情况
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]

    # 所有子类必须实现这个方法
    def forward(self, *input_x):
        raise NotImplementedError()

    # backward 方法的返回值必须和 forward 方法的输入参数数量一致. input_dy 是 Variable 类型，计算结果也是 Variable 类型
    def backward(self, input_dy: Variable):
        raise NotImplementedError()


# 将 np.ndarray 转换成 Variable 类型
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(input_data):
    if np.isscalar(input_data):
        return np.array(input_data)  # 转换成 np.array 类型
    return input_data


class Add(Function):
    def forward(self, input1, input2):
        return input1 + input2

    # backward 方法的返回值必须和 forward 方法的输入参数数量一致
    def backward(self, input_dy: Variable):
        return input_dy, input_dy


def add(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Add()(x0, x1)


class Multiplication(Function):
    def forward(self, input1, input2):
        return input1 * input2

    def backward(self, input_dy: Variable):
        (input_x0, input_x1) = self.input_variable
        return input_dy * input_x1, input_dy * input_x0


def mul(input_x0, input_x1):
    input_x1 = as_array(input_x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    input_x0 = as_array(input_x0)
    return Multiplication()(input_x0, input_x1)


class Sub(Function):
    def forward(self, input1, input2):
        return input1 - input2

    def backward(self, input_dy: Variable):
        return input_dy, - input_dy


def sub(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Sub()(x0, x1)


class Pow(Function):
    def __init__(self, power):
        self.power = power

    def forward(self, input_x):
        return np.power(input_x, self.power)

    def backward(self, input_dy: Variable):
        (input_x,) = self.input_variable
        temp = self.power * (input_x ** (self.power - 1)) * input_dy
        return temp


def pow(input_x, power):
    input_x = as_array(input_x)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Pow(power)(input_x)


class Div(Function):
    def forward(self, input1, input2):
        return input1 / input2

    def backward(self, input_dy: Variable):
        (input_x0, input_x1) = self.input_variable
        return input_dy / input_x1, -input_dy * input_x0 / (input_x1 ** 2)


def div(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Div()(x0, x1)


class Neg(Function):
    def forward(self, input_x: np.ndarray):
        return -input_x

    def backward(self, input_dy: Variable):
        return -input_dy


def neg(input_x):
    input_x = as_array(input_x)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Neg()(input_x)


class Abs(Function):
    def forward(self, input_x):
        return np.abs(input_x)

    def backward(self, input_dy: Variable):
        (input_x,) = self.input_variable
        return input_dy * np.sign(input_x.value)


def abs(input_x):
    input_x = as_array(input_x)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Abs()(input_x)


# 求平方函数，实现了 Function2 类
class Square(Function):
    def forward(self, square_input):
        return square_input ** 2

    def backward(self, input_dy: Variable):
        # 注意：对于单输入函数，input_variable是一个只有一个元素的元组
        # (x, ) 把一个只包含一个元素的元组解包（unpack）成变量 x
        (x,) = self.input_variable
        return 2 * x.value * input_dy


# 平方函数的便捷接口
def square(input_variable):
    input_variable = as_array(input_variable)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Square()(input_variable)


# Exp 函数，实现了 Function 类
class Exp(Function):
    def forward(self, input_x):
        return np.exp(input_x)

    def backward(self, input_dy: Variable):
        (out_dy,) = self.output_variable
        return input_dy * out_dy


# Exp 函数的便捷接口
def exp(input_variable):
    input_variable = as_array(input_variable)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Exp()(input_variable)


# 数值微分, 传入函数和变量, 返回函数在这个变量上的微分
def numerical_differentiation(func, input_var, eps=1e-4):
    x0 = as_variable(as_array(input_var.value - eps))
    x1 = as_variable(as_array(input_var.value + eps))
    y0 = func(x0)
    y1 = func(x1)
    return (y1.value - y0.value) / (2 * eps)


class Variable:
    __array_priority__ = 999

    def __init__(self, input_data, name=None):
        if input_data is not None and not isinstance(input_data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(input_data)))
        self.name = name
        self.value = input_data
        self.grad = None  # 梯度 默认为 None
        self.creator = None  # 创建函数 默认为 None

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    @property
    def dtype(self):
        return self.value.dtype

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        if self.value is None:
            return 'variable(None)'
        p = str(self.value).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    # 运算符重载
    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __pow__(self, other):
        return pow(self, other)

    # def __rpow__(self, other):
    #     return pow(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __neg__(self):
        return neg(self)

    def __abs__(self):
        return abs(self)

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.value))

        # 创建一个列表来存储需要处理的函数和梯度对
        funcs = []
        visited = set()  # 用于跟踪已访问的函数，避免重复处理

        # 后序遍历收集所有函数
        def add_func(temp_func):
            if temp_func not in visited:
                # 先添加输入变量的创建函数
                visited.add(temp_func)
                # 把输入变量的所有创建函数也添加到列表中
                for temp_xx in temp_func.input_variable:
                    if temp_xx.creator is not None:
                        add_func(temp_xx.creator)
                # 再添加当前函数
                funcs.append(temp_func)

        # 如果当前变量有创建函数，开始收集
        if self.creator is not None:
            add_func(self.creator)

        # 按照后序遍历的逆序（从输出到输入）处理每个函数
        for f in funcs[::-1]:
            # 计算当前函数的梯度
            output_grads = [temp_y().grad for temp_y in f.output_variable]  # 元素类型是 Variable 类型
            grads = f.backward(*output_grads)  # 计算结果是 Variable 类型
            if not isinstance(grads, tuple):
                grads = (grads,)

            # 将梯度传递给输入变量
            for i, temp_x in enumerate(f.input_variable):
                if temp_x.grad is None:
                    temp_x.grad = grads[i]
                else:
                    # 不能写成 temp_x.grad += grads[i]，否则在 Python 的语义中，就地修改原有对象，如果其他节点仍然在依赖这个 temp_x.grad, 会被污染数据。
                    temp_x.grad = temp_x.grad + grads[i]

            # 如果不需要保留梯度，则把中间变量的梯度都置为None
            if not retain_grad:
                for temp_y in f.output_variable:
                    temp_y().grad = None  # 弱引用使用 ()


if __name__ == '__main__':
    # 运算符重载测试
    print("\n" + "=" * 60)
    print("\n运算符重载测试")
    print("-" * 40)

    # 测试加法运算符
    print("\n1. 加法运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = x + y
    print(f"x + y = {z.value}")
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试反向加法运算符
    x = as_variable(as_array(2))
    z = 3.0 + x
    print(f"3.0 + x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试乘法运算符
    print("\n2. 乘法运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = x * y
    print(f"x * y = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试反向乘法运算符
    x = as_variable(as_array(2))
    z = 3.0 * x
    print(f"3.0 * x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试减法运算符
    print("\n3. 减法运算符测试")
    x = as_variable(as_array(5))
    y = as_variable(as_array(3))
    z = x - y
    print(f"x - y = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试反向减法运算符
    x = as_variable(as_array(3))
    z = 5.0 - x
    print(f"5.0 - x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试除法运算符
    print("\n4. 除法运算符测试")
    x = as_variable(as_array(6))
    y = as_variable(as_array(3))
    z = x / y
    print(f"x / y = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试反向除法运算符
    x = as_variable(as_array(2))
    z = 6.0 / x
    print(f"6.0 / x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试幂运算符
    print("\n5. 幂运算符测试")
    x = as_variable(as_array(2))
    z = x ** 3
    print(f"x ** 3 = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 注意，不实现反向幂运算

    # 测试取负运算符
    print("\n6. 取负运算符测试")
    x = as_variable(as_array(5))
    z = -x
    print(f"-x = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试绝对值运算符
    print("\n7. 绝对值运算符测试")
    x = as_variable(as_array(-5))
    z = abs(x)
    print(f"abs(x) = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    # 测试组合运算符
    print("\n8. 组合运算符测试")
    x = as_variable(as_array(2))
    y = as_variable(as_array(3))
    z = (x + y) * (x - y)
    print(f"(x + y) * (x - y) = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}, ∂z/∂y = {y.grad.value}")

    # 测试平方函数
    print("\n9. 平方函数测试")
    x = as_variable(as_array(5))
    z = square(x)
    print(f"x² = {z.value}")
    z.backward()
    print(f"梯度: ∂z/∂x = {x.grad.value}")

    print("\n" + "=" * 60)
    print("计算两个函数在 x=1, y=2 处的梯度")
    print("\n函数1: z = x² + y²")
    x1 = as_variable(as_array(1))
    y1 = as_variable(as_array(2))
    z1 = add(square(x1), square(y1))
    print(f"输入: x = {x1.value}, y = {y1.value} 结果: z = {z1.value}")
    z1.backward()
    print(f"梯度: ∂z/∂x = {x1.grad.value} ∂z/∂y = {y1.grad.value}\n")
    print("-" * 40)
    print("函数2: z = 0.26(x² + y²) - 0.48xy")
    x2 = as_variable(as_array(1))
    y2 = as_variable(as_array(2))
    z2 = sub(mul(Variable(as_array(0.26)), add(square(x2), square(y2))),
             mul(Variable(as_array(0.48)), mul(x2, y2)))
    print(f"输入: x = {x2.value}, y = {y2.value} 结果: z = {z2.value}")
    z2.backward()
    print(f"\n梯度: ∂z/∂x =  {x2.grad.value} ∂z/∂y = {y2.grad.value}")
