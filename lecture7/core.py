import math
import os
import subprocess
import warnings
import weakref
import mnist
from matplotlib import pyplot as plt

mnist.temporary_dir = lambda: './'
# 默认的数据源url好像失效了，使用下面的url
mnist.datasets_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
# 将警告转换为错误，便于定位
import numpy as np


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

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return transpose(self)

    def reshape(self, *shape):  # 多入参
        if len(shape) == 1:
            shape = shape[0]
        return reshape(self, shape)

    def matmul(self, other):
        return matmul(self, other)

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

    def __getitem__(self, item):
        return get_item(self, item)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

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


#  ———————————————————————— start 基础运算：加减乘除,平方,指数,幂次, sin/cos/tan/log ——————————————————————————————
class Add(Function):

    def __init__(self):
        # 新增 shape 的记录
        self.input1_shape = None
        self.input2_shape = None

    def forward(self, input1, input2):
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        return input1 + input2

    # backward 方法的返回值必须和 forward 方法的输入参数数量一致
    def backward(self, input_dy: Variable):
        input_dy1, input_dy2 = input_dy, input_dy
        # 处理广播情况
        if self.input1_shape != self.input2_shape:
            input_dy1 = sum_to(input_dy1, self.input1_shape)
            input_dy2 = sum_to(input_dy2, self.input2_shape)
        return input_dy1, input_dy2


def add(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Add()(x0, x1)


class Multiplication(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self, input1, input2):
        self.input1_shape, self.input2_shape = input1.shape, input2.shape
        return input1 * input2

    def backward(self, input_dy):
        (input_x0, input_x1) = self.input_variable
        # 处理广播
        dy1, dy2 = input_dy * input_x1.value, input_dy * input_x0.value
        if self.input1_shape != self.input2_shape:
            dy1 = sum_to(dy1, self.input1_shape)
            dy2 = sum_to(dy2, self.input2_shape)
        return dy1, dy2


def mul(input_x0, input_x1):
    input_x1 = as_array(input_x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    input_x0 = as_array(input_x0)
    return Multiplication()(input_x0, input_x1)


class Sub(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self, input1, input2):
        self.input1_shape, self.input2_shape = input1.shape, input2.shape
        return input1 - input2

    def backward(self, input_dy: Variable):
        dy1, dy2 = input_dy, - input_dy
        if self.input1_shape != self.input2_shape:
            dy1 = sum_to(dy1, self.input1_shape)
            dy2 = sum_to(dy2, self.input2_shape)
        return dy1, dy2


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
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self, input1, input2):
        self.input1_shape, self.input2_shape = input1.shape, input2.shape
        return input1 / input2

    def backward(self, input_dy: Variable):
        (input_x0, input_x1) = self.input_variable
        dy1, dy2 = input_dy / input_x1, -input_dy * input_x0 / (input_x1 ** 2)
        # 处理广播
        if self.input1_shape != self.input2_shape:
            dy1 = sum_to(dy1, self.input1_shape)
            dy2 = sum_to(dy2, self.input2_shape)
        return dy1, dy2


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
        return input_dy * out_dy()


# Exp 函数的便捷接口
def exp(input_variable):
    input_variable = as_array(input_variable)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Exp()(input_variable)


class Sin(Function):
    def forward(self, input_x):
        temp_y = np.sin(input_x)
        return temp_y

    def backward(self, dy):
        (x,) = self.input_variable
        dx = dy * cos(x)
        return dx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, input_x):
        y = np.cos(input_x)
        return y

    def backward(self, dy):
        (x,) = self.input_variable
        dx = dy * -sin(x)
        return dx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, input_x):
        temp_y = np.tanh(input_x)
        return temp_y

    def backward(self, dy):
        temp_y = self.output_variable[0]()
        dx = dy * (1 - temp_y * temp_y)
        return dx


def tanh(x):
    return Tanh()(x)


class Log(Function):
    def forward(self, input_x):
        y = np.log(input_x)
        return y

    def backward(self, dy):
        (x,) = self.input_variable
        dx = dy / x
        return dx


def log(x):
    return Log()(x)


class MatMul(Function):
    def forward(self, input_x, input_W):
        return input_x @ input_W

    def backward(self, input_dy: Variable):
        input_x, input_W = self.input_variable
        dx = matmul(input_dy, input_W.T)
        dW = matmul(input_x.T, input_dy)
        return dx, dW


def matmul(input_x, input_W):
    return MatMul()(input_x, input_W)


#  ———————————————————————— end 基础运算：加减乘除,平方,指数,幂次, sin/cos/tan/log  ——————————————————————————————

#  ———————————————————————— start 激活函数  sigmoid/relu 等  ——————————————————————————————
class Sigmoid(Function):
    def forward(self, x):
        # y = 1 / (1 + np.exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, dy):
        y = self.output_variable[0]
        dx = dy * y() * (1 - y())
        return dx


def sigmoid(x):
    return Sigmoid()(x)


def sigmoid_simple(x):
    y = 1 / (1 + exp(-x))
    return y


#  ———————————————————————— end 激活函数  sigmoid/relu 等  ——————————————————————————————

#  ———————————————————————— start 改变形状: reshape, 转秩, 广播/求和  ——————————————————————————————
class Reshape(Function):
    def __init__(self, target_shape):
        self.target_shape = target_shape
        self.original_shape = None

    def forward(self, input_x):
        self.original_shape = input_x.shape  # 记录一下原始的形式
        return np.reshape(input_x, self.target_shape)

    def backward(self, input_dy: Variable):
        # 这里要使用自身的 reshape 函数， 而不是 np.reshape 函数
        # 因为 input_dy 的类型是 Variable 类型，不能用 np.reshape 直接处理
        return reshape(input_dy, self.original_shape)  # 反向传播时，需要将 dy 的形状恢复到初始input_x 的形状


def reshape(input_x, target_shape):
    return Reshape(target_shape)(as_variable(as_array(input_x)))


class Transpose(Function):
    def forward(self, input_x):
        return np.transpose(input_x)

    def backward(self, input_dy: Variable):
        return transpose(input_dy)


def transpose(input_x):
    return Transpose()(input_x)


class BroadcastTo(Function):
    def __init__(self, target_shape):
        self.original_shape = None
        self.target_shape = target_shape

    def forward(self, input_x):
        self.original_shape = input_x.shape
        return np.broadcast_to(input_x, self.target_shape)

    def backward(self, input_dy: Variable):
        return sum_to(input_dy, self.original_shape)


def broadcast_to(input_x, target_shape):
    if input_x.shape == target_shape:
        return as_variable(input_x)
    return BroadcastTo(target_shape)(as_variable(as_array(input_x)))


def util_sum_to(input_x, target_shape):
    y = input_x
    # 处理广播对齐过程中新增的维度：input_x 比 target_shape 多出来的“前导维度”（leading dimensions）
    while y.ndim > len(target_shape):
        y = y.sum(axis=0)
    # 对 shape=1 的维度求和。被拉伸的维度：target_shape 中为 1，但在 input_x 中被拉伸为 N 的维度。
    for i, sx in enumerate(target_shape):
        if sx == 1:
            y = y.sum(axis=i, keepdims=True)
    return y


class SumTo(Function):
    def __init__(self, target_shape):
        self.original_shape = None
        self.target_shape = target_shape

    def forward(self, input_x):
        self.original_shape = input_x.shape
        return util_sum_to(input_x, self.target_shape)

    def backward(self, input_dy: Variable):
        return broadcast_to(input_dy, self.original_shape)


def sum_to(input_x, target_shape):
    if input_x.shape == target_shape:
        return as_variable(input_x)
    return SumTo(target_shape)(as_variable(as_array(input_x)))


class Sum(Function):
    """
    沿指定轴计算张量的元素总和。
    """

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims  # 和 numpy.sum() 一样，有 keepdims 参数，可选是否保持维度不变
        self.output_shape_kept = None
        self.origin_shape = None

    def forward(self, input_x):
        """
        执行前向传播。
        1. 保存输入形状 `self.origin_shape`，这对于反向传播至关重要。
        2. 计算并保存 `self.output_shape_kept`，记录如果 forward 阶段用了 keepdims=True，
        输出本该是什么 shape，从而在 backward 阶段把梯度 reshape / broadcast 回输入的形状。
        3. 使用 np.sum 执行实际的求和操作。
        """
        self.origin_shape = input_x.shape
        # 如果不传 axis，意思是把所有元素加起来，得出一个标量。这时候要保存
        if self.axis is None:
            self.output_shape_kept = tuple(np.ones(input_x.ndim, dtype=int))
        else:
            # 处理 axis 为 int 或 tuple 的情况。轴可以是单个值，也可以是多个值
            if isinstance(self.axis, int):
                axis_tuple = (self.axis,)
            else:
                axis_tuple = self.axis
            # 归一化轴索引（确保为正整数）
            # 因为在 python 中，下标和轴的值都可以为负数，例如 arr[-1] 指最后一个元素
            normalized_axis = [ax % input_x.ndim for ax in axis_tuple]
            shape_list = list(input_x.shape)
            for ax in normalized_axis:
                shape_list[ax] = 1
            self.output_shape_kept = tuple(shape_list)
        # 执行求和操作
        y = np.sum(input_x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, dy):
        """
        执行反向传播。
        1. 通过 reshape 调整梯度形状。
        2. 使用广播机制将梯度广播回原始输入形状。
        """
        # 将梯度 reshape 为 "keepdims=True" 时的形状
        dy_reshaped = reshape(dy, self.output_shape_kept)

        # 将梯度广播回原始形状
        dx = broadcast_to(dy_reshaped, self.origin_shape)
        return dx


def sum(input_x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(input_x)


#  ———————————————————————— end 改变形状: reshape, 转秩，广播/求和  ——————————————————————————————

# 数值微分, 传入函数和变量, 返回函数在这个变量上的微分
def numerical_differentiation(func, input_var, eps=1e-4):
    x0 = as_variable(as_array(input_var.value - eps))
    x1 = as_variable(as_array(input_var.value + eps))
    y0 = func(x0)
    y1 = func(x1)
    return (y1.value - y0.value) / (2 * eps)


def numerical_gradient_matrix_x(f, x, W, eps=1e-4):
    # 获取x的原始数据
    x_data = x.value
    grad = np.zeros_like(x_data)

    # 对x的每个元素进行扰动
    for idx in np.ndindex(x_data.shape):
        x_plus = x_data.copy()
        x_minus = x_data.copy()
        # 正向扰动
        x_plus[idx] = x_plus[idx] + eps
        y1 = f(Variable(x_plus), W)
        # 负向扰动
        x_minus[idx] = x_minus[idx] - eps
        y2 = f(Variable(x_minus), W)
        # 中心差分法计算梯度
        temp = (y1 - y2).value

        grad[idx] = temp / (2 * eps)
    return grad


def numerical_gradient_matrix_w(f, x, W, eps=1e-4):
    # 获取W的原始数据
    W_data = W.value
    grad = np.zeros_like(W_data)

    # 对W的每个元素进行扰动
    for idx in np.ndindex(W_data.shape):
        W_plus = W_data.copy()
        W_minus = W_data.copy()
        # 正向扰动
        W_plus[idx] = W_plus[idx] + eps
        y1 = f(x, Variable(W_plus))
        # 负向扰动
        W_minus[idx] = W_minus[idx] - eps
        y2 = f(x, Variable(W_minus))
        # 中心差分法计算梯度
        temp = (y1 - y2).value

        grad[idx] = temp / (2 * eps)
    return grad


#  ———————————————————————— start 基础的深度学习网络组件  ——————————————————————————————

class Linear(Function):
    def forward(self, x, W, b):
        y = x @ W
        if b is not None:  # 偏置，是可选项
            y += b
        return y

    def backward(self, dy):
        x, W, b = self.input_variable
        db = None if b.value is None else sum_to(dy, b.shape)
        dx = matmul(dy, W.T)
        dW = matmul(x.T, dy)
        return dx, dW, db


def linear(input_x, W, b=None):
    return Linear()(input_x, W, b)


class MeanSquaredError(Function):
    def forward(self, y0, y1):
        diff = y1 - y0
        # 注意， sum 函数返回的是 Variable 类型，但在forward 方法中，要返回非Variable类型
        return sum(diff ** 2).value / len(diff)

    def backward(self, dy):
        y0, y1 = self.input_variable
        diff = y1 - y0
        dy0 = dy * diff * (2.0 / len(diff))
        dy1 = -dy0
        return dy0, dy1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def abs_loss(x0, x1):
    diff = abs(x1 - x0)
    return sum(diff) / len(diff)  # 除以样本数量, 防止误差过大溢出以及学习率无法调整


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x[self.slices]
        return y

    def backward(self, dy):
        # 构造一个与原始输入相同形状的 0 数组
        dx = np.zeros(self.x_shape, dtype=dy.dtype)
        # np.add.at 可以实现“稀疏加法”（用于切片梯度还原）
        np.add.at(dx, self.slices, dy.value)
        # 最终要返回 Variable 对象
        return Variable(dx)


def get_item(x, slices):
    return GetItem(slices)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        return np.clip(x, self.x_min, self.x_max)

    def backward(self, dy):
        (x,) = self.input_variable
        mask = (x.value >= self.x_min) * (x.value <= self.x_max)
        dx = dy * mask
        return dx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


#  ———————————————————————— end 基础的深度学习网络组件  ——————————————————————————————

#  ———————————————————————— start 计算图构建工具，输出png  ———————————————————————————
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.value is not None:
        if v.name is not None:
            name += ': '
        name += str(v.value.shape) + ' '

    return dot_var.format(id(v), name)


def _dot_func(f):
    # for function
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # for edge
    dot_edge = '{} -> {}\n'
    for x in f.input_variable:
        ret += dot_edge.format(id(x), id(f))
    for y in f.output_variable:
        ret += dot_edge.format(id(f), id(y()))
    return ret


def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    visited = set()

    def add_func(f):
        if f not in visited:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            visited.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.input_variable:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph_ouput/graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.test')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


#  ————————————————————————  end 计算图构建工具，输出png  ———————————————————————————


#  ——————————————————————— start 参数,层,网络模型等高层概念  —————————————————————————
class Parameter(Variable):

    def clear_grad(self):
        self.grad = None


# Layer 层
class Layer:
    def __init__(self):
        self._params_name = set()  # 名字集合，无序而且元素是唯一的

    # 特殊方法，在设置字段值的时候，会调用这个函数
    def __setattr__(self, name, value):
        # 主要用来支持集合类型中的 Parameter/Layer 判断
        def _contains_param(v):
            # 只搜集 Parameter/Layer 类，不搜集 Variable 类和其他类型
            if isinstance(v, (Parameter, Layer)):
                return True
            if isinstance(v, (list, tuple, set)):
                return any(_contains_param(item) for item in v)
            if isinstance(v, dict):
                return any(_contains_param(item) for item in v.values())
            return False

        is_container = isinstance(value, (list, tuple, set, dict))  # 防止先创建空集合然后再添加元素的情况
        if name != '_params_name' and hasattr(self, '_params_name') and (is_container or _contains_param(value)):
            self._params_name.add(name)
        super().__setattr__(name, value)

    def forward(self, inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs, self.outputs = list(inputs), list(outputs)
        return outputs if len(outputs) > 1 else outputs[0]

    def params(self):
        # 定义一个函数，用来递归地获取所有的 Parameter/Layer 类
        def _yield_params(obj):
            if isinstance(obj, Parameter):
                yield obj
            if isinstance(obj, Layer):
                yield from obj.params()
            if isinstance(obj, dict):
                for item in obj.values():
                    yield from _yield_params(item)
            if isinstance(obj, (list, tuple, set)):
                for item in obj:
                    yield from _yield_params(item)

        for name, obj in self.__dict__.items():
            if name == '_params_name':
                continue
            if obj is not None:
                yield from _yield_params(obj)

    # 打平参数
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params_name:
            # __dict__ 是类的属性字典，包含了类的所有属性，包括实例属性和类属性
            obj = self.__dict__[name]
            # key 的设计是支持嵌套的，例如：layer1/W 表示 layer1 层的权重参数 W
            key = parent_key + '/' + name if parent_key else name

            # 如果是 Layer 类型，递归调用 _flatten_params 方法，最终都将参数加入到 params_dict 中
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            elif isinstance(obj, Parameter):
                params_dict[key] = obj
            elif isinstance(obj, (list, tuple, set)):  # 暂不处理 dict 类型
                for i, item in enumerate(obj):
                    sub_key = f"{key}/{i}"
                    if isinstance(item, Layer):
                        item._flatten_params(params_dict, sub_key)
                    elif isinstance(item, Parameter):
                        params_dict[sub_key] = item
            else:
                continue

    def save_params(self, file_path="params.npz"):
        params_dict = {}
        self._flatten_params(params_dict)
        # params_dict 中的对象是 Parameter 类型，要转换成 numpy 数组类型
        for key, param in params_dict.items():
            params_dict[key] = param.value
        np.savez_compressed(file_path, **params_dict)

    def load_params(self, file_path="params.npz"):
        data = np.load(file_path, allow_pickle=True)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.value = data[key]

    def clear_grad(self):
        for param in self.params():
            param.clear_grad()


# 线性层
class LinearLayer(Layer):
    # input_size 可以不指定，在真正进行前向传播的时候延迟初始化
    def __init__(self, output_size, input_size=None, need_bias=True, dtype=np.float32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.need_bias = need_bias
        self.dtype = dtype
        self.W = Parameter(None, "W")
        if input_size is not None:
            self.__init_W()

        if need_bias:
            self.b = Parameter(np.zeros(output_size).astype(dtype), "b")
        else:
            self.b = None

    def __init_W(self):
        # 使用 Xavier 论文的初始化方式
        self.W.value = np.random.randn(self.input_size, self.output_size).astype(self.dtype) * np.sqrt(
            2.0 / (self.input_size + self.output_size))

    def forward(self, input_x):
        if self.W.value is None:
            if self.input_size is None:
                self.input_size = input_x.shape[1]
            self.__init_W()
        return linear(input_x, self.W, self.b)


# 模型，基类
class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)


# 两层线性网络模型
class TwoLayerNet(Model):
    def __init__(self, hidden_size, output_size, dtype=np.float32):
        super().__init__()
        self.l1 = LinearLayer(hidden_size, dtype=dtype)
        self.l2 = LinearLayer(output_size, dtype=dtype)

    def forward(self, x):
        temp = self.l1(x)
        temp = sigmoid(temp)
        result = self.l2(temp)
        return result


# 任意N层网络模型，即 MLP 模型
class MultiLayerNet(Model):
    # 初始化，hidden_size 是一个列表，每个元素是隐藏层的神经元数量
    def __init__(self, hidden_sizes, output_size, dtype=np.float32):
        super().__init__()
        self.layers = []
        for i in range(len(hidden_sizes)):
            self.layers.append(LinearLayer(hidden_sizes[i], dtype=dtype))
        # 最后一层的输出形状是 output_size
        self.layers.append(LinearLayer(output_size, dtype=dtype))

    def forward(self, x):
        # 最后一层不需要激活函数，所以循环到倒数第二层
        for layer in self.layers[:-1]:
            x = layer(x)
            x = sigmoid(x)
            # 最后一层是输出层，不需要激活函数
        return self.layers[-1](x)


#  ——————————————————————— end 参数,层,网络模型等高层概念  —————————————————————————


#  ———————————————————————    start 优化器    —————————————————————————
class Optimizer:
    def __init__(self, model):
        self.target = model  # 也可以传入 Layer 类
        self.hooks = []  # 钩子函数[可选]

    def add_hook(self, hook):
        self.hooks.append(hook)

    def update(self):
        params = self.target.params()
        # 过滤掉梯度为 None 的参数
        params = [p for p in params if p.grad is not None]

        # 调用钩子函数[可选]，可用于权重衰减、梯度裁剪等工作
        for hook in self.hooks:
            hook(params)

        # 逐个更新参数
        for param in params:
            self.update_one(param)

    # 每个参数的更新方法，需要在子类中实现
    def update_one(self, param):
        raise NotImplementedError()


# 梯度下降类
class SGD(Optimizer):
    def __init__(self, model, lr=0.01):
        super().__init__(model)
        self.lr = lr

    def update_one(self, param):
        param.value -= self.lr * param.grad.value


class Momentum(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.v = {}  # 保存每个参数的动量项

    def update_one(self, param):
        if param.grad is None:
            return

        grad = param.grad.value

        # 初始化动量
        if param not in self.v:
            self.v[param] = np.zeros_like(grad)

        v = self.v[param]

        # 计算动量更新
        v[:] = self.momentum * v - self.lr * grad

        # 参数更新
        param.value += v


#  ———————————————————————    end 优化器    —————————————————————————

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]  # 一般 x 的第一个维度是批量数据个数 batch size
    p = softmax_simple(x)
    p = clip(p, 1e-15, 1.0)  # 防止0和1溢出问题
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.value]
    return -1 * sum(tlog_p) / N


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        # 防止数据溢出，进行缩放
        x_shift = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(x_shift)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, dy):
        y = self.output_variable[0]
        dx = y * dy
        sum_dx = dx.sum(axis=self.axis, keepdims=True)
        dx -= y * sum_dx
        return dx


def softmax(x, axis=1):
    return Softmax(axis)(x)


# 使用另一种计算方式。是数学上最稳定的表达形式，避免任何溢出或 underflow。
def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, dy):
        # 反向传播： dL/dx = (y - one_hot(t)) / N
        # 拿到保存的值
        x, t = self.input_variable
        N, _ = x.shape

        dy *= 1 / N
        y = softmax(x)

        # 构造 one-hot
        one_hot = np.zeros_like(y, dtype=np.float32)
        one_hot[np.arange(N), t.value] = 1
        # softmax + crossentropy 的合成梯度
        y = (y - one_hot) * dy
        # 这个Node是指 t 无需梯度，因为t是标签，不是中间变量。如果没有None的话导致框架后续逻辑不兼容
        return y, None


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


#  ———————————————————————    start 数据集和加载器相关    —————————————————————————

class Dataset:
    def __init__(self, is_train=True, y_transform=None, t_transform=None):
        self.is_train = is_train
        self.y_transform = y_transform  # 样本预处理函数，可为 None
        self.t_transform = t_transform  # 标签预处理函数，可为 None

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        y = self.data[index]
        t = None if self.label is None else self.label[index]
        # 延迟 transform 判断（只有在存在时才调用）
        if self.y_transform:
            y = self.y_transform(y)
        if self.t_transform and t is not None:
            t = self.t_transform(t)
        return y, t

    def __len__(self):
        return len(self.data)

    def prepare(self):
        """由子类实现：生成或加载数据"""
        pass


def get_example_data(is_train=True):
    # 测试数据集和训练数据集的随机种子不同，用来模拟。
    np.random.seed(seed=(1 if is_train else 2))
    num_data, num_class, input_dim = 1000, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int32)  # 标签，每个类别对应一个整数

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle, 随机打乱数据
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]  # 用于输入数据集，每个类别对应一个整数
    t = t[indices]  # 标签数据集，每个类别对应一个整数

    # 附，可用于绘图
    # plt.figure(figsize=(6, 6))
    # # 不同类别使用不同颜色
    # for i in range(3):
    #     plt.scatter(x[t == i, 0], x[t == i, 1], label=f"Class {i}", s=20)
    # plt.legend()
    # plt.xlabel("x1")
    # plt.ylabel("x2")
    # plt.show()

    return x, t


class ThreeClassDataset(Dataset):
    def prepare(self):
        # 数据的缩放
        self.data, self.label = get_example_data(self.is_train)


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)
    # 预测值中概率最大的类别，构成与标签相同的形状
    pred = y.value.argmax(axis=1).reshape(t.shape)
    result = pred == t.value  # 预测值与标签值相等的位置为True，否则为False
    acc = result.mean()  # 计算准确率，即正确预测的样本数占总样本数的比例
    return Variable(as_array(acc))


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset  # 原始数据集
        self.iteration = 0  # 当前迭代次数
        self.index = None  # 当前批次的样本索引
        self.batch_size = batch_size  # 每个批次的样本数量
        self.shuffle = shuffle  # 是否在每个 epoch 开始时打乱数据索引
        self.data_size = len(dataset)
        # 每个 epoch 中，迭代的最大次数
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.reset()

    # 重置迭代器，将迭代次数设为0，根据shuffle参数是否为True，重新设置样本索引
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size: (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()


class MNISTDataset(Dataset):
    def prepare(self):
        if self.is_train:
            images = mnist.train_images()  # shape: (60000, 28, 28)
            labels = mnist.train_labels()  # shape: (60000,)
        else:
            images = mnist.test_images()  # shape: (10000, 28, 28)
            labels = mnist.test_labels()  # shape: (10000,)

        # 数据预处理
        # 转为 float32 并归一化到 [0, 1]
        images = images.astype(np.float32) / 255.0

        # 拉平成 (N, 784)，方便输入 MLP
        images = images.reshape(len(images), -1)

        self.data = images
        self.label = labels


#  ———————————————————————    end 数据集和加载器相关    —————————————————————————

if __name__ == '__main__':
    max_epoch = 30  # 最大训练轮数
    batch_size = 32  # 每个批次的样本数量
    hidden_size = 32  # 隐藏层神经元数量
    output_size = 10  # 手写数字识别的最终结果是10分类
    lr = 0.1  # 学习率

    # 使用 DataLoader 进行训练
    train_set = MNISTDataset(is_train=True)
    test_set = MNISTDataset(is_train=False)

    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MultiLayerNet((hidden_size, hidden_size), output_size)
    optimizer = Momentum(model, lr)

    # for 数据可视化
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []

    for epoch in range(max_epoch):
        sum_loss, sum_acc = 0, 0
        # 训练
        for x, t in train_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            model.clear_grad()
            loss.backward()
            optimizer.update()
            acc = accuracy(y, t)
            sum_loss += float(loss.value) * len(t)
            sum_acc += float(acc.value) * len(t)

        print("epoch: {}".format(epoch + 1))
        print("train loss: {:.4f}, accuracy: {:.4f}".format(sum_loss / len(train_set), sum_acc / len(train_set)))

        # 每一轮训练结束后，使用测试集评估模型性能
        sum_loss_test, sum_acc_test = 0, 0
        for x, t in test_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            sum_loss_test += float(loss.value) * len(t)
            sum_acc_test += float(acc.value) * len(t)

        print("test loss: {:.4f}, accuracy: {:.4f}".format(sum_loss_test / len(test_set), sum_acc_test / len(test_set)))

        # ======= 保存数据用于可视化 =======
        train_loss_list.append(sum_loss / len(train_set))
        test_loss_list.append(sum_loss_test / len(test_set))
        train_acc_list.append(sum_acc / len(train_set))
        test_acc_list.append(sum_acc_test / len(test_set))

    # ======= 绘图 =======
    epochs = range(1, max_epoch + 1)
    plt.figure(figsize=(12, 5))

    # --- loss 曲线 ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)

    # --- acc 曲线 ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 保存模型参数
    model.save_params("model_params.pkl")
