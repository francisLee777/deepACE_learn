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
        sum_to(input_dy, self.original_shape)


def broadcast_to(input_x, target_shape):
    if input_x.shape == target_shape:
        return as_variable(input_x)
    return BroadcastTo(target_shape)(as_variable(as_array(input_x)))


class SumTo(Function):
    def __init__(self, target_shape):
        self.original_shape = None
        self.target_shape = target_shape

    def forward(self, input_x):
        self.original_shape = input_x.shape
        return util_sum_to(input_x, self.target_shape)

    def backward(self, input_dy: Variable):
        broadcast_to(input_dy, self.original_shape)


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


#  ———————————————————————— end 基础的深度学习网络组件  ——————————————————————————————


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


def temp_fun(x, y):
    return pow(x + 1, 2) * neg(y) - abs(x - y) + y ** 2


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


def sigmoid_simple(x):
    y = 1 / (1 + exp(-x))
    return y


def numerical_gradient_matrix(func, input_x, eps=1e-4):
    x_ndarray = input_x.value
    rows = x_ndarray.shape[0]
    cols = x_ndarray.shape[1]

    grad = np.zeros_like(x_ndarray)
    for i in range(rows):
        for j in range(cols):
            original_value = x_ndarray[i, j]

            x_ndarray[i, j] = original_value + eps
            v_plus = Variable(x_ndarray.copy())
            y0 = func(v_plus)

            x_ndarray[i, j] = original_value - eps
            v_minus = Variable(x_ndarray.copy())
            y1 = func(v_minus)

            grad[i, j] = (y0.value - y1.value) / (2 * eps)
            x_ndarray[i, j] = original_value
    return grad


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


def tempFunc(x, y):
    return sum(x @ y)


if __name__ == '__main__':
    x = Variable(np.array([[1, 2]], dtype=np.float64))  # (1, 2)
    W = Variable(np.array([[5, 6], [7, 8]], dtype=np.float64))  # (2, 2)

    result = tempFunc(x, W)
    result.backward()

    print("x.grad结果:", x.grad, "W.grad结果:", W.grad)  # 能够正确输出梯度

    print("数值微分法得到的 x.grad ", numerical_gradient_matrix_x(tempFunc, x, W))
    print("数值微分法得到的 W.grad ", numerical_gradient_matrix_w(tempFunc, x, W))

    # 学习率
    # lr = 0.1
    # iters = 100  # 循环次数
    #
    # x = np.random.rand(100, 1)  # 形状 (100, 1) 每个元素都是 [0, 1) 浮点数
    # # print(x.shape)
    # # print(x[2][0])
    # y = 25 * x + 38 + np.random.rand(100, 1)  # 形状 (100, 1)
    #
    # # 权重
    # b = Variable(np.zeros(1))  # 初始化为 0     形状: (1, )
    # W = Variable(np.zeros((1, 1)))  # 初始化为 0  形状 (1, 1)
    #
    #
    # # 把输入数据和权重参数输入到深度学习网络中，得到预测值
    # def predict(x):  # 输出 Variable
    #     return matmul(x, W) + b
    #
    #
    # def mean_square_error(y0, y1):  # 输出 Variable
    #     diff = y1 - y0
    #     return sum(diff ** 2) / len(diff)
    #
    #
    # # 训练
    # for i in range(iters):
    #     # print("W:", W.value)
    #     # print("b:", b.value)
    #
    #     y_predit = predict(x)  # 输出 Variable
    #     loss = mean_square_error(y, y_predit)  # 计算误差，得到 loss 损失值 (Variable类型)
    #
    #     loss.backward()  # 损失函数的反向传播
    #
    #     W.value -= lr * W.grad.value  # 根据梯度值，更新各个权重参数 [有各种方式的方式]
    #     b.value -= lr * b.grad.value
    #
    #     W.grad = None  # 每次迭代之后，需要把梯度重置为0，否则会影响下一次迭代计算梯度
    #     b.grad = None
