import numpy as np


# 1. Function 基类缺少 forward 方法
# 2. 很好，测试代码完整，规范
# 3. 作业3回答正确

class Variable:
    def __init__(self, input_data):
        self.value = input_data
        self.grad = None  # 默认为  None


class Function:
    #  __call__  是一个特殊方法，定义后,  能够  f  =  Function()  后直接调用  f(...)
    #  输入和输出是  Variable  类型
    def __call__(self, input_variable: Variable):
        x = input_variable.value  # 从变量对象中取出实际值
        y = self.forward(x)  # 具体地计算在  forward  方法中，所有子类必须实现这个方法
        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        #  每个Function子类（Square/Exp）的__call__方法中，
        #  都会执行self.input_variable  =  input_variable——
        #  把当前输入的Variable（x/a/b  这些中间结果）绑定到函数实例上，为反向传播  “留痕”。
        output_variable = Variable(y)  # 将计算结果封装成变量对象并返回
        return output_variable

    #  新增反向传播函数。输入和输出是  非Variable  类型
    #  1.  自动微分是什么？
    #  自动微分是一种通用的求导技术，核心是  “将复杂函数的求导拆解为基本运算（+、-、×、÷、平方、指数等）的求导，
    #  再通过链式法则拼接”，它介于  “数值微分（近似、慢）”  和  “符号微分（精确、但复杂函数易爆炸）”  之间，
    #  是深度学习框架（PyTorch/TensorFlow）求梯度的核心。
    #  自动微分分为两种模式：
    #  前向模式：按  “输入→输出”  的顺序求导，适合  “输入少、输出多”  的场景
    #  （比如函数  f  (x1,x2)=[x1²,  x2³]，求每个输出对输入的导数）；
    #  反向模式：按  “输出→输入”  的顺序求导，适合  “输入多、输出少”  的场景
    #  （比如神经网络：输入是海量权重  /  偏置，输出是  1  个损失值，求损失对所有权重的梯度）。
    #  2.  反向传播和自动微分的关系
    #  反向传播  =  自动微分的反向模式  +  神经网络的参数优化逻辑；
    #  简单说：自动微分是  “求梯度的通用方法”，反向传播是  “把这种方法用在神经网络上，计算损失函数对权重  /  偏置的梯度，并更新参数”。
    def backward(self, input_dy):  # 先考虑只有一个输入变量
        raise NotImplementedError()  # 子类的实现可以默认为  1


#  求平方函数，实现了  Function  类
class Square(Function):
    def forward(self, input_x):
        return input_x ** 2

    def backward(self, input_dy):
        return 2 * self.input_variable.value * input_dy  # 通过函数实例保存的input_variable获取中间值


#  Exp  函数，实现了  Function  类
class Exp(Function):
    def forward(self, input_x):
        return np.exp(input_x)

    def backward(self, input_dy):
        return input_dy * np.exp(self.input_variable.value)


#  Sin  函数，实现了  Function  类
class Sin(Function):
    def forward(self, input_x):
        return np.sin(input_x)

    def backward(self, input_dy):
        return input_dy * np.cos(self.input_variable.value)


#  Cos  函数，实现了  Function  类
class Cos(Function):
    def forward(self, input_x):
        return np.cos(input_x)

    def backward(self, input_dy):
        return -input_dy * np.sin(self.input_variable.value)


#  Abs  函数，实现了  Function  类
class Abs(Function):
    def forward(self, input_x):
        return np.abs(input_x)

    def backward(self, input_dy):
        return input_dy * np.sign(self.input_variable.value)


#  Neg  函数，实现了  Function  类
class Neg(Function):
    def forward(self, input_x):
        return -input_x

    def backward(self, input_dy):
        return -input_dy


#  Pow  函数，实现了  Function  类
class Pow(Function):
    def __init__(self, power):
        self.power = power

    def forward(self, input_x):
        return input_x ** self.power

    def backward(self, input_dy):
        return input_dy * self.power * (self.input_variable.value ** (self.power - 1))


#  CompositeFunction  类，继承  Function  类，封装  A、B、C  三个函数
class CompositeFunction(Function):
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    def forward(self, input_x):
        #  前向传播：x  ->  A(x)  ->  B(A(x))  ->  C(B(A(x)))
        #  注意：需要通过__call__方法调用，而不是直接调用forward，这样才能保存input_variable
        input_variable = Variable(input_x)  # 保存输入变量
        a = self.A(input_variable)
        b = self.B(a)
        c = self.C(b)
        return c.value  # 返回实际值，因为forward方法应该返回非Variable类型

    def backward(self, input_dy):
        #  反向传播：dy  ->  C.backward(dy)  ->  B.backward(C.backward(dy))  ->  A.backward(B.backward(C.backward(dy)))
        db = self.C.backward(input_dy)
        da = self.B.backward(db)
        dx = self.A.backward(da)
        return dx


#  数值微分函数（也称梯度检验），用于验证反向传播的正确性
#  取x-h,x+h而不使用x,x+h：中心差分减小误差；eps不取1e-50：float有效精度15-17位不够，可能导致舍入误差
#  数值微分存在的两个问题：(面试题)
#  1.  精度丢失
#  虽然上面的输出结果看起来误差非常小，但如果正好处在函数的导数值比较小的位置，加上一般计算时保留4位小数，
#  则会导致有效位数可能只剩1位了。例如  0.2222  -  0.2221  =  0.0001
#  但有可能真实情况是  0.22229  -  0.22211  =  0.00018  可以看到误差了近  50%
#  2.  计算复杂度过高
#  上面的样例中，input_var  只输入了一个标量值，但在真实的深度学习中，输入变量的维度可能是百万级别的，
#  对于复合函数来说，需要独立的计算每个子函数上百万次，没能利用每个子函数的中间结果。这样的计算量是无法被接受的。
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.value - eps)
    x1 = Variable(x.value + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.value - y0.value) / (2 * eps)


if __name__ == '__main__':
    #  测试各个函数
    print("测试各个函数的前向和反向传播：")
    x = Variable(-1.0)

    #  测试Sin函数
    sin_func = Sin()
    sin_y = sin_func(x)
    sin_y.grad = 1.0
    sin_dx = sin_func.backward(1.0)
    print(f"Sin(-1)  =  {sin_y.value},  梯度  =  {sin_dx}")

    #  测试Cos函数
    cos_func = Cos()
    cos_y = cos_func(x)
    cos_dx = cos_func.backward(1.0)
    print(f"Cos(-1)  =  {cos_y.value},  梯度  =  {cos_dx}")

    #  测试Abs函数
    abs_func = Abs()
    abs_y = abs_func(x)
    abs_dx = abs_func.backward(1.0)
    print(f"Abs(-1)  =  {abs_y.value},  梯度  =  {abs_dx}")

    #  测试Neg函数
    neg_func = Neg()
    neg_y = neg_func(x)
    neg_dx = neg_func.backward(1.0)
    print(f"Neg(-1)  =  {neg_y.value},  梯度  =  {neg_dx}")

    #  测试Pow函数
    pow_func = Pow(3)
    pow_y = pow_func(x)
    pow_dx = pow_func.backward(1.0)
    print(f"Pow(-1,  3)  =  {pow_y.value},  梯度  =  {pow_dx}")

    print("\n测试CompositeFunction类：")
    #  定义A、B、C三个函数
    A = Square()
    B = Exp()
    C = Square()

    #  创建CompositeFunction实例
    composite_func = CompositeFunction(A, B, C)

    #  前向传播
    x = Variable(-1)
    y = composite_func(x)
    print(f"最终结果:  {y.value}")

    #  反向传播
    dy = 1.0
    dx = composite_func.backward(dy)
    print(f"y关于x的梯度结果为:  {dx}")


    #  使用数值微分验证
    def predict_func(x_):
        return composite_func(x_)


    print(f"y关于x的数值微分结果为:  {numerical_diff(predict_func, x)}")

# 3.
# （1）反向传播：返回多梯度
# 当前backward仅返回单个梯度，而二元运算需要返回多个梯度
# 调整backward的返回值：从返回单个数值改为返回元组 / 列表，长度与输入变量数量一致
# （2）数值微分：梯度检验
# 当前numerical_diff仅支持单变量，扩展到多元函数后需要实现多变量数值微分
# 对每个输入变量单独计算偏导数（固定其他变量，仅扰动当前变量）；
# 例如对f(x, y)，计算df/dx时固定y，仅扰动x；计算df/dy时固定x，仅扰动y；
# 数值微分的结果需与反向传播返回的多梯度元组一一对应，用于验证二元函数反向传播的正确性。
