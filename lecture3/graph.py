import os
import subprocess

import numpy as np

from lecture3.core import Variable, add, square


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.input_data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.input_data) + ' '

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
        ret += dot_edge.format(id(f), id(y))
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


def goldstein(x, y):
    z = square(add(x, y))
    return z


if __name__ == '__main__':
    x = Variable(np.array(1.0), name="x")
    y = Variable(np.array(1.0), name="y")
    z = goldstein(x, y)
    z.name = 'z'
    z.backward()
    print(z.grad)
    plot_dot_graph(z, verbose=True, to_file='goldstein.png')
