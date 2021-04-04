# unimodal test function
import numpy as np
from matplotlib import pyplot
from numpy import cos
from numpy import e
from numpy import exp
from numpy import pi
from numpy import sqrt


# Унимодальность означает, что функция имеет единственный глобальный оптимум.
# Унимодальная функция может быть выпуклой и невыпуклой. Выпуклая функция — это функция,
# на агрфике которой между двумя любыми точками можно провести линию и эта линия останется в домене.
# В случае двумерной функции это означает, что поверхность имеет форму чаши, а линии между двумя точками
# проходят по чаше или внутри неё. Давайте рассмотрим несколько примеров унимодальных функций.

def unimodal():
    def objective(x, y):
        return x ** 2.0 + y ** 2.0

    # define range for input
    r_min, r_max = -5.0, 5.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    pyplot.show()


def unimodal2():
    # objective function
    def objective(x, y):
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    # define range for input
    r_min, r_max = -10.0, 10.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    pyplot.show()


def unimodal3():
    # objective function
    def objective(x, y):
        return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))

    # define range for input
    r_min, r_max = -10, 10
    # sample input range uniformly at 0.01 increments
    xaxis = np.arange(r_min, r_max, 0.01)
    yaxis = np.arange(r_min, r_max, 0.01)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    pyplot.show()


# Мультимодальная функция — это функция с более чем одной “модой” или оптимумом (например долиной на графике).
# Мультимодальные функции являются невыпуклыми. Могут иметь место один или несколько ложных оптимумов.
# С другой стороны, может существовать и несколько глобальных оптимумов, например несколько различных
# значений аргументов функции, при которых она достигает минимума.


# Диапазон ограничен -5,0 и 5,0 и одним глобальным оптимумом при [0,0, 0,0]. Эта функция известна как функция Экли.

def multimodal1():
    # objective function
    def objective(x, y):
        return -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(
            0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

    # define range for input
    r_min, r_max = -5.0, 5.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    pyplot.show()


# Диапазон ограничен [-5,0 и 5,0], а функция имеет четыре глобальных оптимума при
# [3,0, 2,0], [-2,805118, 3,131312], [-3,779310, -3,283186], [3,584428, -1,848126].
# Эта функция известна как функция Химмельблау.

def multimodal2():
    # objective function
    def objective(x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    # define range for input
    r_min, r_max = -5.0, 5.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    pyplot.show()


# Диапазон ограничен промежутком [-10,0 и 10,0] и функцией с четырьмя глобальными оптимумами в
# точках [8,05502, 9,66459], [-8,05502, 9,66459], [8,05502, -9,66459], [-8,05502, -9,66459].
# Эта функция известна как табличная функция Хольдера.

def multimodal3():
    # objective function
    def objective(x, y):
        return -np.absolute(np.sin(x) * cos(y) * exp(np.absolute(1 - (sqrt(x**2 + y**2)/pi))))

    # define range for input
    r_min, r_max = -10.0, 10.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    pyplot.show()

# https://en.wikipedia.org/wiki/Test_functions_for_optimization


if __name__ == '__main__':
    unimodal()
    unimodal2()
    unimodal3()
    multimodal1()
    multimodal2()
    multimodal3()
