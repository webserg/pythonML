import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import theano
import theano.tensor as T
from lasagne.updates import nesterov_momentum, rmsprop, adadelta, adagrad, adam

#For reproducibility. Comment it out for randomness
np.random.seed(413)

#Uncoomment and comment next line if you want to try random init
# clean_random_weights = scipy.random.standard_normal((2, 1))
clean_random_weights = np.asarray([[-2.8], [-2.5]])
W = theano.shared(clean_random_weights)
Wprobe = T.matrix('weights')

levels = [x/4.0 for x in range(-8, 2*12, 1)] + [6.25, 6.5, 6.75, 7] + \
         list(range(8, 20, 1))
levels = np.asarray(levels)

O_simple_quad = (W**2).sum()
O_wobbly = (W**2).sum()/3 + T.abs_(W[0][0])*T.sqrt(T.abs_(W[0][0]) + 0.1) + 3*T.sin(W.sum()) + 3.0 + 8*T.exp(-2*((W[0][0] + 1)**2+(W[1][0] + 2)**2))
O_basins_and_walls = (W**2).sum()/2 + T.sin(W[0][0]*4)**2
O_ripple = (W**2).sum()/3 + (T.sin(W[0][0]*20)**2 + T.sin(W[1][0]*20)**2)/15
O_giant_plateu = 4*(1-T.exp(-((W[0][0])**2+(W[1][0])**2)))
O_hills_and_canyon = (W**2).sum()/3 + \
                     3*T.exp(-((W[0][0] + 1)**2+(W[1][0] + 2)**2)) + \
                     T.exp(-1.5*(2*(W[0][0] + 2)**2+(W[1][0] -0.5)**2)) + \
                     3*T.exp(-1.5*((W[0][0] -1)**2+2*(W[1][0] + 1.5)**2)) + \
                     1.5*T.exp(-((W[0][0] + 1.5)**2+3*(W[1][0] + 0.5)**2)) + \
                     4*(1 - T.exp(-((W[0][0] + W[1][0])**2)))

O_two_minimums = 4-0.5*T.exp(-((W[0][0] + 2.5)**2+(W[1][0] + 2.5)**2))-3*T.exp(-((W[0][0])**2+(W[1][0])**2))

nesterov_testsuit = [
    (nesterov_momentum, "nesterov momentum 0.25",    {"learning_rate": 0.01, "momentum": 0.25}),
    (nesterov_momentum, "nesterov momentum 0.9",     {"learning_rate": 0.01, "momentum": 0.9}),
    (nesterov_momentum, "nesterov momentum 0.975",   {"learning_rate": 0.01, "momentum": 0.975})
]

cross_method_testsuit = [
    (nesterov_momentum, "nesterov",     {"learning_rate": 0.01}),
    (rmsprop,           "rmsprop",      {"learning_rate": 0.25}),
    (adadelta,          "adadelta",     {"learning_rate": 100.0}),
    (adagrad,           "adagrad",      {"learning_rate": 1.0}),
    (adam,              "adam",         {"learning_rate": 0.25})
]

for O, plot_label in [
    (O_wobbly, "Wobbly"),
    (O_basins_and_walls, "Basins_and_walls"),
    (O_giant_plateu, "Giant_plateu"),
    (O_hills_and_canyon, "Hills_and_canyon"),
    (O_two_minimums, "Bad_init")
]:

    result_probe = theano.function([Wprobe], O, givens=[(W, Wprobe)])

    history = {}
    for method, history_mark, kwargs_to_method in cross_method_testsuit:
        W.set_value(clean_random_weights)
        history[history_mark] = [W.eval().flatten()]

        updates = method(O, [W], **kwargs_to_method)
        train_fnc = theano.function(inputs=[], outputs=O, updates=updates)

        for i in range(125):
            result_val = train_fnc()
            print("Iteration " + str(i) + " result: "+ str(result_val))
            history[history_mark].append(W.eval().flatten())

        print("-------- DONE {}-------".format(history_mark))

    delta = 0.05
    mesh = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(mesh, mesh)

    Z = []
    for y in mesh:
        z = []
        for x in mesh:
            z.append(result_probe([[x], [y]]))
        Z.append(z)
    Z = np.asarray(Z)

    print("-------- BUILT MESH -------")

    fig, ax = plt.subplots(figsize=[12.0, 8.0])
    CS = ax.contour(X, Y, Z, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(plot_label)

    nphistory = []
    for key in history:
        nphistory.append(
            [np.asarray([h[0] for h in history[key]]),
             np.asarray([h[1] for h in history[key]]),
             key]
        )

    lines = []
    for nph in nphistory:
        lines += ax.plot(nph[0], nph[1], label=nph[2])
    leg = plt.legend()

    plt.savefig(plot_label + '_final.png')

    def animate(i):
        for line, hist in zip(lines, nphistory):
            line.set_xdata(hist[0][:i])
            line.set_ydata(hist[1][:i])
        return lines

    def init():
        for line, hist in zip(lines, nphistory):
            line.set_ydata(np.ma.array(hist[0], mask=True))
        return lines

    ani = animation.FuncAnimation(fig, animate, np.arange(1, 120), init_func=init,
                                  interval=100, repeat_delay=0, blit=True, repeat=True)

    print("-------- WRITING ANIMATION -------")

    # plt.show(block=True) #Uncoomment and comment next line if you just want to watch
    ani.save(plot_label + '.mp4', writer='ffmpeg_file', fps=5)

    print("-------- DONE {} -------".format(plot_label))