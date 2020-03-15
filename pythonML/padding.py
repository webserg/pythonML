import tensorflow as tf

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong

x = tf.constant([[9., 8., 1.],
                 [4., 5., 7.]])
x2 = tf.reshape(x, [1, 2, 3, 1])  # give a shape accepted by tf.nn.max_pool
valid_pad = tf.nn.max_pool(x2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
same_pad = tf.nn.max_pool(x2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#The output shapes are:
#valid_pad: here, no padding so the output shape is [1, 1]
#same_pad: here, we pad the image to the shape [2, 4] (with -inf and then apply max pool), so the output shape is [1, 2]
#

with sess.as_default():
    print(x.eval())
    print(x2.eval())
    print("================")
    print(valid_pad.eval())
    print("================")
    print(same_pad.eval())

valid_pad.get_shape() == [1, 1, 1, 1]  # valid_pad is [5.]
same_pad.get_shape() == [1, 1, 2, 1]   # same_pad is  [5., 6.]