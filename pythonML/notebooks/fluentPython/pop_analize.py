from timeit import Timer
pop_zero = Timer("x.pop(0)",
                 "from __main__ import x")
pop_end = Timer("x.pop()",
                "from __main__ import x")
x = list(range(2000000))
t1 = pop_zero.timeit(number=1000)
print(t1)
#4.8213560581207275
x = list(range(2000000))
t2 = pop_end.timeit(number=1000)
print(t2)
#0.0003161430358886719