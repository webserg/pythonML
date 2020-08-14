def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper


##syntactic sugar
@my_decorator
def say_whee():
    print("Whee!")

say_whee()

##file will be close after block with
with open('./functions.py') as f:
    contents = f.read()
    print(contents)

class CustomOpen(object):
    def __init__(self, filename):
        self.file = open(filename)

    def __enter__(self):
        return self.file

    def __exit__(self, ctx_type, ctx_value, ctx_traceback):
        self.file.close()

with CustomOpen('./functions.py') as f:
    contents = f.read()


from contextlib import contextmanager

@contextmanager
def custom_open(filename):
    f = open(filename)
    try:
        yield f
    finally:
        f.close()

with custom_open('./functions.py') as f:
    contents = f.read()
