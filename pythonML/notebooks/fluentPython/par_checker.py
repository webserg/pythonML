from fluentPython.Stack import Stack


def par_checker(par_string):
    if len(par_string) == 0:
        return True

    s = Stack()
    idx = 0
    while idx < len(par_string):
        if par_string[idx] == '(':
            s.push(par_string[idx])
        else:
            if s.is_empty():
                return False
            else:
                s.pop()
        idx += 1
    return s.is_empty()


print(par_checker(''))
print(par_checker('()'))
print(par_checker('('))
print(par_checker('((()))'))
print(par_checker('(()'))