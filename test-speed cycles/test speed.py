import timeit
import numpy
from ctypes import *
value = 100000000
lib=CDLL("./adder.so")

def loop1() -> value:
    num = 0
    result = 0
    while num < value:
        result +=1
        num += 1
    return result

def loop2() -> value:
    result = 0
    for num in range(value):
        result += num
    return result

def loop3() -> value:
    return sum(range(value))

def loop4() -> value:
    return numpy.sum(numpy.arange(value))

def loop5() -> value:
    lib.loop()

print(f"loop1: {timeit.timeit(loop1, number=1)}")
print(f"loop2: {timeit.timeit(loop2, number=1)}")
print(f"loop3: {timeit.timeit(loop3, number=1)}")
print(f"loop4: {timeit.timeit(loop4, number=1)}")
print(f"loop5: {timeit.timeit(loop5, number=1)}")