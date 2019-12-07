from bisect import bisect_left


def binary_search(a, x):
    i = bisect_left(a, x)
    exists = (i != len(a) and a[i] == x)
    if i:
        return i, exists
    else:
        return -1, exists
