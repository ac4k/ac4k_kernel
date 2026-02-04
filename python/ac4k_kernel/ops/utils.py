def ceil_div(a, b):
    return (a + b - 1) // b


def align_up(a, b):
    return ceil_div(a, b) * b
