from pysat.examples.hitman import Hitman


def MHS(sets):
    """
    minimal hitting set problem 求解，这里用现成的m22
    :param sets: 输入的集合
    """
    h = Hitman(solver="m22", htype="lbx")
    # adding sets to hit
    for s in sets:
        h.hit(s)

    return h


if __name__ == "__main__":
    sets = [[1, 2, 3, 5], [3, 4, 5], [3, 5, 6, 7]]
    h = MHS(sets)
    print(h.get())
