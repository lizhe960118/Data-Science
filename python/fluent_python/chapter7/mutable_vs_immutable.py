if __name__ == "__main__":
    import doctest
    doctest.testmod()
    a = 1
    id1 = id(a)
    a += 1
    id2 = id(a)
    print(id1, id2)
    assert id1 != id2

    b = [1]
    id3 = id(b)
    b.append(2)
    id4 = id(b)
    print(id3, id4)
    assert id3 == id4
