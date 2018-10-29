registry = set()

def register(active=True):
    def decorate(func):
        print("Running register(active=%s -> decorate(%s))" %(active, func))
        if active:
            registry.add(func)
        else:
            registry.discard(func)
        return func
    return decorate

@register(active=False)
def f1():
    print('Running f1()')

@register()
def f2():
    print('Running f2()')

@register()
def f3():
    print('Running f3()')

def main():
    print('Running main()')
    print('registry ->', registry)
    f1()
    f2()
    f3()
    print(registry.pop().__name__)

if __name__ == '__main__':
    main()
    import doctest
    doctest.testmod()
