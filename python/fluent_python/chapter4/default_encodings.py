import sys,locale

expressions = """
    locale.getpreferredencoding()
    type(myfile)
    myfile.encoding
    sys.stdout.isatty()
    sys.stdout.encoding
    sys.stdin.isatty()
    sys.stdin.encoding    
    sys.stderr.isatty()
    sys.stderr.encoding
    sys.getdefaultencoding()
    sys.getfilesystemencoding()
"""

myfile = open("dummy", "w")

for expression in expressions.split():
    value = eval(expression)
    print(expression.rjust(30), '->', repr(value))
