import collections
ct = collections.Counter('abrcaadabra')
print(ct)
ct.update('aaaaazzz')
print(ct)
print(ct.most_common(3))
