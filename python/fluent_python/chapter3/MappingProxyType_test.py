from types import MappingProxyType

d = {1:'A'}

d_proxy = MappingProxyType(d)
#封装类
print(d_proxy)
print(d_proxy[1])

#d_proxy[2] = 'x'
#用户不能修改封装类，只能通过修改源数据来修改

d[2] = "B"
print(d_proxy)
print(d_proxy[2])
