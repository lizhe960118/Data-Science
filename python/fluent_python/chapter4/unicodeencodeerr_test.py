city = 'São Paulo'
print(city.encode('utf-8'))
print(city.encode('utf-16'))
print(city.encode('iso8859_1'))
print(city.encode('cp437', errors='ignore'))
print(city.encode('cp437',errors='replace'))
print(city.encode('cp437',errors='xmlcharrefreplace'))
