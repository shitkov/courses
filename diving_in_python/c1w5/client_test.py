from client import Client

addr = '127.0.0.1'
port = 7777

cl = Client(addr, port, 2)

# cl.put('test.key', 28)

print(cl.get('*'))
