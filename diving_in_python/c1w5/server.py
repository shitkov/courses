import socket

sock = socket.socket()
sock.bind(('127.0.0.1', 7777))
sock.listen(1)
conn, addr = sock.accept()

while True:
    data = conn.recv(1024)
    if not data:
        pass
    else:
        # OK:
        # data = 'ok\n\n'
        # data = 'error\nwrong command\n\n'
        # data = 'ok\npalm.cpu 2.0 1150864248\npalm.cpu 0.5 1150864247\n\n'
        # data = 'ok\npalm.cpu1 2.0 1150864248\npalm.cpu2 0.5 1150864247\n\n'
        # data = 'ok\nwrong data\n\n'
        # data = 'some_text\nok\n\n'
        # data = 'ok\npalm.cpu 2.0 1150864248s\npalm.cpu 0.5s 1150864247\n\n'
        # TEST:
        # data = 'ok\npalm.cpu 2.0 1150864247\npalm.cpu 1 1150864248\neardrum.cpu 3.0 1150864250\n\n'
        # data = 'some_text\nok\n\n'
        data = 'q\n\n'
        conn.send(data.encode('utf-8'))

conn.close()
