import asyncio

STORAGE = {}

class ClientServerProtocol(asyncio.Protocol):    
    def __init__(self):
        self.transport = None
    
    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data):
        try:
            resp = self.process_data(data.decode())
        except:
            resp = 'error\nwrong command\n\n'
        self.transport.write(resp.encode())
    
    def process_data(self, data):
        print('DATA_IN:', data)
        data = data.split('\n')[0]
        print(data)
        data = data.split(' ')
        cmd = data[0]
        try:
            if cmd == 'put':
                key = data[1]
                value = float(data[2])
                timestamp = int(data[3])
                if key not in STORAGE.keys():
                    STORAGE[key] = []
                for i in range(len(STORAGE[key])):
                    if timestamp == STORAGE[key][i][1]:
                        STORAGE[key].pop(i)
                STORAGE[key].append((value, timestamp))
                answer = 'ok\n\n'
            elif cmd == 'get':
                if len(data) > 2:
                    raise Exception
                if data[1] == '*':
                    answer = 'ok\n'
                    for key in STORAGE.keys():
                        for l in STORAGE[key]:
                            answer += key + ' ' + str(l[0]) + ' ' + str(l[1]) + '\n'
                    answer += '\n'
                else:
                    key = data[1]
                    if key not in STORAGE.keys():
                        answer = 'ok\n\n'
                    else:
                        answer = 'ok\n'
                        for l in STORAGE[key]:
                            answer += key + ' ' + str(l[0]) + ' ' + str(l[1]) + '\n'
                        answer += '\n'
            else:
                raise Exception
        except:
            raise Exception
        return answer


def run_server(host, port):
    loop = asyncio.get_event_loop()
    coro = loop.create_server(
        ClientServerProtocol,
        host, port
    )

    server = loop.run_until_complete(coro)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()

if __name__ == "__main__":
    # запуск сервера для тестирования
    run_server('127.0.0.1', 8888)