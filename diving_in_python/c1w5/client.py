import socket
import time


class ClientError(Exception):
    pass


class Client:
    def __init__(self, addr, port, timeout=None):
        self._addr = addr
        self._port = int(port)
        self._timeout = int(timeout)

    def send(self, cmd):
        with socket.create_connection(
                (self._addr, self._port), self._timeout) as sock:
            sock.sendall(cmd.encode("utf8"))
            buf = sock.recv(1024)
            return buf.decode('utf-8')

    def get(self, key):
        if key is None:
            raise ClientError
        resp = self.send('get ' + key + '\n')
        ans = resp.split('\n')
        dct = dict()
        if resp == 'ok\n\n':
            return dct

        if resp[0:3] != 'ok\n':
            raise ClientError(resp)
        elif ans[-1] != '':
            raise ClientError(resp)
        elif ans[-2] != '':
            raise ClientError(resp)
        if len(ans) < 4:
            raise ClientError(resp)
        else:
            ans = ans[1:-2]
            ans = [l.split(' ') for l in ans]
            for rec in ans:
                if len(rec) != 3:
                    raise ClientError
                elif not rec[2].isdigit():
                    raise ClientError
                elif (len(rec[1].split('.')) != 2) and (not rec[1].isdigit()):
                    raise ClientError
                elif (len(rec[1].split('.')) == 2) and not(
                        (rec[1].split('.')[0].isdigit())
                        and (rec[1].split('.')[1].isdigit())):
                    raise ClientError
                else:
                    if rec[0] in dct.keys():
                        dct[rec[0]].append((int(rec[2]),
                                            float(rec[1])))
                    else:
                        dct[rec[0]] = []
                        dct[rec[0]].append((int(rec[2]),
                                            float(rec[1])))
                    keys = list(dct.keys())
                    dct_sort = {}
                    for k in keys:
                        dct_sort[k] = dct[k].sort(key=lambda i: i[0])
                    dct = dct_sort
        return dct

    def put(self, key, val, timestamp=None):
        resp = self.send(
                'put ' + key + ' ' + str(val) + ' '
                + str(timestamp if timestamp else int(time.time())) + '\n')
        if resp[0:3] != 'ok\n':
            raise ClientError(resp)
