import struct
import socket
import time

def recv_timeout(the_socket,timeout=2):
    the_socket.setblocking(0)
    total_data=[];data='';begin=time.time()
    while 1:
        #if you got some data, then break after wait sec
        if total_data and time.time()-begin>timeout:
            break
        #if you got no data at all, wait a little longer
        elif time.time()-begin>timeout*2:
            break
        try:
            data=the_socket.recv(8192)
            if data:
                total_data.append(data)
                begin=time.time()
            else:
                time.sleep(0.1)
        except:
            pass
    the_socket.setblocking(1)
    return ''.join(total_data)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = ''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


class Decoder:
    def __init__(self, soc):
        self.soc = soc

    """
    Results
    """
    def getOkErr(self, n=1):
        return struct.unpack('>' + 'b'*n, self.soc.recv(n))

    def getScore(self):
        return struct.unpack('>' + 'i'*21, self.soc.recv(21*4))

    """
    Configuration
    """
    def configure(self):
        result = self.soc.recv(3*1)
        return struct.unpack('>bbb', result)

    """
    Query server
    """
    def takeScreenshot(self):
        dim = struct.unpack('>ii', self.soc.recv(2*4))
        # rem = dim[0]*dim[1]*3
        # temp = bytearray(rem)
        # while rem > 0:
        #     cnt = self.soc.recv_into(temp, rem)
        #     rem -= cnt
        # return (dim[0], dim[1], temp)
        # data = recv_timeout(self.soc)
        data = recvall(self.soc, dim[0] * dim[1] * 3)
        # print(dim[0], dim[1], len(data))
        return (dim[0], dim[1], data)
        # return (dim[0], dim[1], self.soc.recv(dim[0]*dim[1]*3, socket.MSG_WAITALL))

    def getState(self):
        return struct.unpack('>b', self.soc.recv(1))

    def getBestScores(self):
        return self.getScore()

    def getMyScores(self):
        return self.getscore()

    def getCurrLevel(self):
        return struct.unpack('>b', self.soc.recv(1))


    """
    Execute shot
    """
    def shot(self):
        return self.getOkErr()

    def shotSeq(self, n):
        return self.getOkErr(n)

    """
    Zoom, etc
    """
    def zoomOut(self):
        return self.getOkErr()

    def zoomIn(self):
        return self.getOkErr()

    def clickCenter(self):
        return self.getOkErr()

    """
    Select level
    """
    def loadLevel(self):
        return self.getOkErr()

    def restartLevel(self):
        return self.getOkErr()
