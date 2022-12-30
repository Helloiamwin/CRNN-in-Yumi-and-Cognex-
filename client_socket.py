from re import M
import socket
import time

HOST = "192.168.125.5"
SERVER_PORT = 65432
FORMAT = "utf8"



def sendList(client, list):

    for item in list:
        client.sendall(item.encode(FORMAT))
        #wait response
        client.recv(1024)

    msg = "end"
    client.send(msg.encode(FORMAT))

   
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("CLIENT SIDE")

try:
    client.connect( (HOST, SERVER_PORT) )
    print("client address:",client.getsockname())

    list = ["duchieuvn","15","nam"]
    Start_time = time.time()
    time.sleep(3)

    msg = None
    msg_recv = None
    while (msg != "x"):
        End_time = time.time()
        print (End_time -  Start_time)
        msg = input("talk: ")
        client.sendall(msg.encode(FORMAT))
        #time_end = time.time()
        if (msg == "thang"):
            msg_recv = client.recv(1024).decode(FORMAT)
            print("server has responded",msg_recv)
        if (msg == "list"):
            # wait response
            client.recv(1024)
            sendList(client, list)
    #print(time_end - time_start)
except:
    print("Error")

client.close()