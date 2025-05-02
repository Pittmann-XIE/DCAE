## Decoding the received compressed data(z) and do the inference
import socket
import os

def receive_file(port):
    # 创建一个 TCP/IP 套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 绑定套接字到指定的端口
    server_address = ('', port)
    sock.bind(server_address)

    # 监听连接
    sock.listen(5)
    print(f'Listen at {port} ...')

    while True:
        # 等待客户端连接
        connection, client_address = sock.accept()
        try:
            print(f'connection from: {client_address}')

            # 接收文件名和大小
            header = connection.recv(4096).decode()
            if not header:
                print('No header received, closing connection.')
                continue
            try:
                file_name, file_size = header.split('|')
                file_size = int(file_size)
                print(f'receiving: {file_name}, size:{file_size} bytes')
            except ValueError:
                print("header is wrong, closing connection.")
                connection.close()
                continue

            # 发送确认信息
            connection.sendall('ACK'.encode())

            # 接收文件内容
            received_size = 0
            save_filename = os.path.join("./output/binary/bin", f'received_{os.path.basename(file_name)}')
            with open(save_filename, 'wb') as f:
                while received_size < file_size:
                    data = connection.recv(4096)
                    if not data:
                        break
                    f.write(data)
                    received_size += len(data)
                    print(f'received {received_size} / {file_size} bytes', end='\r')
            print(f'\n {file_name} done')

            if received_size == file_size:
                print(f'\n{save_filename} received successfully')
            else:
                print(f'\n{save_filename} received with error, expected {file_size} bytes but got {received_size} bytes')
        
        except Exception as e:
            print(f'Error occurred: {e}')
        
        finally:
            connection.close()

if __name__ == '__main__':
    # 设置服务器监听的端口号
    receive_file(8888)