# 
# test.py 
#   trys a known picture and an unknown picture
#   against the websocker. 
import sys
import websocket  # pip3 install websocket-client
import base64
import json
import argparse

def get_name(uri, bfr):
  ws = websocket.WebSocket()
  ws.connect(uri, timeout=10)
  ws.send(base64.b64encode(bfr))
  reply = ws.recv()
  ws.close()
  return reply
  
def main():
  # hardcoded for me and my machines and network.
  host = 'mini.local'
  #host = '192.168.1.45'  # bronco.local won't work! BUG somewhere 
  port = 4439
  uri = f'ws://{host}:{port}/Cnn_Shapes'
  ap = argparse.ArgumentParser()
  ap.add_argument("--host", action='store', type=str, default='192.168.1.2',
    nargs='?', help="host ip of mlshapes server")
  ap.add_argument("-p", "--port", action='store', type=int, default='4439',
    nargs='?', help="server port number, 4439 is default")
  args = vars(ap.parse_args())

  uri = f'ws://{args["host"]}:{args["port"]}/Cnn_Shapes'
  print('using', uri)
  fp = './test_image/person.jpg'
  f = open(fp, 'rb')
  img = f.read()
  f.close()
  rp = json.loads(get_name(uri, img))
  #rp = get_name(uri, img)
  onefr = float(rp['time'])
  fps = round(1 / onefr,2)
  print(f"Person? : {rp['value']} {rp['time']} secs {fps} fps")

  fp = './test_image/notperson.jpg'
  f = open(fp, 'rb')
  img = f.read()
  f.close()
  rp = json.loads(get_name(uri, img))
  #rp = get_name(uri, img)
  print(f"Person? : {rp['value']} {rp['time']} secs")


if __name__ == '__main__':
  sys.exit(main())
