# Websockets Server for ML shape detection on Tcp port 4439
# This will be started by systemd (linux) or launchctl (osx)
import cv2
import numpy as np
import imutils
import sys
import json
import argparse
import warnings
from datetime import datetime
import time,threading, sched

import logging
import logging.handlers
from logging.handlers import SysLogHandler
import ctypes
import socket
import asyncio
import websockets
import base64	

debug = False;

threshold = 0.4


class Settings:

  def __init__(self, logw):
    self.log = logw
    self.use_ml = None
    

async def wss_on_message(ws, path):
  global log, shapes_obj, cnn_face_obj, haar_obj, hog_obj, have_cuda, threshold
  #log.info(f'wake up {path}')
  message = await ws.recv()
  addr = ws.remote_address
  stm = datetime.now()
  imageBytes = base64.b64decode(message)
  nparr = np.frombuffer(imageBytes, np.uint8)
  #nparr = np.fromstring(imageBytes, np.uint8)  
  # nparr should be a jpg. Lets see
  # o = open("/tmp/shape.jpg","wb")
  # o.write(imageBytes)
  # o.close()
  frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  if path == '/Cnn_Face':
    if cnn_face_obj == None:
      cnn_face_obj = Algo('Cnn_Face', False, None, None, log, have_cuda)
    result, n = cnn_face_obj.face_detect(frame, threshold, False)
  elif path == '/Cnn_Shapes':
    if shapes_obj == None:
      shapes_obj = Algo('Cnn_Shapes', False, None, None, log, have_cuda)
    result, n = shapes_obj.shapes_detect(frame, threshold, False)
  elif path.startswith('/Haar'):
    if haar_obj == None:
      haar_obj = haar_obj.shapes_init(path[1:-1], log, have_cuda)
    result, n = haar_detect(frame, threshold, False)
  elif path == '/Hog_People':
    if hog_obj == None:
      hog_obj = Algo('Hog_People', False, None, None, log, have_cuda)
    result, n = hog_obj.hog_detect(frame, threshold, False)
  else:
    result = False
    
  etm = datetime.now()
  el = etm - stm
  et = el.total_seconds()
  if type(n) == int and n == 0:
    n = None
  dt = {'value': result, 'rect': n, 'time': et}
  
  log.info('%s %3.2f %s', addr[0], et, dt)
  await ws.send(json.dumps(dt))
     
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP
    
def wss_server_init(port):
  global wss_server, log
  wss_server = websockets.serve(wss_on_message, get_ip(), port)


def main():
  # process args - port number, 
  global log, wss_server, threshold, have_cuda
  ap = argparse.ArgumentParser()
  ap.add_argument("-p", "--port", action='store', type=int, default='4439',
    nargs='?', help="server port number, 4439 is default")
  ap.add_argument("-f", "--cf", action='store', type=float, default='0.4',
    nargs='?', help="confidence 0 .. 1, default 0.4")
  ap.add_argument("-s", "--syslog", action = 'store_true',
    default=False, help="use syslog")
 
  args = vars(ap.parse_args())
  
  # Note websockets is very chatty at DEBUG level. Too chatty to use. Sigh.
  log = logging.getLogger('mlshapes')
  if args['syslog']:
    log.setLevel(logging.INFO)
    handler = logging.handlers.SysLogHandler(address = '/dev/log')
    # formatter for syslog (no date/time or appname.
    formatter = logging.Formatter('%(name)s-%(levelname)-5s: %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
  else:
    logging.basicConfig(level=logging.INFO,datefmt="%H:%M:%S",format='%(asctime)s %(levelname)-5s %(message)s')
  
  settings = Settings(log)
  threshold = args['cf']
  #print('threshold', threshold)
  wss_server_init(args['port'])

  asyncio.get_event_loop().run_until_complete(wss_server)
  asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
  sys.exit(main())


