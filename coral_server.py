# Websockets Server for ML shape detection on Tcp port 4439
# This will be started by systemd (linux) or launchctl (osx)
#import cv2
import numpy as np
from PIL import Image
#from pycoral.adapters import classify
from pycoral.adapters import detect
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import io
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
mean = 0.0
std = 0.0
size = 0
interpreter = None

class Settings:

  def __init__(self, logw):
    self.log = logw
    self.use_ml = None
    

async def wss_on_message(ws, path):
  global log, threshold, labels
  global interpreter, size, mean, std, top_k
  #log.info(f'wake up {path}')
  message = await ws.recv()
  addr = ws.remote_address
  stm = datetime.now()
  imageBytes = base64.b64decode(message)
  nparr = np.frombuffer(imageBytes, np.uint8)
  #nparr = np.fromstring(imageBytes, np.uint8)  
  # nparr should be a jpg. Lets see
  #o = open("/tmp/shape.jpg","wb")
  #o.write(imageBytes)
  #o.close()
  
  #image = Image.open(io.BytesIO(nparr)).resize(size, Image.ANTIALIAS)
  image = Image.open(io.BytesIO(nparr))
  _, scale = common.set_resized_input(
      interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

  # Run inference
  interpreter.invoke()
  objs = detect.get_objects(interpreter, threshold, scale)
  
  etm = datetime.now()
  el = etm - stm
  et = el.total_seconds()
  result = False
  n = None
  if objs:
    for obj in objs:
      objname = labels.get(obj.id, obj.id)
      if objname == "person":
        result = True
        n = (obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax)
        break

  dt = {'value': result, 'rect': n, 'time': et}
  
  log.info('%s %3.2f %s', objname, et, dt)
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
  global log, wss_server, threshold, labels
  global interpreter, size, mean, std, top_k
  ap = argparse.ArgumentParser()
  
  ap.add_argument(
      '-m', '--model', required=False,
      default="coral/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
      help='File path of .tflite file.')
  ap.add_argument(
      '-l', '--labels', default="coral/coco_labels.txt",
      help='File path of labels file.')
  ap.add_argument(
      '-t', '--threshold', type=float, default=0.4,
      help='Classification score threshold')
  ap.add_argument(
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  ap.add_argument("-p", "--port", action='store', type=int, default='4439',
    nargs='?', help="server port number, 4439 is default")
  ap.add_argument("-s", "--syslog", action = 'store_true',
    default=False, help="use syslog")
    
  args = ap.parse_args()
  # Populate the globals for use when the message arrives
  threshold = args.threshold

  labels = read_label_file(args.labels) if args.labels else {}

  interpreter = make_interpreter(*args.model.split('@'))
  interpreter.allocate_tensors()

  # Model must be uint8 quantized
  if common.input_details(interpreter, 'dtype') != np.uint8:
    raise ValueError('Only support uint8 input type.')

  size = common.input_size(interpreter)
 
  
  # Note websockets is very chatty at DEBUG level. Too chatty to use. Sigh.
  log = logging.getLogger('coralshapes')
  if args.syslog:
    log.setLevel(logging.INFO)
    handler = logging.handlers.SysLogHandler(address = '/dev/log')
    # formatter for syslog (no date/time or appname.
    formatter = logging.Formatter('%(name)s-%(levelname)-5s: %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
  else:
    logging.basicConfig(level=logging.INFO,datefmt="%H:%M:%S",format='%(asctime)s %(levelname)-5s %(message)s')
  
  settings = Settings(log)
  #threshold = args['cf']
  #print('threshold', threshold)
  wsp = args.port
  wss_server_init(wsp)

  asyncio.get_event_loop().run_until_complete(wss_server)
  asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
  sys.exit(main())


