import USB.USB_protocol as USB
import time as t

def main():
  pass

while 1:
  USB.sendMesege("050013")
  t.sleep(5)
  USB.sendMesege("031443")
  t.sleep(5)

