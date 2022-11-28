import USB.USB_protocol as USB
import time 


def main():
  pass

tone = 100

while(1):
  USB.sendMesege("10027")
  time.sleep(2.4)

# while (1):
#   USB.sendMesege(str(tone))
#   time.sleep(1)
#   tone += 100

