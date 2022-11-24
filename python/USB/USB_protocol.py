import time
import serial
import codecs
from USB.integerSplit import numberTo16Bit,numberFrom16Bit


ser = serial.Serial(
        port='COM5',
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1                
)


counter=0
resevedText=""
toMCUText="1234"
endingChar=";"
ser.reset_input_buffer()


def sendMesege(number):
    text = numberTo16Bit(number)
    text += ";"
    line = ""
    print("Til MCU - data: " , text,"\n")

    for char in text:
        ser.write(char.encode("utf-8")) # send en char af gangen til MCU

    time.sleep(0.5)
    
    while 1:
        temp = ser.read()
        try:
            resevedText=str(temp.decode("utf-8",errors='strict')) # reseave text from MCU
        except:
            try:
                resevedText=str(temp.decode("ascii",errors="ignore"))
            except:
                resevedText = str(temp)
                pass
            pass
        print (resevedText)
        if resevedText == ";":
            break
        else:
            line += resevedText # adds the new text to the line

    if line + endingChar != text:
        print ("fail")

    print("Fra MCU - data:",line,"\n",numberFrom16Bit(line[0])) 
    ser.reset_input_buffer()
    
    
