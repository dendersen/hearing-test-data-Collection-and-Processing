import time
import serial
import codecs
#from USB.integerSplit import numberTo16Bit,numberFrom16Bit


ser = serial.Serial( 
        port='COM5', #COM5
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


def sendMesege(text):
    text += ";"
    line = ""
    print("Til MCU - data: " , text,"\n")

    for char in text:
        ser.write(char.encode()) # send en char af gangen til MCU

    time.sleep(0.5)
    
    while 1:
        resevedText=str(ser.read().decode("utf-8",errors='replace')) # reseave text from MCU

        if resevedText == ";":
            break
        else:
            line += resevedText # adds the new text to the line

    print("Fra MCU - data:",line,"\n") 
    ser.reset_input_buffer()
    
    
