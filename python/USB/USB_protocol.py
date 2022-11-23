#!/usr/bin/env python
import time
import serial
import codecs

ser = serial.Serial(
        port='COM5',
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1                
)

counter=0
rx=""
tx=""
ser.reset_input_buffer()
while 1:
    line =""    
    tx="p"
    counter+=1
    print("Til MCU - Tx nr.",counter,", data: " , tx,"\n")
    ser.write(tx.encode())
    time.sleep(0.5)
    while 1:
        rx=str(ser.read().decode())
        if rx == ";":
            break
        else:
            line=line+rx
    print("Fra MCU - data:",line,"\n")        
    ser.reset_input_buffer()

    
