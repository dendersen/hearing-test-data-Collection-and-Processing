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
resevedText=""
toMCUText="1234"
endingChar=";"
ser.reset_input_buffer()

# while 1:
#     line =""
#     toMCUText += endingChar
#     counter+=1
    
#     print("Til MCU - Tx nr.",counter,", data: " , toMCUText, endingChar,"\n")
    
#     for char in toMCUText:
#         print("vi sender: ", char)
#         ser.write(toMCUText.encode()) # send en char af gangen til MCU
#     #print("vi sender:", endingChar)
#     #ser.write(endingChar.encode())
    
    
#     time.sleep(0.5)
    
#     while 1:
#         resevedText=str(ser.read().decode()) # reseave text from MCU
#         print("ny char: ",resevedText)
        
#         if resevedText == ";":
#             break
#         else:
#             line += resevedText # adds the new text to the line
    
#     if line + endingChar != toMCUText:
#         print ("fail")
    
#     print("Fra MCU - data:",line,"\n") 
#     ser.reset_input_buffer()

def sendMesege(text):
    text += ";"
    
    while 1:
        line = ""
        counter+=1

        print("Til MCU - data: " , text,"\n")

        for char in text:
            print("vi sender: ", char)
            ser.write(text.encode()) # send en char af gangen til MCU
        #print("vi sender:", endingChar)
        #ser.write(endingChar.encode())


        time.sleep(0.5)

        while 1:
            resevedText=str(ser.read().decode()) # reseave text from MCU
            print("ny char: ",resevedText)

            if resevedText == ";":
                break
            else:
                line += resevedText # adds the new text to the line

        if line + endingChar != text:
            print ("fail")

        print("Fra MCU - data:",line,"\n") 
        ser.reset_input_buffer()
    
    
