#include "HearingTest.c"
#include "USB_connection.c"
#include "stdio.h"



#ifndef F_CPU //hvis F_CPU  (hastighed for MCU, (konstant / her til brug i udregninger)
#define F_CPU 8000000UL  //så definer den 
#endif
#define BAUDRATEVAL 9600
#define BIT(x) (1 << (x))
#define SETBITS(x,y) ((x) |= (y))
#define CLEARBITS(x,y) ((x) &= (~(y)))
#define SETBIT(x,y) SETBITS((x), (BIT((y))))
#define CLEARBIT(x,y) CLEARBITS((x), (BIT((y))))
#define BITVAL(x,y) (((x)>>(y)) & 1)
#define HERTZ(x) ((CPU_CLOCK/400)/2)

float tonePlaying = 500;	

volatile int currentBufferIndex = 0;

int main(int argc, char const *argv[]){
	DDRB = 0x06;
	PORTB = 0x06;
  	init();

  	while(1){


		playtone(1<<10,10);
	
		if(currentBufferIndex != 0 && currentData[currentBufferIndex-1] == ';'){ // når vi er nået slutningen af beskeden
			tx_serial("dette er dit svar");
			tx_serial(";");
			//char numbers[2] = {'1','2'};
			//charListToNumber(numbers,2);
//			tx_serial(";");
			currentBufferIndex=0;
		}
		
  	}
 	return 0;
}

void charListToNumber(char number[], uint8_t lenghOfArray){
	// denne kode skal ændres så den oversætter den liste som der bliver givet som agument
	// isedet for at den oversætter currentBufferIndex
	uint8_t i;
	uint8_t fullNumber = 0;
	uint8_t newDigit = 0;
	for (i=lenghOfArray-1; i>-1 ;i--){
		newDigit = (uint8_t)(number[i]-48);
		uint8_t j = 0;
		for(j = 0; j < lenghOfArray-1-i; j++){
		 	newDigit = newDigit * 10;
		}
		fullNumber += newDigit;	
	}
	tx_serial("the new mesege is:");
	tx_serial_number(fullNumber);
	tx_serial(";");
}

void init(){
	DDRB = 0x06;
	init_seriel();
	sei(); //global interrupt enable, global disable is: cli();
}


ISR(USART_RX_vect){
	// UDR0 er dataen som bliver modtaget af microposesoren
	// den bliver gemt en char af gangen
	currentData[currentBufferIndex] = UDR0;
	currentBufferIndex++; 
}

void riteBuffer(){
	tx_serial_number(counter++);
	tx_serial("-Grettings from MCU I just received: ");
	tx_serial(currentData);
	tx_serial(" ;");
}


	