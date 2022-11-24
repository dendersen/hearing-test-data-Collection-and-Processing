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

float tonePlaying = 100;	

volatile int currentBufferIndex = 0;

volatile char test[] = {'1','2','3',';'};

int main(int argc, char const *argv[]){
	DDRB = 0x06;
	PORTB = 0x06;
	//initTonePlayer();
  	init();

  	while(1){
		if(tonePlaying != 0){
			playtone(tonePlaying);
		}
		
		if(currentBufferIndex != 0 && currentData[currentBufferIndex-1] == ';'){ // når vi er nået slutningen af beskeden
			int i;
			int newTone = 0;
			// for (i=0; i<currentBufferIndex-1;i++){
			// 	newTone += stoi(currentData[i]);
				
			// }
			tx_serial(currentData);
			currentBufferIndex=0;
		}
		
  	}
 	return 0;
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
	