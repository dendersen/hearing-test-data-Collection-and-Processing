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

volatile int currentBufferIndex = 0;
volatile uint8_t playingSound = 0;

int charListToNumber(char number[], int lenghOfArray);

int main(int argc, char const *argv[]){
	DDRB = 0x06;// bit 1 og 2 er output til høretelefonerne
	DDRC = 0b1000; // bit 0 og 1 er input fra knapperne


  init();

	uint16_t firstDigit = 15;
	uint8_t secondDigit = 60;
	uint8_t chosenEar = 0;

  while(1){

	playtone(1<<firstDigit,secondDigit);

	if(playingSound && bit_is_set(PINC,PC0) || bit_is_set(PINC,PC1) || bit_is_set(PINC,PC2)){
		// hvis dette kode køre, er der trykket på en knap og lyd bliver spillet
		uint8_t rightIsPresed = 0;
		uint8_t leftIsPresed = 0;
		uint8_t nonIsPresed = 0;
		uint16_t i;
		for(i = 0; i < 500; i++){
			if(bit_is_set(PINC,PC0)){
				rightIsPresed = 0x01;
			}
			if(bit_is_set(PINC,PC1)){
				leftIsPresed = 0x01;
			}
			if(bit_is_set(PINC,PC2)){
				nonIsPresed = 0x01;
			}
			_delay_ms(1);
		}
		if(nonIsPresed == 0x01){
			tx_serial("00");
		} else{
			tx_serial_number(rightIsPresed);
			tx_serial_number(leftIsPresed);
		}
		tx_serial(";");

		playingSound = 0x00;
		earBeingPlayed = 0x00;
		PORTC = 0b0;
	}
	
	if(currentBufferIndex != 0 && currentData[currentBufferIndex-1] == ';'){ // når vi er nået slutningen af beskeden
		char firstDigitList[2];
		char secondDigitList[3];
		char chosenEarList[1];

		firstDigitList[0] = currentData[0];
		firstDigitList[1] = currentData[1];

		secondDigitList[0] = currentData[2];
		secondDigitList[1] = currentData[3];
		secondDigitList[2] = currentData[4];

		chosenEarList[0] = currentData[5];

		firstDigit = (uint16_t) charListToNumber(firstDigitList,2);
		secondDigit = (uint8_t) charListToNumber(secondDigitList,3);
		chosenEar = (uint8_t) charListToNumber(chosenEarList,1);
		
		earBeingPlayed = chosenEar << 1;
		playingSound = 1;
		PORTC = 0b1000;
		currentBufferIndex=0;
	}
	
  }
 	return 0;
}

int charListToNumber(char number[], int lenghOfArray){
	// denne kode skal ændres så den oversætter den liste som der bliver givet som agument
	// isedet for at den oversætter currentBufferIndex
	int i;
	int fullNumber = 0;
	int newDigit = 0;
	for (i=lenghOfArray-1; i>-1 ;i--){
		newDigit = (int)(number[i]-48);
		int j = 0;
		for(j = 0; j < lenghOfArray-1-i; j++){
		 	newDigit = newDigit * 10;
		}
		fullNumber += newDigit;	
	}
	return fullNumber;
}

void init(){
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