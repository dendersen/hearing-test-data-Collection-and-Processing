#include <avr/io.h>
#include <util/delay.h>
#include <avr/interrupt.h>
#include <stdlib.h>

#ifndef F_CPU //hvis F_CPU  (hastighed for MCU, (konstant / her til brug i udregninger)
#define F_CPU 8000000UL  //så definer den 
#endif
#define BAUDRATEVAL 9600



volatile char currentData[100];
volatile uint8_t counter = 0;

void init_seriel(){
	uint16_t ubrr0;
    
    ubrr0 = (((F_CPU / (BAUDRATEVAL * 16UL))) - 1);
    UBRR0H = (unsigned char) (ubrr0>>8);
    UBRR0L = (unsigned char) (ubrr0);
    UCSR0C = (1<<UCSZ00) | (1<<UCSZ01); //8 bit, 1 stop, no parity
    UCSR0B = (1<<TXEN0) | (1<<RXCIE0) | (1<<RXEN0); /* Enable  transmitter, receiver rx interrupt                 */
}

void tx_serial(volatile char data[]){
	
	uint8_t i = 0;
    while(data[i] != 0) 
    {
    	while (!( UCSR0A & (1<<UDRE0))); 
		    UDR0 = data[i];           
        i++;                             
    }
}

void tx_serial_number(uint16_t n){
	char string[8];
	itoa(n, string,10); //10 is radix
	tx_serial(string);
}