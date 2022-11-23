#include "HearingTest.c"
#include <avr/io.h>
#include <util/delay.h>
#include <avr/interrupt.h>
#include <stdlib.h>


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

void tx_serial_number(uint16_t n);
void tx_serial(volatile char data[]);

volatile char uart_buffer[100];
volatile uint8_t counter = 0;

int main(int argc, char const *argv[]){
  ini();
  while(1){
    //playtone();
  }
  return 0;
}

void ini(){
	DDRB = 0x06;
}


ISR(USART_RX_vect){
	uart_buffer[0] = UDR0;
	tx_serial_number(counter++);
	tx_serial("-Grettings from MCU I just received: ");
	tx_serial(uart_buffer);
	tx_serial(" ;");
}


void init_seriel(){
	uint16_t ubrr0;
    
    ubrr0 = (((F_CPU / (BAUDRATEVAL * 16UL))) - 1);
    UBRR0H = (unsigned char) (ubrr0>>8);
    UBRR0L = (unsigned char) (ubrr0);
    UCSR0C = (1<<UCSZ00) | (1<<UCSZ01); //8 bit, 1 stop, no parity
    UCSR0B = (1<<TXEN0) | (1<<RXCIE0) | (1<<RXEN0); /* Enable  transmitter, receiver rx interrupt                 */
}
	
void init(){
	DDRB = (1<<PB1);
	init_seriel();
	sei(); //global interrupt enable, global disable is: cli();
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
