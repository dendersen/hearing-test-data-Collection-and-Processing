#define BIT(x) (1 << (x))
#define SETBITS(x,y) ((x) |= (y))
#define CLEARBITS(x,y) ((x) &= (~(y)))
#define SETBIT(x,y) SETBITS((x), (BIT((y))))
#define CLEARBIT(x,y) CLEARBITS((x), (BIT((y))))
#define BITVAL(x,y) (((x)>>(y)) & 1)
#define HERTZ(x) ((CPU_CLOCK/400)/2)

#include <avr/io.h>
#include <util/delay.h>

void delayer(uint16_t firstDigit,uint8_t secondDigit);

uint8_t earBeingPlayed = 2; 

void playtone(uint16_t firstDigit,uint8_t secondDigit){
  PORTB = earBeingPlayed;
  //SETBITS(PORTB,BIT(1)|BIT(2));
  delayer(firstDigit,secondDigit);

  PORTB &= 0;
 
  delayer(firstDigit,secondDigit);
}

void delayer(uint16_t firstDigit,uint8_t secondDigit){
    uint8_t localSecondDigit = 0;
    uint16_t localFirstDigit = 0b1;
    while(localSecondDigit!=secondDigit || localFirstDigit!= firstDigit){
        _delay_us(1);
        localFirstDigit=localFirstDigit<<1;
        if(localFirstDigit==0b0){
            localFirstDigit |= 0b1;
            localSecondDigit++;
       }
    }
}