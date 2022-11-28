#define BIT(x) (1 << (x))
#define SETBITS(x,y) ((x) |= (y))
#define CLEARBITS(x,y) ((x) &= (~(y)))
#define SETBIT(x,y) SETBITS((x), (BIT((y))))
#define CLEARBIT(x,y) CLEARBITS((x), (BIT((y))))
#define BITVAL(x,y) (((x)>>(y)) & 1)
#define HERTZ(x) ((CPU_CLOCK/400)/2)

#include <avr/io.h>
#include <util/delay.h>

void delayer(uint16_t mik,uint8_t d);

void playtone(uint16_t mik,uint8_t d){
  SETBITS(PORTB,BIT(1)|BIT(2));
  delayer(mik,d);

  PORTB &= 0;
 
  delayer(mik,d);
}

void delayer(uint16_t mik,uint8_t d){
    uint8_t i = 0;
    uint16_t y = 0b1;
    while(i!=d || y!= mik){
        _delay_us(1);
        y=y<<1;
        if(y==0b0){
            y |= 0b1;
            i++;
       }
    }
}