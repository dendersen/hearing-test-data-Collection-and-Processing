#include <avr/io.h>
#include <util/delay.h>
void delayer(uint16_t mod,uint8_t d);
void playtone(int delay);
// int setDelayFromTone(int tone);

#define BIT(x) (1 << (x))
#define SETBITS(x,y) ((x) |= (y))


int main(){
    DDRB = 0xFF;

    while (1){
        playtone(1<<4,125);
    }
}

// int setDelayFromTone(int tone){
//     delay = (1/tone*1000) / 0.01 /2;
//     return delay;
// }

void playtone(int delay){
    SETBITS(PORTB,BIT(1)|BIT(2));
    delayer(32768,1);
    PORTB = 0;
  
    delayer(32768,1);  
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