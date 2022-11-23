#define BIT(x) (1 << (x))
#define SETBITS(x,y) ((x) |= (y))
#define CLEARBITS(x,y) ((x) &= (~(y)))
#define SETBIT(x,y) SETBITS((x), (BIT((y))))
#define CLEARBIT(x,y) CLEARBITS((x), (BIT((y))))
#define BITVAL(x,y) (((x)>>(y)) & 1)
#define HERTZ(x) ((CPU_CLOCK/400)/2)

#include <avr/io.h>
#include <util/delay.h>

void playtone(){
  SETBITS(PORTB,BIT(1)|BIT(2));
  _delay_ms(0.1);
  PORTB = 0;
  _delay_ms(0.1);
}