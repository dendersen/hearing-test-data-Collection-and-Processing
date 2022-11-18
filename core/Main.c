#define BIT(x) (1 << (x))
#define SETBITS(x,y) ((x) |= (y))
#define CLEARBITS(x,y) ((x) &= (~(y)))
#define SETBIT(x,y) SETBITS((x), (BIT((y))))
#define CLEARBIT(x,y) CLEARBITS((x), (BIT((y))))
#define BITVAL(x,y) (((x)>>(y)) & 1)
#define HERTZ(x) ((CPU_CLOCK/400)/2)

#include <avr/io.h>
#include <util/delay.h>

int main(int argc, char const *argv[]){
  /* code */
  return 0;
}

void ini(){
	DDRB = 0x06; 
}

