#include <stdio.h>
#include <time.h>

int loop();

int loop(){
    time_t start, end;
    start = time(NULL);
    int value = 100000000;
    int result = 0;
    for (int i=0; i<value; i++){
        result++;
    }
    end = time(NULL);
    printf("Цикл использовал %f секунд.\n", difftime(end,start));
    return 0;
}