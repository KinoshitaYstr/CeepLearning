#include <stdio.h>
#include <stdlib.h>

int main(){
    double **test;
    printf("1\n");
    test = (double* *)malloc(sizeof(double*)*10000);
    for(int i = 0;i < 1000;i++) test[i] = (double *)malloc(sizeof(double)*100);
    printf("2\n");
    test = (double* *)realloc(test,sizeof(double*)*10);
    for(int i = 0;i < 10;i++) test[i] = (double *)realloc(test[i],sizeof(double)*10);
    printf("3\n");
    test = (double* *)realloc(test,sizeof(double*)*20000);
    for(int i = 0;i < 2000;i++) test[i] = (double *)realloc(test[i],sizeof(double)*20000);
    printf("4\n");
    test = (double* *)realloc(test,sizeof(double*)*10);
    for(int i = 0;i < 10;i++) test[i] = (double *)realloc(test[i],sizeof(double)*10);
    printf("5\n");
    //for(int i = 0;i < 10;i++) free(test[i]);
    //free(test);
    //printf("5-1\n");
    //test = (double* *)malloc(sizeof(double*)*2000);
    //for(int i = 0;i < 2000;i++) test[i] = (double *)malloc(sizeof(double)*20);

    test = (double* *)realloc(test,sizeof(double*)*20000);
    for(int i = 0;i < 2000;i++) test[i] = (double *)realloc(test[i],sizeof(double)*20000);

    printf("6\n");
    test = (double* *)realloc(test,sizeof(double*)*2000);
    printf("6-1\n");
    for(int i = 0;i < 2000;i++) test[i] = (double *)realloc(test[i],sizeof(double)*20);
    printf("7\n");
}