#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void testReadBinary();

void printBit(unsigned int val,int size);
unsigned int getMnistSize(FILE *fp);
unsigned char getRows(FILE *fp);
unsigned char getCols(FILE *fp);

int main(int argc, char const *argv[]){
    testReadBinary("dataset/mnist/t10k-images.idx3-ubyte");
    testReadBinary("dataset/mnist/train-images.idx3-ubyte");
    return 0;
}

void printBit(unsigned int val,int size){
    unsigned int flag;
    flag = 1<<(size-1);
    for(int i = 0;i < size;i++){
        if(i%4 == 0) printf(" ");
        if(val&flag) printf("1");
        else printf("0");
        flag = flag>>1;
    }
    printf("\n");
}

unsigned int getMnistSize(FILE *fp){
    unsigned int size = 0;
    unsigned char tmp_c[4];
    fseek(fp,4,SEEK_SET);
    fread(&tmp_c,1,4,fp);
    for(int i = 0;i < 3;i++){
        size += tmp_c[i];
        size = size<<8;
    }
    size += tmp_c[3];
    return size;
}

unsigned char getRows(FILE *fp){
    unsigned char rows;
    fseek(fp,8+3,SEEK_SET);
    fread(&rows,1,1,fp);
    return rows;
}

unsigned char getCols(FILE *fp){
    unsigned char cols;
    fseek(fp,12+3,SEEK_SET);
    fread(&cols,1,1,fp);
    return cols;
}

void testReadBinary(char fname[]){
    FILE *fp;
    unsigned char dataType[4];
    unsigned char rows;
    unsigned char cols;
    unsigned char val;
    unsigned int fsize = 0;
    unsigned int flag;
    
    fp = fopen(fname,"rb");
    if(fp == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }

    fsize = 0;
    fseek(fp,4,SEEK_SET);
    fread(&fsize,4,1,fp);
    printf("all fsize = %d\n",fsize);
    printBit(fsize,32);
    
    fsize = 0b1110101001100000;
    printf("all fsize = %d\n",fsize);
    printBit(fsize,32);


    fsize = 10000;
    printf("all fsize = %d\n",fsize);
    flag = 1<<31;
    printBit(fsize,32);

    fsize = 0;
    fseek(fp,4,SEEK_SET);
    fread(&fsize,1,1,fp);
    printf("fsize = %d\n",fsize);
    fread(&fsize,1,1,fp);
    printf("fsize = %d\n",fsize);
    fread(&fsize,1,1,fp);
    printf("fsize = %d\n",fsize);
    fread(&fsize,1,1,fp);
    printf("fsize = %d\n",fsize);
    
    fseek(fp,8+3,SEEK_SET);
    fread(&rows,1,1,fp);
    fseek(fp,12+3,SEEK_SET);
    fread(&cols,1,1,fp);
    printf("(rows,cols) = (%d,%d)\n",rows,cols);
    
    fseek(fp,16,SEEK_SET);
    for(int c = 0;c < cols;c++){
        for(int r = 0;r < rows;r++){
            fread(&val,1,1,fp);
            printf("%4d",val);
        }
        printf("\n");
    }
    

    getMnistSize(fp);

    fclose(fp);
}