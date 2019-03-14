#include <stdio.h>
#include <stdlib.h>

void testReadBinary();

int main(int argc, char const *argv[]){
    testReadBinary("dataset/mnist/t10k-images.idx3-ubyte");
    testReadBinary("dataset/mnist/train-images.idx3-ubyte");
    return 0;
}

void testReadBinary(char fname[]){
    FILE *fp;
    unsigned char dataType[4];
    char rows;
    char cols;
    unsigned char val;
    char flag;
    
    fp = fopen(fname,"rb");
    if(fp == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }

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


    fclose(fp);
}