#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void testReadBinary();

void printBit(unsigned int val,int size);
unsigned int getMnistSize(FILE *fp);
unsigned char getRows(FILE *fp);
unsigned char getCols(FILE *fp);double** getMnistMatrix(FILE *fp,unsigned char rows,unsigned char cols,int num);
double** getMnistMatrix(FILE *fp,unsigned char rows,unsigned char cols,int num);
void printDoubleMatrix(double *m[],unsigned char rows,unsigned char cols);
void writeDoubleMatrix2IntInCSV(double *m[],unsigned char rows,unsigned char cols,char fname[]);

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

double** getMnistMatrix(FILE *fp,unsigned char rows,unsigned char cols,int num){
    double** m;
    unsigned char tmp;
    fseek(fp,16+num*rows*cols,SEEK_SET);
    m = (double* *)malloc(sizeof(double*)*rows);
    for(int i = 0;i < rows;i++) m[i] = (double*)malloc(sizeof(double)*cols);
    for(int i = 0;i < rows;i++){
        for(int j = 0;j < cols;j++){
            fread(&tmp,1,1,fp);
            m[i][j] = (double)tmp;
        }
    }
    return m;
}

void printDoubleMatrix(double *m[],unsigned char rows,unsigned char cols){
    for(int i = 0;i < rows;i++){
        for(int j = 0;j < cols;j++){
            printf("%4d",(int)m[i][j]);
        }
        printf("\n");
    }
}

void writeDoubleMatrix2IntInCSV(double *m[],unsigned char rows,unsigned char cols,char fname[]){
    FILE *fp;
    if((fp = fopen(fname,"w")) == NULL){
        printf("create file error\n");
        exit(-1);
    }
    for(int i = 0;i < rows;i++){
        for(int j = 0;j < cols-1;j++) fprintf(fp,"%d,",(int)m[i][j]);
        fprintf(fp,"%d\n",(int)m[i][cols-1]);
    }
    fclose(fp);
}

void testReadBinary(char fname[]){
    FILE *fp;
    unsigned char dataType[4];
    unsigned char rows;
    unsigned char cols;
    unsigned char val;
    unsigned int fsize = 0;
    double** m;
    
    fp = fopen(fname,"rb");
    if(fp == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }

    fsize = getMnistSize(fp);
    printf("fsize is %d\n",fsize);
    rows = getRows(fp);
    printf("rows is %d\n",rows);
    cols = getCols(fp);
    printf("cols is %d\n",cols);

    m = (double* *)malloc(sizeof(double*)*rows);
    for(int i = 0;i < rows;i++) m[i] = (double *)malloc(sizeof(double)*cols);
    for(int i = 0;i < 3;i++){
        m = getMnistMatrix(fp,rows,cols,i);
        printDoubleMatrix(m,rows,cols);
        writeDoubleMatrix2IntInCSV(m,rows,cols,"result/test.csv");
    }

    fclose(fp);
}