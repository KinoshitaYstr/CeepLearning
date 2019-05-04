#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
    unsigned char row;
    unsigned char col;
    double **m;
}matrix;

typedef struct{
    unsigned int size;
    double *v;
}vector;

void testReadBinary();

void printBit(unsigned int val,int size);
unsigned int getMnistSize(FILE *fp);
unsigned char getRows(FILE *fp);
unsigned char getCols(FILE *fp);
double** getMnistMatrix(FILE *fp,unsigned char rows,unsigned char cols,int num);
void printDoubleMatrix(matrix m);
void writeDoubleMatrix2IntInCSV(matrix m,char fname[]);
void initMatrix(matrix *m,FILE *fp);
void readMnistMatrix(matrix *m,FILE *fp,int num);
void initVector(vector *v,FILE *fp);
void readMnistVector(vector *v,FILE *fp,int num);
void exchangeMatrx2Vector(matrix m,vector *v);
void exchangeVector2Matrix(vector v,matrix *m);

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

void printDoubleMatrix(matrix m){
    for(int i = 0;i < m.row;i++){
        for(int j = 0;j < m.col;j++){
            printf("%4d",(int)m.m[i][j]);
        }
        printf("\n");
    }
}

void writeDoubleMatrix2IntInCSV(matrix m,char fname[]){
    FILE *fp;
    if((fp = fopen(fname,"w")) == NULL){
        printf("create file error\n");
        exit(-1);
    }
    for(int i = 0;i < m.row;i++){
        for(int j = 0;j < m.col-1;j++) fprintf(fp,"%d,",(int)m.m[i][j]);
        fprintf(fp,"%d\n",(int)m.m[i][m.col-1]);
    }
    fclose(fp);
}

void initMatrix(matrix *m,FILE *fp){
    m->row = getRows(fp);
    m->col = getCols(fp);
    m->m = (double* *)malloc(sizeof(double*)*m->row);
    for(int i = 0;i < m->row;i++) m->m[i] = (double *)malloc(sizeof(double)*m->col);
}

void readMnistMatrix(matrix *m,FILE *fp,int num){
    unsigned char tmp;
    fseek(fp,16+num*m->row*m->col,SEEK_SET);
    for(int i = 0;i < m->row;i++){
        for(int j = 0;j < m->col;j++){
            fread(&tmp,1,1,fp);
            m->m[i][j] = (double)tmp;
        }
    }
}

void testReadBinary(char fname[]){
    FILE *fp;
    unsigned char dataType[4];
    unsigned char rows;
    unsigned char cols;
    unsigned char val;
    unsigned int fsize = 0;
    matrix testM;
    vector v;

    fp = fopen(fname,"rb");
    if(fp == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }

    fsize = getMnistSize(fp);
    printf("fsize is %d\n",fsize);
    initMatrix(&testM,fp);
    initVector(&v,fp);
    for(int i = 0;i < 3;i++){
        readMnistMatrix(&testM,fp,i);
        exchangeMatrx2Vector(testM,&v);
        exchangeVector2Matrix(v,&testM);
        printDoubleMatrix(testM);
        writeDoubleMatrix2IntInCSV(testM,"result/test.csv");
    }
    printf("-------------------------------");
    printf("%d,%d\n",testM.col,testM.row);
    printDoubleMatrix(testM);

    fclose(fp);
}

void initVector(vector *v,FILE *fp){
    v->size = getCols(fp)*getRows(fp);
    v->v = (double *)malloc(sizeof(double)*v->size);
}

void readMnistVector(vector *v,FILE *fp,int num){
    unsigned char tmp;
    fseek(fp,16+num*v->size,SEEK_SET);
    for(int i = 0;i < v->size;i++){
        fread(&tmp,1,1,fp);
        v->v[i] = (double)tmp;
    }
}

void exchangeMatrx2Vector(matrix m,vector *v){
    v->size = m.row*m.col;
    for(int i = 0;i < m.row;i++)
        for(int j = 0;j < m.col;j++)
            v->v[i*m.row+j] = m.m[i][j];
}

void exchangeVector2Matrix(vector v,matrix *m){
    m->row = (unsigned char)sqrt((double)v.size);
    m->col = m->row;
    for(int i = 0;i < m->row;i++)
        for(int j = 0;j < m->col;j++)
            m->m[i][j] = v.v[i*m->row+j];
}