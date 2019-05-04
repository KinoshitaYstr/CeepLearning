#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct{
    unsigned char row;
    unsigned char col;
    double **m;
}matrix;

typedef struct{
    unsigned int size;
    double *v;
}vector;

typedef struct{
    double *bias;
    double **weights;
    unsigned int input_size;
    unsigned int output_size;
}neuron_params;

typedef struct{
    double *array;
    unsigned int size;
    double result;
}labal;

void testReadBinary();

void printBit(unsigned int val,int size);
unsigned int getMnistSize(FILE *fp);
unsigned char getRows(FILE *fp);
unsigned char getCols(FILE *fp);
void printDoubleMatrix(matrix m);
void writeDoubleMatrix2IntInCSV(matrix m,char fname[]);
void initMatrix(matrix *m,unsigned int row,unsigned int col);
void readMnistMatrix(matrix *m,FILE *fp,int num);
void initVector(vector *v,unsigned int size);
void readMnistVector(vector *v,FILE *fp,int num);
void exchangeMatrx2Vector(matrix m,vector *v);
void exchangeVector2Matrix(vector v,matrix *m);
void initNeuron(neuron_params *n,unsigned int input_size,unsigned int output_size);
void calcVectorNeuron(vector v,neuron_params n,vector *r);
double sigmoid(double x);
void initLabel(labal *l,unsigned int size);
void readMnistLabel(labal *l,FILE *fp,int num);

int main(int argc, char const *argv[]){
    //testReadBinary("dataset/mnist/t10k-images.idx3-ubyte","dataset/mnist/t10k-labels.idx1-ubyte");
    //testReadBinary("dataset/mnist/train-images.idx3-ubyte","dataset/mnist/train-labels.idx1-ubyte");
    vector input;
    neuron_params wb1;
    vector h1;
    neuron_params wb2;
    vector output;
    labal label;
    FILE *dataset;
    FILE *labelset;
    unsigned int input_row;
    unsigned int input_col;
    unsigned int input_size;
    unsigned int hidden_size;
    unsigned int output_size;

    if((dataset = fopen("dataset/mnist/t10k-images.idx3-ubyte","rb")) == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }
    if((labelset = fopen("dataset/mnist/t10k-labels.idx1-ubyte","rb")) == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }

    input_row = getRows(dataset);
    input_col = getCols(dataset);
    input_size = input_row*input_row;
    hidden_size = 100;
    output_size = 10;

    initVector(&input,input_size);
    initNeuron(&wb1,input_size,hidden_size);
    initVector(&h1,hidden_size);
    initNeuron(&wb2,hidden_size,output_size);
    initVector(&output,output_size);
    initLabel(&label,output_size);

    fclose(dataset);
    fclose(labelset);
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
            printf("%4d",(int)(m.m[i][j]*255));
        }
        printf("\n");
    }
}

void writeDoubleMatrix2IntInCSV(matrix m,char fname[]){
    FILE *fp;
    int tmp;
    if((fp = fopen(fname,"w")) == NULL){
        printf("create file error\n");
        exit(-1);
    }
    for(int i = 0;i < m.row;i++){
        for(int j = 0;j < m.col-1;j++){
            tmp = m.m[i][j] <= 0.0 ? 0 : (int)(m.m[i][j]*255);
            tmp = tmp >= 255 ? 255 : tmp;
            fprintf(fp,"%d,",tmp);
        }
        tmp = m.m[i][m.col-1] <= 0.0 ? 0 : (int)(m.m[i][m.col-1]*255);
        tmp = tmp >= 255 ? 255 : tmp;
        fprintf(fp,"%d\n",tmp);
    }
    fclose(fp);
}

void initMatrix(matrix *m,unsigned int row,unsigned int col){
    m->row = row;
    m->col = col;
    m->m = (double* *)malloc(sizeof(double*)*m->row);
    for(int i = 0;i < m->row;i++) m->m[i] = (double *)malloc(sizeof(double)*m->col);
}

void readMnistMatrix(matrix *m,FILE *fp,int num){
    unsigned char tmp;
    fseek(fp,16+num*m->row*m->col,SEEK_SET);
    for(int i = 0;i < m->row;i++){
        for(int j = 0;j < m->col;j++){
            fread(&tmp,1,1,fp);
            m->m[i][j] = (double)tmp/255.0;
        }
    }
}

void testReadBinary(char fname[],char fname2[]){
    FILE *fp;
    FILE *fp2;
    unsigned char dataType[4];
    unsigned char row;
    unsigned char col;
    unsigned char val;
    unsigned int fsize = 0;
    matrix testM;
    vector v;
    vector tmp;
    neuron_params n;
    labal l;

    fp = fopen(fname,"rb");
    if(fp == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }
    fp2 = fopen(fname2,"rb");
    if(fp == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }

    fsize = getMnistSize(fp);
    printf("fsize is %d\n",fsize);
    row = getRows(fp);
    col = getCols(fp);
    initMatrix(&testM,row,col);
    initVector(&v,row*col);
    initVector(&tmp,row*col);
    initNeuron(&n,v.size,tmp.size);
    initLabel(&l,10);
    for(int i = 0;i < 1;i++){
        readMnistMatrix(&testM,fp,i);
        exchangeMatrx2Vector(testM,&v);
        calcVectorNeuron(v,n,&tmp);
        v = tmp;
        exchangeVector2Matrix(v,&testM);
        printDoubleMatrix(testM);
        writeDoubleMatrix2IntInCSV(testM,"result/test.csv");
    }
    printf("-------------------------------");
    printf("%d,%d\n",testM.col,testM.row);
    printDoubleMatrix(testM);
    for(int j = 0;j < 100;j++){
        readMnistLabel(&l,fp2,j);
        printf("label is %f\n",l.result);
        for(int i = 0;i < 10;i++) printf("%4.1f,",l.array[i]);
        printf("\n");
    }

    fclose(fp);
}

void initVector(vector *v,unsigned int size){
    v->size = size;
    v->v = (double *)malloc(sizeof(double)*v->size);
}

void readMnistVector(vector *v,FILE *fp,int num){
    unsigned char tmp;
    fseek(fp,16+num*v->size,SEEK_SET);
    for(int i = 0;i < v->size;i++){
        fread(&tmp,1,1,fp);
        v->v[i] = (double)tmp/255.0;
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

void initNeuron(neuron_params *n,unsigned int input_size,unsigned int output_size){
    srand((unsigned)time(NULL));
    n->input_size = input_size;
    n->output_size = output_size;
    n->weights = (double* *)malloc(sizeof(double*)*output_size);
    for(int i = 0;i < output_size;i++) n->weights[i] = (double *)malloc(sizeof(double)*input_size);
    for(int i = 0;i < output_size;i++)
        for(int j = 0;j < input_size;j++)
            //n->weights[i][j] = 1;
            n->weights[i][j] = (double)rand()/RAND_MAX;
    n->bias = (double *)malloc(sizeof(double)*output_size);
    for(int i = 0;i < output_size;i++) n->bias[i] = (double)rand()/RAND_MAX;
    //for(int i = 0;i < output_size;i++) n->bias[i] = 0;
}

void calcVectorNeuron(vector v,neuron_params n,vector *r){
    if(v.size != n.input_size || r->size != n.output_size){
        printf("vec and weight size error\n");
        exit(-1);
    }
    for(int i = 0;i < n.output_size;i++){
        r->v[i] = 0.0;
        for(int j = 0;j < n.input_size;j++){
            r->v[i] += v.v[j]*n.weights[i][j];
        }
        r->v[i] = sigmoid(r->v[i]+n.bias[i]);
    }
}

double sigmoid(double x){
    return 1/(1+exp(-1*x));
}

void initLabel(labal *l,unsigned int size){
    l->array = (double *)malloc(sizeof(double)*size);
    l->size = size;
    l->result = 0;
}

void readMnistLabel(labal *l,FILE *fp,int num){
    unsigned char tmp;
    fseek(fp,8+num,SEEK_SET);
    fread(&tmp,1,1,fp);
    l->result = (double)tmp;
    for(int i = 0;i < l->size;i++) l->array[i] = 0;
    l->array[tmp] = 1.0;
}