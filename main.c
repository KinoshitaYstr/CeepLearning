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
}label;

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
void initLabel(label *l,unsigned int size);
void readMnistLabel(label *l,FILE *fp,int num);
void softmax(vector input,vector *output);
double getCrossEntropyError(vector r,label l);
void forward(vector input,neuron_params wb[],unsigned int wb_size,vector *output);
void calcNumericalGradientForClossEntropyErrorAndSoftmax(vector x,neuron_params wb[],unsigned int wb_size,label t,neuron_params *grad);
void SGD(neuron_params *wb,unsigned int wb_size,FILE *dataset_fp,FILE *label_fp,int dataset_size);
void writeNeuronsInCSV(char fname[],neuron_params n[],int neuron_size);

int main(int argc, char const *argv[]){
    //testReadBinary("dataset/mnist/t10k-images.idx3-ubyte","dataset/mnist/t10k-labels.idx1-ubyte");
    //testReadBinary("dataset/mnist/train-images.idx3-ubyte","dataset/mnist/train-labels.idx1-ubyte");
    vector input;
    neuron_params wb[2];
    neuron_params grad[2];
    vector h[2];
    vector output;
    label label;
    FILE *dataset;
    FILE *labelset;
    unsigned int input_row;
    unsigned int input_col;
    unsigned int input_size;
    unsigned int hidden_size;
    unsigned int output_size;
    double e;

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
    initNeuron(&wb[0],input_size,hidden_size);
    initNeuron(&grad[0],input_size,hidden_size);
    initVector(&h[0],hidden_size);
    initNeuron(&wb[1],hidden_size,output_size);
    initNeuron(&grad[1],hidden_size,output_size);
    initVector(&h[1],output_size);
    initVector(&output,output_size);
    initLabel(&label,output_size);
    SGD(wb,2,dataset,labelset,60000);
    writeNeuronsInCSV("test.csv",wb,2);
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
            n->weights[i][j] = ((double)rand()/RAND_MAX)*0.01;
    n->bias = (double *)malloc(sizeof(double)*output_size);
    //for(int i = 0;i < output_size;i++) n->bias[i] = (double)rand()/RAND_MAX;
    for(int i = 0;i < output_size;i++) n->bias[i] = 0;
}

void calcVectorNeuron(vector v,neuron_params n,vector *r){
    int i,j;
    if(v.size != n.input_size || r->size != n.output_size){
        printf("vec and weight size error\n");
        exit(-1);
    }
    for(i = 0;i < n.output_size;i++){
        r->v[i] = 0.0;
        for(j = 0;j < n.input_size;j++){
            r->v[i] += v.v[j]*n.weights[i][j];
        }
        r->v[i] = sigmoid(r->v[i]+n.bias[i]);
    }
}

double sigmoid(double x){
    return 1/(1+exp(-1*x));
}

void initLabel(label *l,unsigned int size){
    l->array = (double *)malloc(sizeof(double)*size);
    l->size = size;
    l->result = 0;
}

void readMnistLabel(label *l,FILE *fp,int num){
    unsigned char tmp;
    fseek(fp,8+num,SEEK_SET);
    fread(&tmp,1,1,fp);
    l->result = (double)tmp;
    for(int i = 0;i < l->size;i++) l->array[i] = 0;
    l->array[tmp] = 1.0;
}

void softmax(vector input,vector *output){
    if(input.size != output-> size){
        printf("input and output error\n");
        exit(-1);
    }
    double max_val = 0;
    double all_exp = 0;
    for(int i = 0;i < input.size;i++)
        max_val = max_val < input.v[i] ? input.v[i] : max_val;
    for(int i = 0;i < input.size;i++) all_exp += exp(input.v[i]-max_val);
    for(int i = 0;i < input.size;i++) output->v[i] = exp(input.v[i]-max_val)/all_exp;

}

double getCrossEntropyError(vector r,label l){
    return -1*log(r.v[(int)l.result]+0.0001);
}

void forward(vector input,neuron_params wb[],unsigned int wb_size,vector *output){
    int i,j;
    for(i = 0;i < wb_size;i++){
        initVector(output,wb[i].output_size);
        calcVectorNeuron(input,wb[i],output);
        initVector(&input,wb[i].output_size);
        for(j = 0;j < wb[i].output_size;j++) input.v = output->v;
    }
}

void calcNumericalGradientForClossEntropyErrorAndSoftmax(vector x,neuron_params wb[],unsigned int wb_size,label t,neuron_params *grad){
    double delta = 0.001;
    double e1,e2;
    int i,j,k;
    vector forward_r;
    vector r;
    initVector(&forward_r,t.size);
    initVector(&r,t.size);
    for(i = 0;i < wb_size;i++){
        for(j = 0;j < wb[i].output_size;j++){
            for(k = 0;k < wb[i].input_size;k++){
                //printf("%d,%d,%d\n",i,j,k);
                wb[i].weights[j][k] -= delta;
                forward(x,wb,wb_size,&forward_r);
                softmax(forward_r,&r);
                e1 = getCrossEntropyError(r,t);
                wb[i].weights[j][k] += 2*delta;
                forward(x,wb,wb_size,&forward_r);
                softmax(forward_r,&r);
                e2 = getCrossEntropyError(r,t);
                wb[i].weights[j][k] -= delta;
                grad[i].weights[j][k] += (e2-e1)/(2*delta);
            }
            wb[i].bias[j] -= delta;
            forward(x,wb,wb_size,&forward_r);
            softmax(forward_r,&r);
            e1 = getCrossEntropyError(r,t);
            wb[i].bias[j] += 2*delta;
            forward(x,wb,wb_size,&forward_r);
            softmax(forward_r,&r);
            e2 = getCrossEntropyError(r,t);
            wb[i].bias[j] -= delta;
            grad[i].bias[j] += (e2-e1)/(2*delta);
        }
    }

}

void SGD(neuron_params *wb,unsigned int wb_size,FILE *dataset_fp,FILE *label_fp,int dataset_size){
    double learning_rate = 0.1;
    int batch_size = 1;
    double e;
    int i,x,y,z;
    int num = 0;
    neuron_params *grad;
    vector r1;
    vector r2;
    vector input;
    label label_data;
    grad = (neuron_params *)malloc(sizeof(neuron_params)*wb_size);
    for(i = 0;i < wb_size;i++) initNeuron(&grad[i],wb[i].input_size,wb[i].output_size);
    initVector(&input,wb[0].input_size);
    initLabel(&label_data,wb[wb_size-1].output_size);
    initVector(&r1,wb[wb_size-1].output_size);
    initVector(&r2,wb[wb_size-1].output_size);

    do{
        for(i = 0;i < batch_size;i++){
            num = (int)(rand()*(dataset_size+1)/(RAND_MAX+1));
            readMnistVector(&input,dataset_fp,num);
            readMnistLabel(&label_data,label_fp,num);
            for(x = 0;x <wb_size;x++){
                for(y = 0;y < wb[x].output_size;y++){
                    for(z = 0;z < wb[x].input_size;z++) grad[x].weights[y][z] = 0;
                    grad[x].bias[y] = 0;
                }
            }
            calcNumericalGradientForClossEntropyErrorAndSoftmax(input,wb,wb_size,label_data,grad);
        }
        for(x = 0;x < wb_size;x++){
            for(y = 0;y < wb[x].output_size;y++){
                for(z = 0;z < wb[x].input_size;z++){
                    wb[x].weights[y][z] -= learning_rate*grad[x].weights[y][z]/batch_size;
                    grad[x].weights[y][z] = 0.0;
                }
                wb[x].bias[y] -= learning_rate*grad[x].bias[y]/batch_size;
                grad[x].bias[y] = 0.0;
            }
        }
        forward(input,wb,wb_size,&r1);
        softmax(r1,&r2);
        e = getCrossEntropyError(r2,label_data);
        printf("error = %.50f\n",e);
    }while(abs(e) > 0.00001);

    free(grad);
}

void writeNeuronsInCSV(char fname[],neuron_params n[],int neuron_size){
    FILE *fp;
    int i,j,k;
    if((fp = fopen(fname,"w")) == NULL){
        printf("create file error\n");
        exit(-1);
    }
    for(i = 0;i < neuron_size;i++){
        for(j = 0;j < n[i].output_size;j++){
            for(k = 0;k < n[i].input_size;k++) fprintf(fp,"%f\n",n[i].weights[j][k]);
            fprintf(fp,"%f\n",n[i].bias[j]);
        }
    }
    fclose(fp);
}