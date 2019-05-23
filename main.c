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
void calcBackProbagationForClossEntropyErrorAndSoftamx(vector x,neuron_params wb[],unsigned int wb_size,label t,neuron_params *grad);
void BP(neuron_params *wb,unsigned int wb_size,FILE *dataset_fp,FILE *label_fp,int dataset_size);
matrix createMatrix(unsigned char row,unsigned char col);
vector createVector(unsigned int size);
neuron_params createNeuronParams(unsigned int input_size,unsigned output_size);
label createLabel(unsigned int size);

int main(int argc, char const *argv[]){
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

    printf("open dataset and labelset\n");
    if((dataset = fopen("dataset/mnist/t10k-images.idx3-ubyte","rb")) == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }
    if((labelset = fopen("dataset/mnist/t10k-labels.idx1-ubyte","rb")) == NULL){
        fputs("file read open error\n",stderr);
        exit(-1);
    }

    printf("read dataset size etc...\n");
    input_row = getRows(dataset);
    input_col = getCols(dataset);

    printf("network size setting\n");
    input_size = input_row*input_row;
    hidden_size = 100;
    output_size = 10;

    printf("init vector etc...\n");
    input = createVector(input_size);
    wb[0] = createNeuronParams(input_size,hidden_size);
    grad[0] = createNeuronParams(input_size,hidden_size);
    h[0] = createVector(hidden_size);
    wb[1] = createNeuronParams(hidden_size,output_size);
    grad[1] = createNeuronParams(hidden_size,output_size);
    h[1] = createVector(output_size);
    output = createVector(output_size);
    label = createLabel(output_size);
    
    printf("learning\n");
    //SGD(wb,2,dataset,labelset,60000);
    BP(wb,2,dataset,labelset,60000);
    
    //printf("write result's neuron\n");
    //writeNeuronsInCSV("test.csv",wb,2);

    printf("close dataset and labelset\n");
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
    m->m = (double* *)realloc(m->m,sizeof(double*)*m->row);
    for(int i = 0;i < m->row;i++) m->m[i] = (double *)realloc(m->m[i],sizeof(double)*m->col);
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
    v->v = (double *)realloc(v->v,sizeof(double)*size);
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
    n->weights = (double* *)realloc(n->weights,sizeof(double*)*output_size);
    for(int i = 0;i < output_size;i++) n->weights[i] = (double *)realloc(n->weights[i],sizeof(double)*input_size);
    for(int i = 0;i < output_size;i++)
        for(int j = 0;j < input_size;j++)
            //n->weights[i][j] = 1;
            n->weights[i][j] = ((double)rand()/RAND_MAX)*0.01;
    n->bias = (double *)realloc(n->bias,sizeof(double)*output_size);
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
    l->array = (double *)realloc(l->array,sizeof(double)*size);
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
    vector r;
    vector input1;
    input1 = createVector(input.size);
    for(i = 0;i < input.size;i++) input1.v[i] = input.v[i];
    r = createVector(wb[0].output_size);
    for(i = 0;i < wb_size;i++){
        initVector(&r,wb[i].output_size);
        calcVectorNeuron(input1,wb[i],&r);
        initVector(&input1,wb[i].output_size);
        for(j = 0;j < r.size;j++) input1.v[j] = r.v[j];
    }
    for(i = 0;i < r.size;i++) output->v[i] = r.v[i];
    free(r.v);
    free(input1.v);
}

void calcNumericalGradientForClossEntropyErrorAndSoftmax(vector x,neuron_params wb[],unsigned int wb_size,label t,neuron_params *grad){
    double delta = 0.001;
    double e1,e2;
    int i,j,k;
    int tmp;
    vector forward_r;
    vector r;
    forward_r = createVector(t.size);
    r = createVector(t.size);
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
    free(forward_r.v);
    free(r.v);
}

void SGD(neuron_params *wb,unsigned int wb_size,FILE *dataset_fp,FILE *label_fp,int dataset_size){
    double learning_rate = 1;
    int batch_size = 1;
    double e;
    int i,x,y,z;
    int num = 0;
    int count = 1;
    neuron_params *grad;
    vector r1;
    vector r2;
    vector input;
    label label_data;
    grad = (neuron_params *)malloc(sizeof(neuron_params)*wb_size);
    for(i = 0;i < wb_size;i++){
        grad[i] = createNeuronParams(wb[i].input_size,wb[i].output_size);
    }
    input = createVector(wb[0].input_size);
    label_data = createLabel(wb[wb_size-1].output_size);
    r1 = createVector(wb[wb_size-1].output_size);
    r2 = createVector(wb[wb_size-1].output_size);
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
        printf("No%d -> error = %.50f\n",count,e);
        count++;
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

void calcBackProbagationForClossEntropyErrorAndSoftamx(vector x,neuron_params wb[],unsigned int wb_size,label t,neuron_params *grad){
    int i,j,k,n,m;
    int count = 0;
    vector forward_r;
    vector r;
    vector calc_x[wb_size+1];
    forward_r = createVector(t.size);
    r = createVector(t.size);
    calc_x[0] = createVector(x.size);
    for(i = 0;i < x.size;i++) calc_x[0].v[i] = x.v[i];
    for(i = 0;i < wb_size;i++){
        calc_x[i+1] = createVector(wb[i].output_size);
        calcVectorNeuron(calc_x[i],wb[i],&calc_x[i+1]);
    }
    for(int i = 0;i < t.size;i++) forward_r.v[i] = calc_x[wb_size].v[i];
    softmax(forward_r,&calc_x[wb_size]);
    for(i = 0;i < r.size;i++) r.v[i] =  calc_x[wb_size].v[i]-t.array[i];

    for(i = 0;i < wb_size;i++)
    for(j = 0;j < wb[i].output_size;j++)
        for(k = 0;k < wb[i].input_size;k++)
            if(grad[i].weights[j][k] != 0) count++;
    printf("not zero size is %d\n",count);


    for(i = wb_size-1;i >= 0;i--){
        initVector(&forward_r,wb[i].input_size);
        /*
        printf("-------------------------\n");
        printf("i is %d\n",i);
        printf("forward_r size is %d\n",forward_r.size);
        printf("wb[%d] inout size %d\n",i,wb[i].input_size);
        printf("wb[%d] output size %d\n",i,wb[i].output_size);
        printf("calc_x[%d] size is %d\n",i+1,calc_x[i+1].size);
        printf("%d -> %d,%d\n",wb[i].input_size,wb[i].output_size,calc_x[i].size);
        printf("-------------------------\n");
        printf("calc_x[%d] size is %d\n",i,calc_x[i].size);
        printf("r size is %d\n",r.size);
        printf("grad[%d] inout size %d\n",i,grad[i].input_size);
        printf("grad[%d] output size %d\n",i,grad[i].output_size);
        */
        for(j = 0;j < grad[i].output_size;j++){
            grad[i].bias[j] += r.v[j];
            for(k = 0;k < grad[i].input_size;k++){
                grad[i].weights[j][k] += r.v[j]*calc_x[i].v[k];
            }
        }
        for(j = 0;j < forward_r.size;j++){
            forward_r.v[j] = 0;
            for(k = 0;k < r.size;k++){
                forward_r.v[j] += r.v[k]*wb[i].weights[k][j];
            }
        }
        initVector(&r,wb[i].input_size);
        for(j = 0;j < r.size;j++) r.v[j] = forward_r.v[j];
    }
    count = 0;
    for(i = 0;i < wb_size;i++)
        for(j = 0;j < wb[i].output_size;j++)
            for(k = 0;k < wb[i].input_size;k++)
                if(grad[i].weights[j][k] != 0.0){
                    count++;
                    //printf("%f\n",grad[i].weights[j][k]);
                }
    printf("not zero size is %d\n",count);
    free(forward_r.v);
    free(r.v);
    for(i = 0;i < wb_size;i++) free(calc_x[i].v);
}

void calcBackProbagationForClossEntropyErrorAndSoftamx2(vector x,neuron_params wb[],unsigned int wb_size,label t,neuron_params *grad){
    int i,j,k,n,m;
    int count = 0;
    vector calc_x[wb_size+1];
    vector forward_r;
    calc_x[0] = createVector(x.size);
    for(i = 0;i < x.size;i++) calc_x[0].v[i] = x.v[i];
    for(i = 0;i < wb_size;i++){
        calc_x[i+1] = createVector(wb[i].output_size);
        calcVectorNeuron(calc_x[i],wb[i],&calc_x[i+1]);
    }
    forward_r = createVector(t.size);
    for(i = 0;i < forward_r.size;i++) forward_r.v[i] = calc_x[wb_size].v[i];
    softmax(forward_r,&calc_x[wb_size]);
    for(i = 0;i < t.size;i++) forward_r.v[i] = calc_x[wb_size].v[i] - t.array[i];
    for(i = wb_size-1;i >= 0;i--){
        // grad バイアスの代入
        printf("grad[%d] bias[%d] = forward_r v[%d]\n",i,grad[i].output_size,forward_r.size);
        for(j = 0;j < calc_x[i+1].size;j++) grad[i].bias[j] = forward_r.v[j];
        // grad 重さの計算
        printf("grad[%d] weights[%d,%d] = grad[%d] bias[%d]*calc_x[%d] v[%d]\n",i,grad[i].output_size,grad[i].input_size,i,grad[i].output_size,i,calc_x[i].size);
        for(j = 0;j < grad[i].output_size;j++)
            for(k = 0;k < grad[i].input_size;k++)
                grad[i].weights[j][k] = grad[i].bias[j]*calc_x[i].v[k];
        // x 一個前の値の計算？
        printf("forward_r v[%d] <- wb[%d] input[%d]\n",forward_r.size,i,wb[i].input_size);
        forward_r = createVector(wb[i].input_size);
        printf("forward_r v[%d] = grad[%d] bias[%d] * wb[%d] mat[%d,%d]\n",forward_r.size,i,grad[i].output_size,i,wb[i].output_size,wb[i].input_size);
        for(j = 0;j < forward_r.size;j++){
            forward_r.v[j] = 0.0;
            for(k = 0;k < grad[i].output_size;k++){
                forward_r.v[j] += grad[i].bias[k]*wb[i].weights[k][j];
            }
        }
    }
}

void BP(neuron_params *wb,unsigned int wb_size,FILE *dataset_fp,FILE *label_fp,int dataset_size){
    double learning_rate = 0.1;
    int batch_size = 1;
    double e;
    int i,x,y,z;
    int num = 0;
    int count = 1;
    int aaa;
    neuron_params *grad;
    neuron_params *grad2;
    vector r1;
    vector r2;
    vector input;
    label label_data;
    printf("create grad\n");
    grad = (neuron_params *)malloc(sizeof(neuron_params)*wb_size);
    grad2 = (neuron_params *)malloc(sizeof(neuron_params)*wb_size);
    //for(i = 0;i < wb_size;i++) initNeuron(&grad[i],wb[i].input_size,wb[i].output_size);
    for(i = 0;i < wb_size;i++){
        printf("%d -> ",i);
        grad[i] = createNeuronParams(wb[i].input_size,wb[i].output_size);
        grad2[i] = createNeuronParams(wb[i].input_size,wb[i].output_size);
        printf("%d\n",i);
    }
    printf("create input\n");
    input = createVector(wb[0].input_size);
    printf("create label\n");
    label_data = createLabel(wb[wb_size-1].output_size);
    r1 = createVector(wb[wb_size-1].output_size);
    r2 = createVector(wb[wb_size-1].output_size);

    do{
        for(i = 0;i < batch_size;i++){
            num = (int)(rand()*(dataset_size+1)/(RAND_MAX+1));
            readMnistVector(&input,dataset_fp,num);
            readMnistLabel(&label_data,label_fp,num);
            for(x = 0;x < wb_size;x++){
                for(y = 0;y < wb[x].output_size;y++){
                    for(z = 0;z < wb[x].input_size;z++) grad[x].weights[y][z] = 0.0;
                    grad[x].bias[y] = 0.0;
                }
            }
            calcBackProbagationForClossEntropyErrorAndSoftamx2(input,wb,wb_size,label_data,grad);
            //calcNumericalGradientForClossEntropyErrorAndSoftmax(input,wb,wb_size,label_data,grad2);
        }
        aaa = 0;
        for(x = 0;x < wb_size;x++){
            for(y = 0;y < wb[x].output_size;y++){
                for(z = 0;z < wb[x].input_size;z++){
                    if(grad[x].weights[y][z] != 0) aaa++;
                    //printf("%f : %f\n",grad[x].weights[y][z],grad2[x].weights[y][z]);
                    wb[x].weights[y][z] -= learning_rate*grad[x].weights[y][z]/batch_size;
                    grad[x].weights[y][z] = 0.0;
                }
              //  printf("%f : %f\n",grad[x].bias[y],grad2[x].bias[y]);
                wb[x].bias[y] -= learning_rate*grad[x].bias[y]/batch_size;
                grad[x].bias[y] = 0.0;
            }
        }
        printf("aaa is %d\n",aaa);
        forward(input,wb,wb_size,&r1);
        softmax(r1,&r2);
        e = getCrossEntropyError(r2,label_data);
        printf("No%d -> error = %.50f\n",count,e);
        count++;
    }while(abs(e) > 0.00001);
    
    free(grad);
}

matrix createMatrix(unsigned char row,unsigned char col){
    matrix m;
    m.row = row;
    m.col = col;
    m.m = (double* *)malloc(sizeof(double*)*row);
    for(int i = 0;i < row;i++) m.m[i] = (double *)malloc(sizeof(double)*col);
    return m;
}

vector createVector(unsigned int size){
    vector v;
    v.size = size;
    v.v = (double *)malloc(sizeof(double)*size);
    return v;
}

neuron_params createNeuronParams(unsigned int input_size,unsigned output_size){
    neuron_params n;
    int i,j;
    n.input_size = input_size;
    n.output_size = output_size;
    n.weights = (double* *)malloc(sizeof(double*)*output_size);
    n.bias = (double *)malloc(sizeof(double)*output_size);
    for(i = 0;i < output_size;i++){
        n.weights[i] = (double *)malloc(sizeof(double)*input_size);
        for(j = 0;j < input_size;j++) n.weights[i][j] = ((double)rand()/RAND_MAX)*0.01;
        n.bias[i] = 0;
    }
    return n;
}

label createLabel(unsigned int size){
    label l;
    l.size = size;
    l.result = 0;
    l.array = (double *)malloc(sizeof(double)*size);
    return l;
}