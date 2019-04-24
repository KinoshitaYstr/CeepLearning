#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
    unsigned short bfType;
	unsigned long  bfSize;
	unsigned short bfReserved1;
	unsigned short bfReserved2;
	unsigned long  bfOffBits;
} BITMAPFILEHEADER;

typedef struct{
    unsigned long biSize;
	         long biWidth;
	         long biHeight;
	unsigned short biPlanes;
	unsigned short biBitCount;
	unsigned long biCompression;
	unsigned long biSizeimage;
	         long biXPixPerMeter;
	         long biYPixPerMeter;
	unsigned long biClrUsed;
	unsigned long biClrImportant;
}BITMAPINFOHEADER;

void testReadBinary();

void printBit(unsigned int val,int size);
unsigned int getMnistSize(FILE *fp);
unsigned char getRows(FILE *fp);
unsigned char getCols(FILE *fp);double** getMnistMatrix(FILE *fp,unsigned char rows,unsigned char cols,int num);
double** getMnistMatrix(FILE *fp,unsigned char rows,unsigned char cols,int num);
void printDoubleMatrix(double *m[],unsigned char rows,unsigned char cols);
BITMAPFILEHEADER createBitMapFileHeaderForRGB_DATA(int width,int heigth);
BITMAPINFOHEADER createBitMapInfoHeaderForRGB_DATA(int width,int heigth);
void createBMPForMono(char fname[],double *m[],unsigned char rows,unsigned char cols);

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

BITMAPFILEHEADER createBitMapFileHeaderForRGB_DATA(int width,int heigth)
{
    BITMAPFILEHEADER header;
    header.bfType = 0x4d42;
    //header.bfSize = 54+width*heigth*3;
    header.bfSize = 54-24+width*heigth*3;
    header.bfReserved1 = 0x00;
    header.bfReserved2 = 0x00;
    //header.bfOffBits = 54;
    header.bfOffBits = 54-24;
    return header;
}

BITMAPINFOHEADER createBitMapInfoHeaderForRGB_DATA(int width,int heigth)
{
    BITMAPINFOHEADER header;
    header.biSize = 40;
    header.biWidth = width;
    header.biHeight = heigth;
    header.biPlanes = 1;
    header.biBitCount = 24;
    //header.biCompression = 0;
    //header.biSizeimage = 3*heigth*width;
    //header.biXPixPerMeter = 0;
    //header.biYPixPerMeter = 0;
    //header.biClrUsed = 0;
    //header.biClrImportant = 0;
    return header;
}

void createBMPForMono(char fname[],double *m[],unsigned char rows,unsigned char cols)
{
    FILE *fp;
    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;
    unsigned char val;
    if((fp = fopen(fname,"wb")) == NULL)
    {
        printf("create BMP file error\n");
        exit(-1);
    }
    fileHeader = createBitMapFileHeaderForRGB_DATA((int)cols,(int)rows);
    infoHeader = createBitMapInfoHeaderForRGB_DATA((int)cols,(int)rows);
    fseek(fp,0,SEEK_SET);
    fwrite(&fileHeader.bfType,sizeof(unsigned short),1,fp);
    fwrite(&fileHeader.bfSize,sizeof(unsigned long),1,fp);
    fwrite(&fileHeader.bfReserved1,sizeof(unsigned short),1,fp);
    fwrite(&fileHeader.bfReserved2,sizeof(unsigned short),1,fp);
    fwrite(&fileHeader.bfOffBits,sizeof(unsigned long),1,fp);
    fseek(fp,14,SEEK_SET);
    fwrite(&infoHeader.biSize,sizeof(unsigned long),1,fp);
    fwrite(&infoHeader.biWidth,sizeof(long),1,fp);
    fwrite(&infoHeader.biHeight,sizeof(long),1,fp);
    fwrite(&infoHeader.biPlanes,sizeof(unsigned short),1,fp);
    fwrite(&infoHeader.biBitCount,sizeof(unsigned short),1,fp);
    fwrite(&infoHeader.biCompression,sizeof(unsigned long),1,fp);
    fwrite(&infoHeader.biSizeimage,sizeof(unsigned long),1,fp);
    fwrite(&infoHeader.biXPixPerMeter,sizeof(long),1,fp);
    fwrite(&infoHeader.biYPixPerMeter,sizeof(long),1,fp);
    fwrite(&infoHeader.biClrUsed,sizeof(unsigned long),1,fp);
    fwrite(&infoHeader.biClrImportant,sizeof(unsigned long),1,fp);
    fseek(fp,54,SEEK_SET);
    for(int i = 0;i < infoHeader.biHeight;i++)
    {
        for(int j = 0;j < infoHeader.biWidth;j++)
        {
            val = (unsigned char)m[i][j];
            fwrite(&val,sizeof(unsigned char),1,fp);
            fwrite(&val,sizeof(unsigned char),1,fp);
            fwrite(&val,sizeof(unsigned char),1,fp);
        }
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
        createBMPForMono("test.bmp",m,rows,cols);
    }

    fclose(fp);
}