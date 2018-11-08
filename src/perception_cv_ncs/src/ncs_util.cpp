//
// Created by pesong on 18-9-5.
//

#include "ncs_util.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

extern "C" {
#include "fp16.h"
}


//! movidius 设备预处理
bool g_graph_Success;
ncStatus_t retCode;
struct ncDeviceHandle_t* deviceHandlePtr;
struct ncGraphHandle_t* graphHandlePtr;
void* graphFileBuf;
unsigned int graphFileLen;
// Now we need to allocate graph and create and in/out fifos
struct ncFifoHandle_t* inFifoHandlePtr;
struct ncFifoHandle_t* outFifoHandlePtr;


image* ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;

    image* out = new image(w,h,c);

    ipl_into_image(src, out);
    return out;
}


void ipl_into_image(IplImage* src, image* im)
{
//  unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;
    // std::cout << "h: " << h << " w: " << w << " c: " << c << std::endl;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im->data[k*w*h + i*w + j] = ((unsigned char *)src->imageData)[i*step + j*c + k]/255.;
            }
        }
    }
}


unsigned char* image_to_stb(image* in)
{
    int i,j,k;
    int w = in->w;
    int h = in->h;
    int c =3;
    unsigned char *img = (unsigned char*) malloc(c*w*h);
    for(k = 0; k < c; ++k){
        for(j=0; j <h; ++j){
            for(i=0; i<w; ++i){
                int src_index = i + w*j + w*h*k;
                int dst_index = k + c*i + c*w*j;
                img[dst_index] = (unsigned char)(255*in->data[src_index]);
            }
        }
    }
    delete in;
    // std::cout << "xxxxx" << std::endl;
    return img;
}



unsigned char* cvMat_to_charImg(cv::Mat pic)
{
    IplImage copy = pic;
    IplImage *pic_Ipl = &copy;

    image* buff_;
    buff_ = ipl_to_image(pic_Ipl);
    unsigned char* pic_final = image_to_stb(buff_);

    //cvReleaseImage(&pic_Ipl);

    return pic_final;
}

// Load a graph file, caller must free the buffer returned.
// path is the full or relative path to the graph file, length is the number of bytes read upon return
void *LoadFile(const char *path, unsigned int *length)
{
    FILE *fp;
    char *buf;

    fp = fopen(path, "rb");
    if(fp == NULL)
        return 0;
    fseek(fp, 0, SEEK_END);
    *length = ftell(fp);
    rewind(fp);
    if(!(buf = (char*) malloc(*length)))
    {
        fclose(fp);
        return 0;
    }
    if(fread(buf, 1, *length, fp) != *length)
    {
        fclose(fp);
        free(buf);
        return 0;
    }
    fclose(fp);
    return buf;
}

// 加载为movidius所需要的图片格式，并resize为固定大小
float *LoadImage32(unsigned char *img, int target_w, int target_h, int ori_w, int ori_h, float *mean)
{
    int i;
    unsigned char *imgresized;
    float *imgfp32;

    if(!img)
    {
        printf("The picture  could not be loaded\n");
        return 0;
    }
    imgresized = (unsigned char*) malloc(3*target_w*target_h);
    if(!imgresized)
    {
        free(img);
        perror("malloc");
        return 0;
    }
    stbir_resize_uint8(img, ori_w, ori_h, 0, imgresized, target_w, target_h, 0, 3);
    free(img);
    imgfp32 = (float*) malloc(sizeof(*imgfp32) * target_w * target_h * 3);
    if(!imgfp32)
    {
        free(imgresized);
        perror("malloc");
        return 0;
    }
    for(i = 0; i < target_w * target_h * 3; i++)
    {
        imgfp32[i] = imgresized[i];
    }
    free(imgresized);

    for(i = 0; i < target_w*target_h; i++)
    {
        // imgfp32 comes in RGB order but network expects to be in
        // BRG order so convert to BGR here while subtracting the mean.
        float blue, green, red;
        blue = imgfp32[3*i+2];
        green = imgfp32[3*i+1];
        red = imgfp32[3*i+0];

        imgfp32[3*i+0] = (blue-mean[0]) * 0.007843;
        imgfp32[3*i+1] = (green-mean[1]) * 0.007843;
        imgfp32[3*i+2] = (red-mean[2]) * 0.007843;

        // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
        //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
    }
    return imgfp32;
}


// 加载为movidius所需要的图片格式，并resize为固定大小
half *LoadImage(unsigned char *img, int target_w, int target_h, int ori_w, int ori_h, float *mean)
{
    int i;
    unsigned char *imgresized;
    float *imgfp32;
    half *imgfp16;

    if(!img)
    {
        printf("The picture  could not be loaded\n");
        return 0;
    }
    imgresized = (unsigned char*) malloc(3*target_w*target_h);
    if(!imgresized)
    {
        free(img);
        perror("malloc");
        return 0;
    }
    //std::cout << "img: " << img << std::endl;
    stbir_resize_uint8(img, ori_w, ori_h, 0, imgresized, target_w, target_h, 0, 3);
    free(img);
    imgfp32 = (float*) malloc(sizeof(*imgfp32) * target_w * target_h * 3);
    if(!imgfp32)
    {
        free(imgresized);
        perror("malloc");
        return 0;
    }
    for(i = 0; i < target_w * target_h * 3; i++)
        imgfp32[i] = imgresized[i];
    free(imgresized);
    imgfp16 = (half*) malloc(sizeof(*imgfp16) * target_w * target_h * 3);
    if(!imgfp16)
    {
        free(imgfp32);
        perror("malloc");
        return 0;
    }
    //adjust values to range between -1.0 and + 1.0
    //change color channel
    for(i = 0; i < target_w*target_h; i++)
    {
        float blue, green, red;
        blue = imgfp32[3*i+2];
        green = imgfp32[3*i+1];
        red = imgfp32[3*i+0];

        imgfp32[3*i+0] = (blue-mean[0]) * 0.007843;
        imgfp32[3*i+1] = (green-mean[1]) * 0.007843;
        imgfp32[3*i+2] = (red-mean[2]) * 0.007843;

        // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
        //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
    }

    floattofp16((unsigned char *)imgfp16, imgfp32, 3*target_w*target_h);
    free(imgfp32);
    return imgfp16;
}