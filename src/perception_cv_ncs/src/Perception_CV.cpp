#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <stdlib.h>
#include <mvnc.h>
#include "ncs_util.h"
#include "Perception_CV.hpp"

// Check for xServer
#include <X11/Xlib.h>


namespace perception_cv_ncs {

    Perception_CV::Perception_CV(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_){

        ROS_INFO("[SSD_Detector] Node started.");

        init_ncs();
        init();

    }

    Perception_CV::~Perception_CV() {
        {
            isNodeRunning_ = false;
        }
        //clear and close ncs
        ROS_INFO("Delete movidius SSD graph");

        ncFifoDestroy(&inFifoHandlePtr);
        ncFifoDestroy(&outFifoHandlePtr);
        ncGraphDestroy(&graphHandlePtr);
        ncDeviceClose(deviceHandlePtr);
        ncDeviceDestroy(&deviceHandlePtr);

        segThread_.join();
    }


    //  init movidius:open ncs device, creat graph and read in a graph
    void Perception_CV::init_ncs() {

        // load param
        std::string graphPath;
        std::string graphModel;
        bool flip_flag;
        nodeHandle_.param("seg_mobilenetv1/graph_file/name", graphModel, std::string("seg_ncs_v2.graph"));
        nodeHandle_.param("graph_path", graphPath, std::string("graph"));
        graphPath += "/" + graphModel;
        GRAPH_FILE_NAME = new char[graphPath.length() + 1];
        strcpy(GRAPH_FILE_NAME, graphPath.c_str());
        nodeHandle_.param("seg_mobilenetv1/networkDim", networkDim, 300);
        nodeHandle_.param("seg_mobilenetv1/target_h", target_h, 300);
        nodeHandle_.param("seg_mobilenetv1/target_w", target_w, 300);
        nodeHandle_.param("camera/image_flip", flip_flag, false);


        // Try to create the first Neural Compute device (at index zero)
        retCode = ncDeviceCreate(0, &deviceHandlePtr);
        if (retCode != NC_OK)
        {   // failed to create the device.
            printf("Could not create NC device\n");
            exit(-1);
        }

        // deviceHandle is created and ready to be opened
        retCode = ncDeviceOpen(deviceHandlePtr);
        if (retCode != NC_OK)
        {   // failed to open the device.
            printf("Could not open NC device\n");
            exit(-1);
        }

        // The device is open and ready to be used.
        // Pass it to other NC API calls as needed and close and destroy it when finished.
        printf("Successfully opened NC device!\n");

        // Create the graph
        retCode = ncGraphCreate("GoogLeNet Graph", &graphHandlePtr);
        if (retCode != NC_OK)
        {   // error allocating graph
            printf("Could not create graph.\n");
            printf("Error from ncGraphCreate is: %d\n", retCode);
        }else { // successfully created graph.  Now we need to destory it when finished with it.
            // Now we need to allocate graph and create and in/out fifos
            inFifoHandlePtr = NULL;
            outFifoHandlePtr = NULL;

            // Now read in a graph file from disk to memory buffer and
            // then allocate the graph based on the file we read
            void* graphFileBuf = LoadFile(GRAPH_FILE_NAME, &graphFileLen);
            retCode = ncGraphAllocateWithFifos(deviceHandlePtr, graphHandlePtr, graphFileBuf, graphFileLen, &inFifoHandlePtr, &outFifoHandlePtr);
            free(graphFileBuf);

            if (retCode != NC_OK)
            {   // error allocating graph or fifos
                printf("Could not allocate graph with fifos.\n");
                printf("Error from ncGraphAllocateWithFifos is: %d\n", retCode);
            }else{
                // Now graphHandle is ready to go we it can now process inferences.
                printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME);
            }
        }
    }

    // 对ros节点进行初始化
    void Perception_CV::init() {
        ROS_INFO("[Perception_CV] init().");

        // load param
        std::string cameraTopicName;
        std::string segTopicName;
        nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                            std::string("/camera/image"));
        nodeHandle_.param("subscribers/seg_image/topic", segTopicName,
                          std::string("/seg_ros/seg_image"));

        // seg thread
        segThread_ = std::thread(&Perception_CV::seg, this);

        imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, 1,
                                   &Perception_CV::imageCallback, this);
        imageSegPub_ = imageTransport_.advertise(segTopicName, 1);

    }


    //turn output to mask image
    cv::Mat Perception_CV::ncs_result_process(float* output, int h, int w)
    {

        //    out = out.reshape(-1, 2).T.reshape(2, 331, -1)
        //    out = out.argmax(axis=0)
        //    out = out[:-11, :-11]
        //    for (int i = 20000; i < 20050 ; ++i) {
        //        printf("ncs out: %f", output[i]);
        //    }

//        // the output of graph was bigger than original image
//        int margin = 11;
//        int h_margin = h + margin;
//        int w_margin = w + margin;

        cv::Mat mask_gray(h, w, CV_8UC1);
        cv::Mat mask;

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                if(output[2*(w*i + j)] < output[2*(h*i + j) + 1]){
                    mask_gray.at<uchar>(i,j) = 255;
                } else{
                    mask_gray.at<uchar>(i,j) = 0;
                }
            }
        }
        // gray -> color
        cv::cvtColor(mask_gray, mask, cv::COLOR_GRAY2BGR);

        return mask;

    }

    // callback for inference
    void Perception_CV::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
//        ROS_DEBUG("[Seg:callback] image received.");
        cv_bridge::CvImagePtr cam_image;

        try {
            cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if(cam_image){
            {
                //flip
                cv::Mat image0 = cam_image->image.clone();
                IplImage copy = image0;
                IplImage *frame = &copy;
                //std::cout << "flipFlag: " << flipFlag << std::endl;
                if(flipFlag)
                    cvFlip(frame, NULL, 0); //翻转
                camImageCopy_ = cv::cvarrToMat(frame, true);
            }

            {
                //这里对imageStatus进行赋值操作，为避免另一线程yolo()在此时读取imageStatus_，在赋值前先将其锁住
//                std::cout << "imageStatus_" << std::endl;
                imageStatus_ = true;
            }
        }else{
            imageStatus_ = false;
        }
        return;
    }

    //  movidius 推理
    void *Perception_CV::segThread()
    {

        cv::Mat ROS_img = getCVImage();
        cv::Mat ROS_img_resized;
        cv::resize(ROS_img, ROS_img_resized, cv::Size(300, 300), 0, 0, CV_INTER_LINEAR);

        //// 将cvmat转为movidius使用的image类型

        // Now graphHandle is ready to go we it can now process inferences.
        // assumption here that floats are single percision 32 bit.
        unsigned char *img = cvMat_to_charImg(ROS_img);
        unsigned int tensorSize = 0;  /* size of image buffer should be: sizeof(float) * reqsize * reqsize * 3;*/
        float* imageBufFP32Ptr = LoadImage32(img, target_w, target_h, ROS_img.cols, ROS_img.rows, networkMean);
        tensorSize = sizeof(float) * networkDim * networkDim * 3;

        // std::cout << "networkDim: " << networkDim << " imageBufFp16: " << sizeof(*imageBufFp16) << " lenBufFp16: " << lenBufFp16 << std::endl;
//        std::cout << " imageBufFp16: " << *imageBufFp16 << std::endl;

        // queue the inference to start, when its done the result will be placed on the output fifo
        retCode = ncGraphQueueInferenceWithFifoElem(
                graphHandlePtr, inFifoHandlePtr, outFifoHandlePtr, imageBufFP32Ptr, &tensorSize, NULL);

        if (retCode != NC_OK)
        {   // error queuing input tensor for inference
            printf("Could not queue inference\n");
            printf("Error from ncGraphQueueInferenceWithFifoElem is: %d\n", retCode);
        }
        else
        {
            // the inference has been started, now read the output queue for the inference result
            printf("Successfully queued the inference for image\n");

            // get the size required for the output tensor.  This depends on the  network definition as well as the output fifo's data type.
            // if the network outputs 1000 tensor elements and the fifo  is using FP32 (float) as the data type then we need a buffer of
            // sizeof(float) * 1000 into which we can read the inference results.  Rather than calculate this size we can also query the fifo itself
            // for this size with the fifo option NC_RO_FIFO_ELEMENT_DATA_SIZE.
            unsigned int outFifoElemSize = 0;
            unsigned int optionSize = sizeof(outFifoElemSize);
            ncFifoGetOption(outFifoHandlePtr,  NC_RO_FIFO_ELEMENT_DATA_SIZE, &outFifoElemSize, &optionSize);

            float* resultDataFP32Ptr = (float*) malloc(outFifoElemSize);
            void* UserParamPtr = NULL;

            // read the output of the inference.  this will be in FP32 since that is how the
            // fifos are created by default.
            retCode = ncFifoReadElem(outFifoHandlePtr, (void*)resultDataFP32Ptr, &outFifoElemSize, &UserParamPtr);
            if (retCode == NC_OK)
            {   // Successfully got the inference result.
                // The inference result is in the buffer pointed to by resultDataFP32Ptr
                printf("Successfully got the inference result for image\n");
                int numResults = outFifoElemSize/(int)sizeof(float);

                printf("resultData is %d bytes which is %d 32-bit floats.\n", outFifoElemSize, numResults);

                //post process
                cv::Mat mask = ncs_result_process(resultDataFP32Ptr, target_h, target_w);

                //图像混合
                double alpha = 0.7;
                cv::addWeighted(ROS_img_resized, alpha, mask, 1 - alpha, 0.0, seg_out_img);

            }
            free((void*)resultDataFP32Ptr);
            delete imageBufFP32Ptr;
            return 0;
        }
    }


    void *Perception_CV::publishThread() {
        // 发布topic
        sensor_msgs::ImagePtr msg_seg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", seg_out_img).toImageMsg();
        imageSegPub_.publish(msg_seg);

        return 0;
    }

    void Perception_CV::seg() {
        const auto wait_duration = std::chrono::milliseconds(2000);
        //等待image
        while (!getImageStatus()) {
            printf("Waiting for image.\n");
            if (!isNodeRunning()) {
                return;
            }
            std::this_thread::sleep_for(wait_duration);
        }

        std::thread seg_thread;
        srand(2222222);

        while (!demoDone_) {

            std::cout << getImageStatus() << std::endl;

            seg_thread = std::thread(&Perception_CV::segThread, this);

            publishThread();

            seg_thread.join();

            if (!isNodeRunning()) {
                demoDone_ = true;
            }
        }

    }

    cv::Mat Perception_CV::getCVImage() {
        // std::cout << "getCVImage" << std::endl;
        cv::Mat ROS_img;
        ROS_img = camImageCopy_;
        //camImageCopy_.copyTo(ROS_img);
        // camImageCopy_.release();

        return ROS_img;
    }

    bool Perception_CV::getImageStatus(void) {
        return imageStatus_;
    }

    bool Perception_CV::isNodeRunning(void) {
        return isNodeRunning_;
    }

}
