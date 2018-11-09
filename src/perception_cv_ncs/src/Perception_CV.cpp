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

namespace perception_cv {

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
        ncFifoDestroy(&inFifoHandlePtr_seg);
        ncFifoDestroy(&outFifoHandlePtr_seg);
        ncFifoDestroy(&inFifoHandlePtr_det);
        ncFifoDestroy(&outFifoHandlePtr_det);
        ncGraphDestroy(&graphHandlePtr_seg);
        ncGraphDestroy(&graphHandlePtr_det);
        ncDeviceClose(deviceHandlePtr);
        ncDeviceDestroy(&deviceHandlePtr);

        inferThread_.join();
    }


    //  init movidius:open ncs device, creat graph and read in a graph
    void Perception_CV::init_ncs() {

        // load param
        bool flip_flag;
        float thresh;
        std::string graphPath;
        std::string graphModelSeg;
        std::string graphModelDet;
        GRAPH_FILE_NAME_SEG = new char[graphPath.length() + 1];
        GRAPH_FILE_NAME_DET = new char[graphPath.length() + 1];

        nodeHandle_.param("backbone_graph/graph_file/seg_graph_name", graphModelSeg, std::string("seg_ncs_v2.graph"));
        nodeHandle_.param("backbone_graph/graph_file/det_graph_name", graphModelDet, std::string("det_ncs_v2.graph"));
        nodeHandle_.param("graph_path", graphPath, std::string("graph"));
        nodeHandle_.param("backbone_graph/networkDim", networkDim, 300);
        nodeHandle_.param("backbone_graph/target_h", target_h, 300);
        nodeHandle_.param("backbone_graph/target_w", target_w, 300);
        nodeHandle_.param("camera/image_flip", flip_flag, false);
        nodeHandle_.param("ssd_model/detection_classes/names", classLabels_, std::vector<std::string>(0));
        nodeHandle_.param("ssd_model/threshold/value", thresh, (float) 0.3);

        numClasses_ = classLabels_.size();
        ssd_threshold = thresh;

        strcpy(GRAPH_FILE_NAME_SEG, (graphPath + "/" + graphModelSeg).c_str());
        strcpy(GRAPH_FILE_NAME_DET, (graphPath + "/" + graphModelDet).c_str());

        // Try to create the first Neural Compute device (at index zero)
        retCodeSeg = ncDeviceCreate(0, &deviceHandlePtr);
        if (retCodeSeg != NC_OK)
        {   // failed to create the device.
            printf("Could not create NC device\n");
            exit(-1);
        }

        // deviceHandle is created and ready to be opened
        retCodeSeg = ncDeviceOpen(deviceHandlePtr);
        if (retCodeSeg != NC_OK)
        {   // failed to open the device.
            printf("Could not open NC device\n");
            exit(-1);
        }

        // The device is open and ready to be used.
        // Pass it to other NC API calls as needed and close and destroy it when finished.
        printf("Successfully opened NC device!\n");

        // Create the graph
        retCodeSeg = ncGraphCreate("Mobilenet Seg Graph", &graphHandlePtr_seg);
        retCodeDet = ncGraphCreate("Mobilenet Detection Graph", &graphHandlePtr_det);

        if (retCodeSeg != NC_OK)
        {   // error allocating graph
            printf("Could not create graph.\n");
            printf("Error from ncGraphCreate is: %d\n", retCodeSeg);
        }else { // successfully created graph.  Now we need to destory it when finished with it.
            // Now we need to allocate graph and create and in/out fifos
            inFifoHandlePtr_seg = NULL;
            outFifoHandlePtr_seg = NULL;
            inFifoHandlePtr_det = NULL;
            outFifoHandlePtr_det = NULL;

            // Now read in a graph file from disk to memory buffer and
            // then allocate the graph based on the file we read
            void* graphFileBuf_seg = LoadFile(GRAPH_FILE_NAME_SEG, &graphFileLenSeg);
            retCodeSeg = ncGraphAllocateWithFifos(deviceHandlePtr, graphHandlePtr_seg, graphFileBuf_seg, graphFileLenSeg, &inFifoHandlePtr_seg, &outFifoHandlePtr_seg);

            void* graphFileBuf_det = LoadFile(GRAPH_FILE_NAME_DET, &graphFileLenDet);
            retCodeDet = ncGraphAllocateWithFifos(deviceHandlePtr, graphHandlePtr_det, graphFileBuf_det, graphFileLenDet, &inFifoHandlePtr_det, &outFifoHandlePtr_det);

            free(graphFileBuf_seg);
            free(graphFileBuf_det);

            if (retCodeSeg != NC_OK)
            {   // error allocating graph or fifos
                printf("Could not allocate seg graph with fifos.\n");
                printf("Error from ncGraphAllocateWithFifos is: %d\n", retCodeSeg);
            }else{
                // Now graphHandle is ready to go we it can now process inferences.
                printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME_SEG);
            }

            if (retCodeDet != NC_OK)
            {   // error allocating graph or fifos
                printf("Could not allocate detection graph with fifos.\n");
                printf("Error from ncGraphAllocateWithFifos is: %d\n", retCodeDet);
            }else{
                // Now graphHandle is ready to go we it can now process inferences.
                printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME_DET);
            }

        }
    }

    // 对ros节点进行初始化
    void Perception_CV::init() {
        ROS_INFO("[Perception_CV] init().");

        // load param
        std::string cameraTopicName;
        std::string segTopicName;
        nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/camera/image"));
        nodeHandle_.param("subscribers/seg_image/topic", segTopicName, std::string("/seg_ros/seg_image"));

        // infer thread
        inferThread_ = std::thread(&Perception_CV::infer, this);

        imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, 1, &Perception_CV::imageCallback, this);
        imageSegPub_ = imageTransport_.advertise(segTopicName, 1);

    }


    // imageCallback
    void Perception_CV::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
//        ROS_DEBUG("[infer:callback] image received.");
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
                    cvFlip(frame, NULL, 0);
                camImageCopy_ = cv::cvarrToMat(frame, true);

            }
            {
                imageStatus_ = true;
            }
        }else{
            imageStatus_ = false;
        }
        return;
    }


    // control the infer thread
    void Perception_CV::infer() {
        const auto wait_duration = std::chrono::milliseconds(2000);
        //Waiting for image
        while (!getImageStatus()) {
            printf("Waiting for image.\n");
            if (!isNodeRunning()) {
                return;
            }
            std::this_thread::sleep_for(wait_duration);
        }

        std::thread seg_thread;
        std::thread det_thread;

        srand(2222222);

        int count = 0;
        demoTime_ = getWallTime();

        while (!demoDone_) {

            ////convert the image format used by movidius
            cv::Mat ROS_img = getCVImage();
            cv::resize(ROS_img, ROS_img_resized, cv::Size(300, 300), 0, 0, CV_INTER_LINEAR);
            // Now graphHandle is ready to  process inferences. assumption  floats are single percision 32 bit.
            unsigned char *img = cvMat_to_charImg(ROS_img);
            imageBufFP32Ptr = LoadImage32(img, target_w, target_h, ROS_img.cols, ROS_img.rows, networkMean);


//            seg_thread = std::thread(&Perception_CV::segThread, this);
            det_thread = std::thread(&Perception_CV::detThread, this);

            if (count % 1 == 0) {
                fps_ = 1. / (getWallTime() - demoTime_);
                demoTime_ = getWallTime();
            }

            publishThread();

//            seg_thread.join();
            det_thread.join();


            if (!isNodeRunning()) {
                demoDone_ = true;
            }
        }

    }

    //  movidius segmentation inference thread
    void *Perception_CV::segThread()
    {

        unsigned int tensorSize = 0;  /* size of image buffer should be: sizeof(float) * reqsize * reqsize * 3;*/
        tensorSize = sizeof(float) * networkDim * networkDim * 3;

        // queue the inference to start, when its done the result will be placed on the output fifo
        retCodeSeg = ncGraphQueueInferenceWithFifoElem(
                graphHandlePtr_seg, inFifoHandlePtr_seg, outFifoHandlePtr_seg, imageBufFP32Ptr, &tensorSize, NULL);

        if (retCodeSeg != NC_OK)
        {   // error queuing input tensor for inference
            printf("Could not queue seg inference\n");
            printf("Error from ncGraphQueueInferenceWithFifoElem is: %d\n", retCodeSeg);
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
            ncFifoGetOption(outFifoHandlePtr_seg,  NC_RO_FIFO_ELEMENT_DATA_SIZE, &outFifoElemSize, &optionSize);

            float* resultDataFP32Ptr = (float*) malloc(outFifoElemSize);
            void* UserParamPtr = NULL;

            // read the output of the inference.  this will be in FP32 since that is how the
            // fifos are created by default.
            retCodeSeg = ncFifoReadElem(outFifoHandlePtr_seg, (void*)resultDataFP32Ptr, &outFifoElemSize, &UserParamPtr);
            if (retCodeSeg == NC_OK)
            {   // Successfully got the inference result.
                // The inference result is in the buffer pointed to by resultDataFP32Ptr
                printf("Successfully got the inference result for image\n");
                int numResults = outFifoElemSize/(int)sizeof(float);

                printf("resultData is %d bytes which is %d 32-bit floats.\n", outFifoElemSize, numResults);

                //post process
                cv::Mat mask = seg_result_process(resultDataFP32Ptr, target_h, target_w);

                //get masked image
                double alpha = 0.7;
                cv::addWeighted(ROS_img_resized, alpha, mask, 1 - alpha, 0.0, seg_out_img);

            }
            delete imageBufFP32Ptr;
            free((void*)resultDataFP32Ptr);
            return 0;
        }
    }


    //  movidius detection inference thread
    void *Perception_CV::detThread()
    {

        unsigned int tensorSizeDet = 0;  /* size of image buffer should be: sizeof(float) * reqsize * reqsize * 3;*/
        tensorSizeDet = sizeof(float) * networkDim * networkDim * 3;

        // queue the inference to start, when its done the result will be placed on the output fifo
        retCodeDet = ncGraphQueueInferenceWithFifoElem(
                graphHandlePtr_det, inFifoHandlePtr_det, outFifoHandlePtr_det, imageBufFP32Ptr, &tensorSizeDet, NULL);

        if (retCodeDet != NC_OK)
        {   // error queuing input tensor for inference
            printf("Could not queue detection inference\n");
            printf("Error from ncGraphQueueInferenceWithFifoElem is: %d\n", retCodeDet);
        }
        else
        {
            // the inference has been started, now read the output queue for the inference result
            printf("---------------Successfully queued the detection inference for image-----------\n");

            // get the size required for the output tensor.  This depends on the  network definition as well as the output fifo's data type.
            // if the network outputs 1000 tensor elements and the fifo  is using FP32 (float) as the data type then we need a buffer of
            // sizeof(float) * 1000 into which we can read the inference results.  Rather than calculate this size we can also query the fifo itself
            // for this size with the fifo option NC_RO_FIFO_ELEMENT_DATA_SIZE.
            unsigned int outFifoElemSize = 0;
            unsigned int optionSize = sizeof(outFifoElemSize);
            ncFifoGetOption(outFifoHandlePtr_det,  NC_RO_FIFO_ELEMENT_DATA_SIZE, &outFifoElemSize, &optionSize);

            float* resultDataFP32Ptr = (float*) malloc(outFifoElemSize);
            void* UserParamPtr = NULL;

            // read the output of the inference.  this will be in FP32 since that is how the fifos are created by default.
            retCodeDet = ncFifoReadElem(outFifoHandlePtr_det, (void*)resultDataFP32Ptr, &outFifoElemSize, &UserParamPtr);
            if (retCodeDet == NC_OK)
            {   // Successfully got the inference result.
                // The inference result is in the buffer pointed to by resultDataFP32Ptr
                printf("----------Successfully got the detection inference result for image------------\n");
                int numResults = outFifoElemSize/(int)sizeof(float);
                printf("---------resultData is %d bytes which is %d 32-bit floats-----------\n", outFifoElemSize, numResults);

                //post process
                std::vector <Bbox> resultBoxes;
//                ssd_result_process(resultDataFP32Ptr, resultBoxes, ROS_img_resized, numClasses_);
                printf("FPS:%.1f\n", fps_);


            }
//            delete imageBufFP32Ptr;
            free((void*)resultDataFP32Ptr);
            return 0;
        }
    }


    void *Perception_CV::publishThread() {
        // publish topic
        sensor_msgs::ImagePtr msg_seg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ROS_img_resized).toImageMsg();
        imageSegPub_.publish(msg_seg);

        return 0;
    }

    cv::Mat Perception_CV::getCVImage() {
        cv::Mat ROS_img;
        ROS_img = camImageCopy_;
        return ROS_img;
    }

    bool Perception_CV::getImageStatus(void) {
        return imageStatus_;
    }

    bool Perception_CV::isNodeRunning(void) {
        return isNodeRunning_;
    }

}
