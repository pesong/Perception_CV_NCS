//
// Created by pesong on 18-9-5.
//
#include "Perception_CV.hpp"

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "perception_cv_ncs");
    ros::NodeHandle nh, priv_nh("~");

    perception_cv_ncs::Perception_CV perception_cv(priv_nh);
    ros::spin();
}