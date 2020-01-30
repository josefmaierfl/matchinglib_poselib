//
// Created by maierj on 04.03.19.
//

#include "side_funcs.h"

#include <opencv2/core/eigen.hpp>
#include "opencv2/imgproc/imgproc.hpp"

//#include <pcl/common/common.h>
#include "pcl/common/transforms.h"

#include "helper_funcs.h"

using namespace std;
using namespace cv;

void gen_palette(int num_labels, std::vector<cv::Vec3b> &pallete) {
    const float addHue = sqrt(0.1f); //use an irrational number to achieve many different hues
    float currHue = 0.0f;

    for (int k = 0; k < num_labels; ++k) {
        unsigned char R = 0, G = 0, B = 0;
        float H = currHue - floor(currHue);
        float V = 0.75f + 0.25f * ((float) (k % 4) / 3.f);
        color_HSV2RGB(H, V, V, R, G, B);
        cv::Vec3b col = cv::Vec3b(R, G, B);
        pallete.push_back(col);
        currHue += addHue;
    }
}

void color_HSV2RGB(float H, float S, float V, unsigned char &R, unsigned char &G, unsigned char &B) {
    if (S == 0)                       //HSV values = 0 ÷ 1
    {
        R = (unsigned char) round(min(V * 255.f, 255.f));
        G = R;
        B = R;
    } else {
        float var_h, var_1, var_2, var_3, var_r, var_g, var_b;
        int var_i;

        var_h = H * 6.0f;

        if (var_h == 6.0f)
            var_h = 0;      // H must be < 1

        var_i = int(var_h);     // Or ... var_i = floor( var_h )
        var_1 = V * (1 - S);
        var_2 = V * (1 - S * (var_h - var_i));
        var_3 = V * (1 - S * (1 - (var_h - var_i)));

        if (var_i == 0) {
            var_r = V;
            var_g = var_3;
            var_b = var_1;
        } else if (var_i == 1) {
            var_r = var_2;
            var_g = V;
            var_b = var_1;
        } else if (var_i == 2) {
            var_r = var_1;
            var_g = V;
            var_b = var_3;
        } else if (var_i == 3) {
            var_r = var_1;
            var_g = var_2;
            var_b = V;
        } else if (var_i == 4) {
            var_r = var_3;
            var_g = var_1;
            var_b = V;
        } else {
            var_r = V;
            var_g = var_1;
            var_b = var_2;
        }

        R = (unsigned char) round(min((var_r * 255.f), 255.f));    //RGB results = 0 ÷ 255
        G = (unsigned char) round(min((var_g * 255.f), 255.f));
        B = (unsigned char) round(min((var_b * 255.f), 255.f));
    }
}

void buildColorMapHSV2RGB(const cv::Mat &in16, cv::Mat &rgb8, uint16_t nrLabels, cv::InputArray mask) {
    std::vector<cv::Vec3b> pallete;
    gen_palette(nrLabels, pallete);
    Mat mask_;
    if (!mask.empty()) {
        mask_ = mask.getMat();
    } else {
        mask_ = Mat::ones(in16.size(), CV_8UC1);
    }

    rgb8 = Mat::zeros(in16.size(), CV_8UC3);
    for (int y = 0; y < in16.rows; ++y) {
        for (int x = 0; x < in16.cols; ++x) {
            if (mask_.at<uint8_t>(y, x) > 0) {
                uint16_t lnr = in16.at<uint16_t>(y, x);
                rgb8.at<cv::Vec3b>(y, x) = pallete[lnr];
            }
        }
    }
}

void startPCLViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer) {
    //--------------------
    // -----Main loop-----
    //--------------------
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    viewer->close();
}

void setPCLViewerCamPars(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                         Eigen::Matrix4f cam_extrinsics,
                         const cv::Mat &K1) {
    //Eigen::Matrix4f cam_extrinsics(m.matrix());
    Eigen::Matrix3f zRotPi;
    zRotPi << -1.f, 0, 0,
            0, -1.f, 0,
            0, 0, 1.f;
    cam_extrinsics.block<3, 3>(0, 0) = cam_extrinsics.block<3, 3>(0, 0) * zRotPi;
    Eigen::Matrix3d cam_intrinsicsd;
    Eigen::Matrix3f cam_intrinsics;
    cv::cv2eigen(K1, cam_intrinsicsd);
    cam_intrinsics = cam_intrinsicsd.cast<float>();
    viewer->setCameraParameters(cam_intrinsics, cam_extrinsics);
}

Eigen::Affine3f initPCLViewerCoordinateSystems(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                                               cv::InputArray R_C2W,
                                               cv::InputArray t_C2W) {
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(5.0);

    Eigen::Affine3f m;
    if (!R_C2W.empty() && !t_C2W.empty()) {
        m = addVisualizeCamCenter(viewer, R_C2W.getMat(), t_C2W.getMat());
    }

    return m;
}

void getNColors(cv::OutputArray colorMat, size_t nr_Colors, int colormap) {
    Mat colors = Mat(nr_Colors, 1, CV_8UC1);
    unsigned char addc = nr_Colors > 255 ? (unsigned char)255 : (unsigned char) nr_Colors;
    addc = addc < (unsigned char)2 ? (unsigned char)255 : ((unsigned char)255 / (addc - (unsigned char)1));
    colors.at<unsigned char>(0) = 0;
    for (int k = 1; k < (int)nr_Colors; ++k) {
        colors.at<unsigned char>(k) = colors.at<unsigned char>(k - 1) + addc;
    }
    applyColorMap(colors, colorMat, colormap);
}

void getCloudCentroids(std::vector<pcl::PointCloud<pcl::PointXYZ>> &pointclouds,
                       std::vector<pcl::PointXYZ> &cloudCentroids) {
    cloudCentroids.reserve(pointclouds.size());
    for (auto& i : pointclouds) {
        pcl::PointXYZ point;
        getCloudCentroid(i, point);
        cloudCentroids.push_back(point);
    }
}

void getCloudCentroids(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &pointclouds,
                       std::vector<pcl::PointXYZ> &cloudCentroids) {
    cloudCentroids.reserve(pointclouds.size());
    for (auto& i : pointclouds) {
        pcl::PointXYZ point;
        getCloudCentroid(*i.get(), point);
        cloudCentroids.push_back(point);
    }
}

void getCloudCentroid(pcl::PointCloud<pcl::PointXYZ> &pointcloud, pcl::PointXYZ &cloudCentroid) {
    Eigen::Matrix<float, 4, 1> pm;
    pcl::compute3DCentroid(pointcloud, pm);
    cloudCentroid.x = pm(0);
    cloudCentroid.y = pm(1);
    cloudCentroid.z = pm(2);
}

void getMeanCloudStandardDevs(std::vector<pcl::PointCloud<pcl::PointXYZ>> &pointclouds,
                              std::vector<float> &cloudExtensions,
                              std::vector<pcl::PointXYZ> &cloudCentroids) {
    cloudExtensions.reserve(pointclouds.size());
    for (size_t i = 0; i < pointclouds.size(); ++i) {
        float cloudExtension = 0;
        getMeanCloudStandardDev(pointclouds[i], cloudExtension, cloudCentroids[i]);
        cloudExtensions.push_back(cloudExtension);
    }
}

void getMeanCloudStandardDev(pcl::PointCloud<pcl::PointXYZ> &pointcloud, float &cloudExtension,
                             pcl::PointXYZ &cloudCentroid) {
    pcl::PointXYZ cloudDim;
    getCloudDimensionStdDev(pointcloud, cloudDim, cloudCentroid);
    cloudExtension = (cloudDim.x + cloudDim.y + cloudDim.z) / 3.f;
}

Eigen::Affine3f addVisualizeCamCenter(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                                      const cv::Mat &R,
                                      const cv::Mat &t) {
    Eigen::Affine3f m;
    m.setIdentity();
    Eigen::Vector3d te;
    Eigen::Matrix3d Re;
    cv::cv2eigen(R, Re);
    cv::cv2eigen(t, te);
    m.matrix().block<3, 3>(0, 0) = Re.cast<float>();
    m.matrix().block<3, 1>(0, 3) = te.cast<float>();
    viewer->addCoordinateSystem(1.0, m);

    return m;
}

void getCloudDimensionStdDev(pcl::PointCloud<pcl::PointXYZ> &pointcloud, pcl::PointXYZ &cloudDim,
                             pcl::PointXYZ &cloudCentroid) {
    Eigen::Matrix<float, 4, 1> pm;
    Eigen::Matrix<float, 3, 3> covariance_matrix;
    pm << cloudCentroid.x, cloudCentroid.y, cloudCentroid.z, 1.f;
    pcl::computeCovarianceMatrixNormalized(pointcloud, pm, covariance_matrix);
    cloudDim.x = sqrt(covariance_matrix(0, 0));
    cloudDim.y = sqrt(covariance_matrix(1, 1));
    cloudDim.z = sqrt(covariance_matrix(2, 2));
}

void getSecPartContourPos(std::vector<cv::Point> &target, std::vector<cv::Point> &source, int idxStart, int idxEnd) {
    std::vector<cv::Point>::iterator a1begin, a1end;
    if (idxStart == idxEnd) {
        idxEnd++;
    }
    idxStart++;
    if (idxStart >= (int)source.size()) {
        a1end = source.end();
    } else {
        a1end = source.begin() + idxStart;
    }
    a1begin = source.begin();
    target.insert(target.end(), a1begin, a1end);
    std::reverse(target.begin(), target.end());
    vector<Point> secPosib;
    if (idxEnd < (int) source.size()) {
        a1begin = source.begin() + idxEnd;
        a1end = source.end();
        secPosib.insert(secPosib.end(), a1begin, a1end);
        std::reverse(secPosib.begin(), secPosib.end());
        target.insert(target.end(), secPosib.begin(), secPosib.end());
    }
}

void getSecPartContourNeg(std::vector<cv::Point> &target, std::vector<cv::Point> &source, int idxStart, int idxEnd) {
    std::vector<cv::Point>::iterator a1begin, a1end;
    if (idxStart == idxEnd) {
        idxStart++;
    }
    if (idxStart < (int) source.size()) {
        a1begin = source.begin() + idxStart;
        a1end = source.end();
        target.insert(target.end(), a1begin, a1end);
    }
    idxEnd++;
    a1begin = source.begin();
    a1end = source.begin() + idxEnd;
    target.insert(target.end(), a1begin, a1end);
}

void getFirstPartContourPos(std::vector<cv::Point> &target, std::vector<cv::Point> &source, int idxStart, int idxEnd) {
    std::vector<cv::Point>::iterator a1begin, a1end;
    idxEnd++;
    if (idxEnd >= (int)source.size()) {
        a1end = source.end();
    } else {
        a1end = source.begin() + idxEnd;
    }
    a1begin = source.begin() + idxStart;
    target.insert(target.end(), a1begin, a1end);
}

void getFirstPartContourNeg(std::vector<cv::Point> &target, std::vector<cv::Point> &source, int idxStart, int idxEnd) {
    std::vector<cv::Point>::iterator a1begin, a1end;
    idxStart++;
    if (idxStart >= (int)source.size()) {
        a1end = source.end();
    } else {
        a1end = source.begin() + idxStart;
    }
    a1begin = source.begin() + idxEnd;
    target.insert(target.end(), a1begin, a1end);
    std::reverse(target.begin(), target.end());
}

bool checkPointValidity(const cv::Mat &mask, const cv::Point_<int32_t> &pt){
    if(mask.at<unsigned char>(pt) > 0){
        return true;
    }

    return false;
}

bool getValidRegBorders(const cv::Mat &mask, cv::Rect &validRect){
    if(countNonZero(mask) < 100)
        return false;
    Mat mask_tmp = mask.clone();
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours(mask_tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    validRect = cv::boundingRect(contours[0]);

    if(validRect.width < (int)floor(0.2 * (double)mask.cols))
        return false;

    if(validRect.height < (int)floor(0.2 * (double)mask.rows))
        return false;

    return true;
}

//Search for the corresponding index entries and delete it
int deletedepthCatsByIdx(std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> &seedsFromLast,
                         std::vector<size_t> &delListCorrs,
                         const cv::Mat &ptMat){
    cv::Point_<int32_t> pt;
    int nrDel = 0;
    vector<size_t> delList;
    for (size_t j = 0; j < delListCorrs.size(); ++j) {
        int idx = (int)delListCorrs[j];
        pt = cv::Point_<int32_t>((int32_t)round(ptMat.at<double>(0,idx)),
                                 (int32_t)round(ptMat.at<double>(1,idx)));
        bool found = false;
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                for (size_t i = 0; i < seedsFromLast[y][x].size(); ++i) {
                    int32_t diff = abs(seedsFromLast[y][x][i].x - pt.x) + abs(seedsFromLast[y][x][i].y - pt.y);
                    if(diff == 0){
                        seedsFromLast[y][x].erase(seedsFromLast[y][x].begin() + i);
                        delList.push_back(j);
                        nrDel++;
                        found  = true;
                        break;
                    }
                }
                if(found) break;
            }
            if(found) break;
        }
    }

    for (int k = (int)delList.size() - 1; k >= 0; --k) {
        delListCorrs.erase(delListCorrs.begin() + delList[k]);
    }

    return nrDel;
}

int deletedepthCatsByNr(std::vector<cv::Point_<int32_t>> &seedsFromLast,
                        int32_t nrToDel,
                        const cv::Mat &ptMat,
                        std::vector<size_t> &delListCorrs){
    delListCorrs.clear();
    if(nrToDel <= 0) return 0;
    int nrToDel_tmp = (int)nrToDel;
    std::vector<size_t> delList(seedsFromLast.size());
    std::iota(delList.begin(), delList.end(), 0);

    if(nrToDel_tmp < (int)seedsFromLast.size()){
        std::shuffle(delList.begin(), delList.end(), std::mt19937{std::random_device{}()});
        delList.erase(delList.begin() + nrToDel_tmp, delList.end());
        sort(delList.begin(), delList.end(), [](size_t first, size_t second){return first < second;});
    }
    else{
        nrToDel_tmp = (int)seedsFromLast.size();
    }

    cv::Point_<int32_t> pts, ptg;
    for (int32_t i = 0; i < nrToDel_tmp; ++i) {
        ptg = seedsFromLast[delList[i]];
        for (size_t j = 0; j < (size_t)ptMat.cols; ++j) {
            pts = cv::Point_<int32_t>((int32_t)round(ptMat.at<double>(0, (int)j)),
                                      (int32_t)round(ptMat.at<double>(1, (int)j)));
            int32_t diff = abs(ptg.x - pts.x) + abs(ptg.y - pts.y);
            if(diff == 0){
                delListCorrs.push_back(j);
                break;
            }
        }
    }
    sort(delListCorrs.begin(), delListCorrs.end(), [](size_t first, size_t second){return first < second;});

    deleteVecEntriesbyIdx(seedsFromLast, delList);

    if(nrToDel_tmp != (int)delListCorrs.size()){
        cout << "Could not find every backprojected keypoint in the given array!" << endl;
    }

    return nrToDel_tmp;
}

int deletedepthCatsByNr(std::vector<cv::Point2d> &seedsFromLast,
                        int32_t nrToDel,
                        const cv::Mat &ptMat,
                        std::vector<size_t> &delListCorrs){
    delListCorrs.clear();
    if(nrToDel <= 0) return 0;
    int nrToDel_tmp = (int)nrToDel;
    std::vector<size_t> delList(seedsFromLast.size());
    std::iota(delList.begin(), delList.end(), 0);

    if(nrToDel_tmp < (int)seedsFromLast.size()){
        std::shuffle(delList.begin(), delList.end(), std::mt19937{std::random_device{}()});
        delList.erase(delList.begin() + nrToDel_tmp, delList.end());
        sort(delList.begin(), delList.end(), [](size_t first, size_t second){return first < second;});
    }
    else{
        nrToDel_tmp = (int)seedsFromLast.size();
    }

    cv::Point2d pts, ptg;
    for (int32_t i = 0; i < nrToDel_tmp; ++i) {
        ptg = seedsFromLast[delList[i]];
        for (int j = 0; j < ptMat.cols; ++j) {
            pts = cv::Point2d(ptMat.at<double>(0, j), ptMat.at<double>(1, j));
            double diff = abs(ptg.x - pts.x) + abs(ptg.y - pts.y);
            if(nearZero(diff)){
                delListCorrs.push_back((size_t)j);
                break;
            }
        }
    }
    sort(delListCorrs.begin(), delListCorrs.end(), [](size_t first, size_t second){return first < second;});

    deleteVecEntriesbyIdx(seedsFromLast, delList);

    if(nrToDel_tmp != (int)delListCorrs.size()){
        cout << "Could not find every backprojected keypoint in the given array!" << endl;
    }

    return nrToDel_tmp;
}

/*Rounds a rotation matrix to its nearest integer values and checks if it is still a rotation matrix and does not change more than 22.5deg from the original rotation matrix.
As an option, the error of the rounded rotation matrix can be compared to an angular difference of a second given rotation matrix R_fixed to R_old.
The rotation matrix with the smaller angular difference is selected.
This function is used to select a proper rotation matrix if the "look at" and "up vector" are nearly equal. I trys to find the nearest rotation matrix aligened to the
"look at" vector taking into account the rotation matrix calculated from the old/last "look at" vector
*/
bool roundR(const cv::Mat R_old, cv::Mat &R_round, cv::InputArray R_fixed) {
    R_round = roundMat(R_old);
    if (!isMatRotationMat(R_round)) {
        return false;
    }

    double rd = abs(RAD2DEG(rotDiff(R_old, R_round)));
    double rfd = 360.0;

    if (!R_fixed.empty()) {
        Mat Rf = R_fixed.getMat();
        rfd = abs(RAD2DEG(rotDiff(Rf, R_old)));

        if (rfd < rd) {
            Rf.copyTo(R_round);
            return true;
        }
    }

    if (rd < 22.5) {
        return true;
    }

    return false;
}