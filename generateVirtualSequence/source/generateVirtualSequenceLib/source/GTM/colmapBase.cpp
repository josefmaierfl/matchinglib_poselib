//
// Created by maierj on 5/14/20.
//
#include "GTM/colmapBase.h"
//#include <opencv2/core/eigen.hpp>

using namespace colmap;
using namespace std;

bool colmapBase::prepareColMapData(const megaDepthFolders& folders){
    try{
        ReadText(folders.sfmF);
    } catch (colmapException &e) {
        cerr << e.what() << endl;
        return false;
    }
    filterExistingDepth(folders);
    filterInterestingImgs();
    return getCorrespondingImgs();
}

void colmapBase::getMaxDepthImgDim(){
    vector<int> imgSizes;
    for(auto &i: correspImgs_){
        cv::Mat img = cv::imread(i.second.img1, cv::IMREAD_UNCHANGED);
        i.second.imgSize1 = img.size();
        imgSizes.emplace_back(max(i.second.imgSize1.height, i.second.imgSize1.width));
        img = cv::imread(i.second.img2, cv::IMREAD_UNCHANGED);
        i.second.imgSize2 = img.size();
        imgSizes.emplace_back(max(i.second.imgSize2.height, i.second.imgSize2.width));
    }
    maxdepthImgSize = *std::max_element(imgSizes.begin(), imgSizes.end());
}

void colmapBase::getUndistortedScaledDims(){
    getMaxDepthImgDim();
    UndistortCameraOptions options;
    options.max_image_size = maxdepthImgSize;
    for(auto &i: correspImgs_){
        i.second.undistortedCam1 = UndistortCamera(options, Camera(Image(i.first).CameraId()));
        i.second.undistortedCam2 = UndistortCamera(options, Camera(Image(i.second.imgID).CameraId()));
    }
}

bool colmapBase::checkCorrectDimensions(){
    for(auto &i: correspImgs_){
        int diff = static_cast<int>(i.second.undistortedCam1.Width()) - i.second.imgSize1.width;
        diff += static_cast<int>(i.second.undistortedCam1.Height()) - i.second.imgSize1.height;
        diff += static_cast<int>(i.second.undistortedCam2.Width()) - i.second.imgSize2.width;
        diff += static_cast<int>(i.second.undistortedCam2.Height()) - i.second.imgSize2.height;
        if(diff){
            return false;
        }
    }
    return true;
}

Camera colmapBase::UndistortCamera(const UndistortCameraOptions& options,
                                   const class Camera& camera) {
    CHECK_GE(options.blank_pixels, 0);
    CHECK_LE(options.blank_pixels, 1);
    CHECK_GT(options.min_scale, 0.0);
    CHECK_LE(options.min_scale, options.max_scale);
    CHECK_NE(options.max_image_size, 0);
    CHECK_GE(options.roi_min_x, 0.0);
    CHECK_GE(options.roi_min_y, 0.0);
    CHECK_LE(options.roi_max_x, 1.0);
    CHECK_LE(options.roi_max_y, 1.0);
    CHECK_LT(options.roi_min_x, options.roi_max_x);
    CHECK_LT(options.roi_min_y, options.roi_max_y);

    class Camera undistorted_camera;
    undistorted_camera.SetModelId(PinholeCameraModel::model_id);
    undistorted_camera.SetWidth(camera.Width());
    undistorted_camera.SetHeight(camera.Height());

    // Copy focal length parameters.
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    CHECK_LE(focal_length_idxs.size(), 2);//<< "Not more than two focal length parameters supported.";
    if (focal_length_idxs.size() == 1) {
        undistorted_camera.SetFocalLengthX(camera.FocalLength());
        undistorted_camera.SetFocalLengthY(camera.FocalLength());
    } else if (focal_length_idxs.size() == 2) {
        undistorted_camera.SetFocalLengthX(camera.FocalLengthX());
        undistorted_camera.SetFocalLengthY(camera.FocalLengthY());
    }

    // Copy principal point parameters.
    undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX());
    undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY());

    // Modify undistorted camera parameters based on ROI if enabled
    size_t roi_min_x = 0;
    size_t roi_min_y = 0;
    size_t roi_max_x = camera.Width();
    size_t roi_max_y = camera.Height();

    const bool roi_enabled = options.roi_min_x > 0.0 || options.roi_min_y > 0.0 ||
                             options.roi_max_x < 1.0 || options.roi_max_y < 1.0;

    if (roi_enabled) {
        roi_min_x = static_cast<size_t>(
                std::round(options.roi_min_x * static_cast<double>(camera.Width())));
        roi_min_y = static_cast<size_t>(
                std::round(options.roi_min_y * static_cast<double>(camera.Height())));
        roi_max_x = static_cast<size_t>(
                std::round(options.roi_max_x * static_cast<double>(camera.Width())));
        roi_max_y = static_cast<size_t>(
                std::round(options.roi_max_y * static_cast<double>(camera.Height())));

        // Make sure that the roi is valid.
        roi_min_x = std::min(roi_min_x, camera.Width() - 1);
        roi_min_y = std::min(roi_min_y, camera.Height() - 1);
        roi_max_x = std::max(roi_max_x, roi_min_x + 1);
        roi_max_y = std::max(roi_max_y, roi_min_y + 1);

        undistorted_camera.SetWidth(roi_max_x - roi_min_x);
        undistorted_camera.SetHeight(roi_max_y - roi_min_y);

        undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX() -
                                              static_cast<double>(roi_min_x));
        undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY() -
                                              static_cast<double>(roi_min_y));
    }

    // Scale the image such the the boundary of the undistorted image.
    if (roi_enabled || (camera.ModelId() != SimplePinholeCameraModel::model_id &&
                        camera.ModelId() != PinholeCameraModel::model_id)) {
        // Determine min/max coordinates along top / bottom image border.

        double left_min_x = std::numeric_limits<double>::max();
        double left_max_x = std::numeric_limits<double>::lowest();
        double right_min_x = std::numeric_limits<double>::max();
        double right_max_x = std::numeric_limits<double>::lowest();

        for (size_t y = roi_min_y; y < roi_max_y; ++y) {
            // Left border.
            const Eigen::Vector2d world_point1 =
                    camera.ImageToWorld(Eigen::Vector2d(0.5, y + 0.5));
            const Eigen::Vector2d undistorted_point1 =
                    undistorted_camera.WorldToImage(world_point1);
            left_min_x = std::min(left_min_x, undistorted_point1(0));
            left_max_x = std::max(left_max_x, undistorted_point1(0));
            // Right border.
            const Eigen::Vector2d world_point2 =
                    camera.ImageToWorld(Eigen::Vector2d(camera.Width() - 0.5, y + 0.5));
            const Eigen::Vector2d undistorted_point2 =
                    undistorted_camera.WorldToImage(world_point2);
            right_min_x = std::min(right_min_x, undistorted_point2(0));
            right_max_x = std::max(right_max_x, undistorted_point2(0));
        }

        // Determine min, max coordinates along left / right image border.

        double top_min_y = std::numeric_limits<double>::max();
        double top_max_y = std::numeric_limits<double>::lowest();
        double bottom_min_y = std::numeric_limits<double>::max();
        double bottom_max_y = std::numeric_limits<double>::lowest();

        for (size_t x = roi_min_x; x < roi_max_x; ++x) {
            // Top border.
            const Eigen::Vector2d world_point1 =
                    camera.ImageToWorld(Eigen::Vector2d(x + 0.5, 0.5));
            const Eigen::Vector2d undistorted_point1 =
                    undistorted_camera.WorldToImage(world_point1);
            top_min_y = std::min(top_min_y, undistorted_point1(1));
            top_max_y = std::max(top_max_y, undistorted_point1(1));
            // Bottom border.
            const Eigen::Vector2d world_point2 =
                    camera.ImageToWorld(Eigen::Vector2d(x + 0.5, camera.Height() - 0.5));
            const Eigen::Vector2d undistorted_point2 =
                    undistorted_camera.WorldToImage(world_point2);
            bottom_min_y = std::min(bottom_min_y, undistorted_point2(1));
            bottom_max_y = std::max(bottom_max_y, undistorted_point2(1));
        }

        const double cx = undistorted_camera.PrincipalPointX();
        const double cy = undistorted_camera.PrincipalPointY();

        // Scale such that undistorted image contains all pixels of distorted image.
        const double min_scale_x =
                std::min(cx / (cx - left_min_x),
                         (undistorted_camera.Width() - 0.5 - cx) / (right_max_x - cx));
        const double min_scale_y = std::min(
                cy / (cy - top_min_y),
                (undistorted_camera.Height() - 0.5 - cy) / (bottom_max_y - cy));

        // Scale such that there are no blank pixels in undistorted image.
        const double max_scale_x =
                std::max(cx / (cx - left_max_x),
                         (undistorted_camera.Width() - 0.5 - cx) / (right_min_x - cx));
        const double max_scale_y = std::max(
                cy / (cy - top_max_y),
                (undistorted_camera.Height() - 0.5 - cy) / (bottom_min_y - cy));

        // Interpolate scale according to blank_pixels.
        double scale_x = 1.0 / (min_scale_x * options.blank_pixels +
                                max_scale_x * (1.0 - options.blank_pixels));
        double scale_y = 1.0 / (min_scale_y * options.blank_pixels +
                                max_scale_y * (1.0 - options.blank_pixels));

        // Clip the scaling factors.
        scale_x = Clip(scale_x, options.min_scale, options.max_scale);
        scale_y = Clip(scale_y, options.min_scale, options.max_scale);

        // Scale undistorted camera dimensions.
        const size_t orig_undistorted_camera_width = undistorted_camera.Width();
        const size_t orig_undistorted_camera_height = undistorted_camera.Height();
        undistorted_camera.SetWidth(static_cast<size_t>(
                                            std::max(1.0, scale_x * undistorted_camera.Width())));
        undistorted_camera.SetHeight(static_cast<size_t>(
                                             std::max(1.0, scale_y * undistorted_camera.Height())));

        // Scale the principal point according to the new dimensions of the camera.
        undistorted_camera.SetPrincipalPointX(
                undistorted_camera.PrincipalPointX() *
                static_cast<double>(undistorted_camera.Width()) /
                static_cast<double>(orig_undistorted_camera_width));
        undistorted_camera.SetPrincipalPointY(
                undistorted_camera.PrincipalPointY() *
                static_cast<double>(undistorted_camera.Height()) /
                static_cast<double>(orig_undistorted_camera_height));
    }

    if (options.max_image_size > 0) {
        const double max_image_scale_x =
                options.max_image_size /
                static_cast<double>(undistorted_camera.Width());
        const double max_image_scale_y =
                options.max_image_size /
                static_cast<double>(undistorted_camera.Height());
        const double max_image_scale =
                std::min(max_image_scale_x, max_image_scale_y);
        if (max_image_scale < 1.0) {
            undistorted_camera.Rescale(max_image_scale);
        }
    }

    return undistorted_camera;
}

void colmapBase::filterExistingDepth(const megaDepthFolders& folders){
    vector<point2D_t> del_ids;
    for(auto &i: images_){
        string imgName = i.second.Name();
        std::string root, ext;
        SplitFileExtension(imgName, &root, &ext);
        string depthName = JoinPaths(folders.mdDepth, root + folders.depthExt);
        if(!ExistsFile(depthName)) {
            point2D_t idx1 = 0;
            for (auto &pt2d: i.second.Points2D()) {
                if (pt2d.HasPoint3D()) {
                    points3D_[pt2d.Point3DId()].Track().DeleteElement(i.second.ImageId(), idx1);
                }
                idx1++;
            }
            del_ids.push_back(i.first);
        }
    }
    if(!del_ids.empty()){
        for(auto &id: del_ids){
            images_.erase(id);
        }
    }
}
//getFileNames,
bool colmapBase::getFileNames(const megaDepthFolders& folders){
    vector<point2D_t> del_ids;
    for(auto &i: correspImgs_){
        string imgName = images_.at(i.first).Name();
        std::string root, ext;
        SplitFileExtension(imgName, &root, &ext);
        i.second.depthImg1 = JoinPaths(folders.mdDepth, root + folders.depthExt);
        i.second.img1 = JoinPaths(folders.mdImgF, imgName);
        i.second.imgOrig1 = JoinPaths(folders.sfmImgF, imgName);
        imgName = images_.at(i.second.imgID).Name();
        SplitFileExtension(imgName, &root, &ext);
        i.second.depthImg2 = JoinPaths(folders.mdDepth, root + folders.depthExt);
        i.second.img2 = JoinPaths(folders.mdImgF, imgName);
        i.second.imgOrig2 = JoinPaths(folders.sfmImgF, imgName);
        if(!ExistsFile(i.second.depthImg1) || !ExistsFile(i.second.img1) || !ExistsFile(i.second.imgOrig1) ||
           !ExistsFile(i.second.depthImg2) || !ExistsFile(i.second.img2) || !ExistsFile(i.second.imgOrig2)){
            del_ids.push_back(i.first);
            continue;
        }
    }
    if(!del_ids.empty()){
        for(auto &id: del_ids){
            correspImgs_.erase(id);
        }
    }
    return !correspImgs_.empty();
}

bool colmapBase::getCorrespondingImgs(){
    for(auto &i: images_){
        EIGEN_STL_UMAP(image_t, struct corrStats) corresp;
        EIGEN_STL_UMAP(image_t, struct corrStats)::iterator it, it1;
        image_t bestId = kInvalidImageId;
        size_t maxCnt = 0;
        double maxWeight = 0;
        for(auto &pt2d: i.second.Points2D()){
            if(pt2d.HasPoint3D()){
                for(auto &img: points3D_[pt2d.Point3DId()].Track().Elements()){
                    if(img.image_id == i.second.ImageId()){
                        continue;
                    }
                    it = corresp.find(img.image_id);
                    if(it != corresp.end()){
                        it->second.nrCorresp3D++;
                        double nrCorresp3Dw = static_cast<double>(it->second.nrCorresp3D) / 10.;
                        nrCorresp3Dw *= nrCorresp3Dw;
                        it->second.weight = nrCorresp3Dw * it->second.viewAngle * it->second.tvecNorm;
                        if(it->second.weight > maxWeight){
                            it1 = correspImgs_.find(img.image_id);
                            if(it1 == correspImgs_.end()) {
                                bestId = img.image_id;
                                maxCnt = it->second.nrCorresp3D;
                                maxWeight = it->second.weight;
                            }
                        }
                    }else{
                        Eigen::Matrix3d K1, R0, R1, Rrel;
                        Eigen::Vector3d t0, t1, trel;
                        R0 = i.second.RotationMatrix();
                        t0 = i.second.Tvec();
                        colmap::Image &img2 = images_.at(img.image_id);
                        K1 = cameras_.at(img2.CameraId()).CalibrationMatrix();
                        R1 = img2.RotationMatrix();
                        t1 = img2.Tvec();
                        double angle = getViewAngleAbsoluteCams(R0, t0, K1, R1, t1, false);
                        getRelativeFromAbsPoses(R0, t0, R1, t1, Rrel, trel);
                        double tvec_norm = trel.norm();
                        corresp.emplace(img.image_id, corrStats(img.image_id, 1, angle, tvec_norm, Rrel, trel));
                    }
                }
            }
        }
        if((bestId != kInvalidImageId) && (maxCnt > 12)) {
            corrStats &bestStats = corresp.at(bestId);
            correspImgs_.emplace(i.second.ImageId(), corrStats(bestId, maxCnt, bestStats.viewAngle,
                    bestStats.tvecNorm, maxWeight, bestStats.R_rel, bestStats.t_rel));
        }
    }
    return !correspImgs_.empty();
}

void colmapBase::filterInterestingImgs(){
    vector<point2D_t> num3D(images_.size());
    size_t idx = 0;
    for(auto & i: images_){
        num3D[idx++] = i.second.NumPoints3D();
    }
    double mean = Mean(num3D);
    double sd = StdDev(num3D);
    int th1 = 100;
    if(mean > 500.){
        th1 = 400;
    }else if(mean > 250.){
        th1 = 150;
    }
    min_num_points3D_Img_ = static_cast<point2D_t>(std::max(std::min(static_cast<int>(round(mean - sd)), th1), 32));
    vector<point2D_t> del_ids;
    for(auto &i: images_){
        if(i.second.NumPoints3D() < min_num_points3D_Img_){
            point2D_t idx1 = 0;
            for(auto &pt2d: i.second.Points2D()){
                if(pt2d.HasPoint3D()){
                    points3D_[pt2d.Point3DId()].Track().DeleteElement(i.second.ImageId(), idx1);
                }
                idx1++;
            }
            del_ids.push_back(i.first);
        }else{
            point2D_t idx1 = 0;
            for(auto &pt2d: i.second.Points2D()){
                if(pt2d.HasPoint3D()){
                    i.second.IncrementCorrespondenceHasPoint3D(idx1);
                }
                idx1++;
            }
        }
    }
    if(!del_ids.empty()){
        for(auto &id: del_ids){
            images_.erase(id);
        }
    }
    del_ids.clear();
    vector<size_t> scores;
    std::size_t min_score = images_.begin()->second.Point3DVisibilityScoreMax() / 3;
    for(auto &i: images_){
        scores.emplace_back(i.second.Point3DVisibilityScore());
    }
    size_t max_score = *max_element(scores.begin(), scores.end());
    if(max_score <= min_score){
        min_score = static_cast<size_t>(round((Mean(scores) + Median(scores)) / 2.));
    }
    for(auto &i: images_){
        const std::size_t min_score = i.second.Point3DVisibilityScoreMax() / 3;
        if(i.second.Point3DVisibilityScore() < min_score){
            point2D_t idx1 = 0;
            for(auto &pt2d: i.second.Points2D()){
                if(pt2d.HasPoint3D()){
                    points3D_[pt2d.Point3DId()].Track().DeleteElement(i.second.ImageId(), idx1);
                }
                idx1++;
            }
            del_ids.push_back(i.first);
        }
    }
    if(!del_ids.empty()){
        for(auto &id: del_ids){
            images_.erase(id);
        }
    }
}

void colmapBase::ReadText(const std::string& path) {
    ReadCamerasText(JoinPaths(path, "cameras.txt"));
    ReadImagesText(JoinPaths(path, "images.txt"));
    ReadPoints3DText(JoinPaths(path, "points3D.txt"));
}

void colmapBase::ReadCamerasText(const std::string& path) {
    cameras_.clear();

    std::ifstream file(path);
    if(!file.is_open()){
        cerr <<  "Unable to open " << path << endl;
        return;
    }

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        colmap::StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);

        class Camera camera;

        // ID
        std::getline(line_stream, item, ' ');
        camera.SetCameraId(std::stoul(item));

        // MODEL
        std::getline(line_stream, item, ' ');
        camera.SetModelIdFromName(item);

        // WIDTH
        std::getline(line_stream, item, ' ');
        camera.SetWidth(std::stoll(item));

        // HEIGHT
        std::getline(line_stream, item, ' ');
        camera.SetHeight(std::stoll(item));

        // PARAMS
        camera.Params().clear();
        while (!line_stream.eof()) {
            std::getline(line_stream, item, ' ');
            camera.Params().push_back(std::stold(item));
        }

        if(!camera.VerifyParams()){
            cerr << "Verification of camera parameters failed" << endl;
        }

        cameras_.emplace(camera.CameraId(), camera);
    }
}

void colmapBase::ReadImagesText(const std::string& path) {
    images_.clear();

    std::ifstream file(path);
    CHECK(file.is_open());

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream1(line);

        // ID
        std::getline(line_stream1, item, ' ');
        const image_t image_id = std::stoul(item);

        class Image image;
        image.SetImageId(image_id);

        image.SetRegistered(true);
        reg_image_ids_.push_back(image_id);

        // QVEC (qw, qx, qy, qz)
        std::getline(line_stream1, item, ' ');
        image.Qvec(0) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(1) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(2) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Qvec(3) = std::stold(item);

        image.NormalizeQvec();

        // TVEC
        std::getline(line_stream1, item, ' ');
        image.Tvec(0) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Tvec(1) = std::stold(item);

        std::getline(line_stream1, item, ' ');
        image.Tvec(2) = std::stold(item);

        // CAMERA_ID
        std::getline(line_stream1, item, ' ');
        image.SetCameraId(std::stoul(item));

        // NAME
        std::getline(line_stream1, item, ' ');
        image.SetName(item);

        // POINTS2D
        if (!std::getline(file, line)) {
            break;
        }

        StringTrim(&line);
        std::stringstream line_stream2(line);

        std::vector<Eigen::Vector2d> points2D;
        std::vector<point3D_t> point3D_ids;

        if (!line.empty()) {
            while (!line_stream2.eof()) {
                Eigen::Vector2d point;

                std::getline(line_stream2, item, ' ');
                point.x() = std::stold(item);

                std::getline(line_stream2, item, ' ');
                point.y() = std::stold(item);

                points2D.push_back(point);

                std::getline(line_stream2, item, ' ');
                if (item == "-1") {
                    point3D_ids.push_back(kInvalidPoint3DId);
                } else {
                    point3D_ids.push_back(std::stoll(item));
                }
            }
        }

        image.SetUp(Camera(image.CameraId()));
        image.SetPoints2D(points2D);

        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
             ++point2D_idx) {
            if (point3D_ids[point2D_idx] != kInvalidPoint3DId) {
                image.SetPoint3DForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
            }
        }

        images_.emplace(image.ImageId(), image);
    }
}

void colmapBase::ReadPoints3DText(const std::string& path) {
    points3D_.clear();

    std::ifstream file(path);
    CHECK(file.is_open());

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);

        // ID
        std::getline(line_stream, item, ' ');
        const point3D_t point3D_id = std::stoll(item);

        // Make sure, that we can add new 3D points after reading 3D points
        // without overwriting existing 3D points.
        num_added_points3D_ = std::max(num_added_points3D_, point3D_id);

        class Point3D point3D;

        // XYZ
        std::getline(line_stream, item, ' ');
        point3D.XYZ(0) = std::stold(item);

        std::getline(line_stream, item, ' ');
        point3D.XYZ(1) = std::stold(item);

        std::getline(line_stream, item, ' ');
        point3D.XYZ(2) = std::stold(item);

        // Color
        std::getline(line_stream, item, ' ');
        point3D.Color(0) = static_cast<uint8_t>(std::stoi(item));

        std::getline(line_stream, item, ' ');
        point3D.Color(1) = static_cast<uint8_t>(std::stoi(item));

        std::getline(line_stream, item, ' ');
        point3D.Color(2) = static_cast<uint8_t>(std::stoi(item));

        // ERROR
        std::getline(line_stream, item, ' ');
        point3D.SetError(std::stold(item));

        // TRACK
        while (!line_stream.eof()) {
            TrackElement track_el;

            std::getline(line_stream, item, ' ');
            StringTrim(&item);
            if (item.empty()) {
                break;
            }
            track_el.image_id = std::stoul(item);

            std::getline(line_stream, item, ' ');
            track_el.point2D_idx = std::stoul(item);

            point3D.Track().AddElement(track_el);
        }

        point3D.Track().Compress();

        points3D_.emplace(point3D_id, point3D);
    }
}
