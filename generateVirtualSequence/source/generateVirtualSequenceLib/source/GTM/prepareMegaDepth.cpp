//
// Created by maierj on 5/11/20.
//
#include "GTM/prepareMegaDepth.h"
#include <unordered_map>
#include <functional>
#include "GTM/colmap/camera.h"
#include "GTM/colmap/image.h"
#include "GTM/colmap/point3d.h"
#include "GTM/colmap/logging.h"
#include "GTM/colmap/misc.h"
#include "GTM/colmap/math.h"
#include <fstream>


using namespace colmap;
using namespace std;

class colmapBase{
public:
    void ReadText(const std::string& path);
    void filterInterestingImgs();
private:
    void ReadCamerasText(const std::string& path);
    void ReadImagesText(const std::string& path);
    void ReadPoints3DText(const std::string& path);

    // Get const objects.
    inline const class Camera& Camera(const camera_t camera_id) const;
    inline const class Image& Image(const image_t image_id) const;

    // Get mutable objects.
    inline class Camera& Camera(const camera_t camera_id);
    inline class Image& Image(const image_t image_id);

    EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
    EIGEN_STL_UMAP(image_t, class Image) images_;
    EIGEN_STL_UMAP(point3D_t, class Point3D) points3D_;
    std::vector<image_t> reg_image_ids_;
    // Total number of added 3D points, used to generate unique identifiers.
    point3D_t num_added_points3D_;
    point2D_t min_num_points3D_Img_;
};

const class Camera& colmapBase::Camera(const camera_t camera_id) const {
    return cameras_.at(camera_id);
}

const class Image& colmapBase::Image(const image_t image_id) const {
    return images_.at(image_id);
}

class Camera& colmapBase::Camera(const camera_t camera_id) {
    return cameras_.at(camera_id);
}

class Image& colmapBase::Image(const image_t image_id) {
    return images_.at(image_id);
}

bool readColmapData(){

}

void colmapBase::filterInterestingImgs(){
    vector<point2D_t> num3D(images_.size());
    size_t idx = 0;
    for(auto & i: images_){
        num3D[idx++] = i.second.NumPoints3D();
    }
    double mean = Mean(num3D);
    double sd = StdDev(num3D);
    int th1 = 0;
    if(mean > 500.){
        th1 = 400;
    }else if(mean > 250.){
        th1 = 150;
    }else{
        th1 = 100;
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
    for(auto &i: images_){
        const std::size_t min_score = (5 * i.second.Point3DVisibilityScoreMax()) / 12;
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
