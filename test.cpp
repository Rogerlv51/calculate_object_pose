#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <cmath>
#include <iostream> // Added for std::cout in main if not already there for PCL_ERROR

// Input：法向量normal，PCA主方向mainDirection（均为单位向量）
// 输出：旋转矩阵R，对应的ZYX欧拉角（单位：度），以及四元数 (w, x, y, z)
void computeRotationMatrixAndEulerZYX(
    const cv::Point3f& normal,
    cv::Point3f mainDirection,
    cv::Mat& R,
    cv::Vec3f& eulerZYX_deg,
    cv::Vec4f& quaternion_wxyz) // Added quaternion output
{
    // 1. Z轴 = 法向量（normal）
    cv::Point3f zAxis = normal;
    // 2. X轴 = mainDirection，但要保证和z轴正交
    // 去除 z 方向分量
    float dotProd = zAxis.dot(mainDirection);
    cv::Point3f xAxis = mainDirection - dotProd * zAxis;

    // 如果xAxis过小（可能平行），用备用构造方式
    float normX = cv::norm(xAxis);
    if (normX < 1e-6f)
    {
        // 备用：选一个和z轴正交的向量
        if (std::abs(zAxis.x) < std::abs(zAxis.y) && std::abs(zAxis.x) < std::abs(zAxis.z))
            xAxis = cv::Point3f(0, -zAxis.z, zAxis.y);
        else if (std::abs(zAxis.y) < std::abs(zAxis.z))
            xAxis = cv::Point3f(-zAxis.z, 0, zAxis.x);
        else
            xAxis = cv::Point3f(-zAxis.y, zAxis.x, 0);
        normX = cv::norm(xAxis);
    }
    xAxis /= normX;

    // 3. Y轴 = Z × X，保证右手系
    cv::Point3f yAxis = zAxis.cross(xAxis);
    // Ensure yAxis is also unit length, though cross product of unit orthogonal vectors should be unit length
    yAxis /= cv::norm(yAxis);


    // 4. 构造旋转矩阵 R，列为 x,y,z 轴
    R = (cv::Mat_<float>(3, 3) <<
        xAxis.x, yAxis.x, zAxis.x,
        xAxis.y, yAxis.y, zAxis.y,
        xAxis.z, yAxis.z, zAxis.z);

    // --- 提取ZYX欧拉角（Yaw-Pitch-Roll） ---
    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    // R = [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
    //      [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
    //      [-sp,   cp*sr,            cp*cr]]
    // where cy=cos(yaw), sy=sin(yaw), cp=cos(pitch), sp=sin(pitch), sr=cos(roll), sr=sin(roll)

    float r00 = R.at<float>(0, 0);
    float r01 = R.at<float>(0, 1);
    float r02 = R.at<float>(0, 2);
    float r10 = R.at<float>(1, 0);
    float r11 = R.at<float>(1, 1);
    // float r12 = R.at<float>(1, 2); // Not used in this Euler angle formulation
    float r20 = R.at<float>(2, 0);
    float r21 = R.at<float>(2, 1);
    float r22 = R.at<float>(2, 2);

    float roll, pitch, yaw;
    const float epsilon = 1e-6f;

    // Pitch calculation
    // pitch = atan2(-r20, sqrt(r00^2 + r10^2)) or atan2(-r20, sqrt(r21^2 + r22^2))
    // Both sqrt terms are abs(cos(pitch))
    pitch = std::atan2(-r20, std::sqrt(r21*r21 + r22*r22));

    if (std::abs(r20) > 1.0f - epsilon) { // Gimbal lock: pitch is +/- 90 degrees
        if (r20 < 0) { // Pitch = +90 deg (r20 = -sin(pitch) = -1)
            pitch = static_cast<float>(CV_PI) / 2.0f;
            // Conventionally, set roll to 0
            // yaw + roll = atan2(r01, r11) (from some derivations, check specific ZYX convention)
            // For ZYX (Diebel): alpha + gamma = atan2(r_01, r_11) where alpha=yaw, gamma=roll
            // If roll (gamma) = 0, then yaw (alpha) = atan2(r01, r11)
            roll = 0.0f;
            yaw = std::atan2(r01, r11);
        } else { // Pitch = -90 deg (r20 = -sin(pitch) = 1)
            pitch = -static_cast<float>(CV_PI) / 2.0f;
            // Conventionally, set roll to 0
            // For ZYX (Diebel): alpha - gamma = atan2(-r_01, r_11)
            // If roll (gamma) = 0, then yaw (alpha) = atan2(-r01, r11)
            roll = 0.0f;
            yaw = std::atan2(-r01, r11);
        }
    } else { // No gimbal lock
        // roll = atan2(sin_roll * cos_pitch, cos_roll * cos_pitch)
        roll  = std::atan2(r21, r22);
        // yaw = atan2(sin_yaw * cos_pitch, cos_yaw * cos_pitch)
        yaw   = std::atan2(r10, r00);
    }

    const float RAD2DEG = 180.0f / static_cast<float>(CV_PI);
    eulerZYX_deg[0] = roll * RAD2DEG;
    eulerZYX_deg[1] = pitch * RAD2DEG;
    eulerZYX_deg[2] = yaw * RAD2DEG;

    // --- Calculate Quaternion (w, x, y, z) from Rotation Matrix R ---
    // Using Shepperd's method (robust)
    float trace = r00 + r11 + r22;
    float qw, qx, qy, qz;

    if (trace > 0.0f) {
        float S = std::sqrt(trace + 1.0f) * 2.0f; // S = 4*qw
        qw = 0.25f * S;
        qx = (r21 - R.at<float>(1,2)) / S; // r21 - r12
        qy = (R.at<float>(0,2) - r20) / S; // r02 - r20
        qz = (r10 - r01) / S;
    } else if ((r00 > r11) && (r00 > r22)) {
        float S = std::sqrt(1.0f + r00 - r11 - r22) * 2.0f; // S = 4*qx
        qw = (r21 - R.at<float>(1,2)) / S;
        qx = 0.25f * S;
        qy = (r01 + r10) / S;
        qz = (R.at<float>(0,2) + r20) / S; // r02 + r20
    } else if (r11 > r22) {
        float S = std::sqrt(1.0f + r11 - r00 - r22) * 2.0f; // S = 4*qy
        qw = (R.at<float>(0,2) - r20) / S;
        qx = (r01 + r10) / S;
        qy = 0.25f * S;
        qz = (R.at<float>(1,2) + r21) / S; // r12 + r21
    } else {
        float S = std::sqrt(1.0f + r22 - r00 - r11) * 2.0f; // S = 4*qz
        qw = (r10 - r01) / S;
        qx = (R.at<float>(0,2) + r20) / S;
        qy = (R.at<float>(1,2) + r21) / S;
        qz = 0.25f * S;
    }
    quaternion_wxyz[0] = qw;
    quaternion_wxyz[1] = qx;
    quaternion_wxyz[2] = qy;
    quaternion_wxyz[3] = qz;

    // Optional: Normalize quaternion if needed due to precision, though these formulas aim for unit quaternions
    // float norm_q = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    // if (norm_q > epsilon) {
    //     quaternion_wxyz[0] /= norm_q;
    //     quaternion_wxyz[1] /= norm_q;
    //     quaternion_wxyz[2] /= norm_q;
    //     quaternion_wxyz[3] /= norm_q;
    // }
}

int main(){
    std::string pcdFile = "cloud_plane_projected.pcd"; // 替换为你的PCD文件路径
    std::vector<cv::Point3f> points;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFile, *cloud) == -1) {
        PCL_ERROR("Couldn't read PCD file\n");
        return -1;
    }

    if (cloud->points.empty()) {
        std::cerr << "Error: PCD file is empty or failed to load points." << std::endl;
        return -1;
    }

    for (const auto& pt : cloud->points) {
        points.emplace_back(pt.x, pt.y, pt.z);
    }

    if (points.size() < 3) { // PCA needs at least 3 points for 3D
        std::cerr << "Error: Not enough points for PCA analysis (" << points.size() << " points found)." << std::endl;
        return -1;
    }

    cv::Mat data((int)points.size(), 3, CV_32F);
    for (size_t i = 0; i < points.size(); ++i) { // Use size_t for loop
        data.at<float>(static_cast<int>(i), 0) = points[i].x;
        data.at<float>(static_cast<int>(i), 1) = points[i].y;
        data.at<float>(static_cast<int>(i), 2) = points[i].z;
    }

    // 求中心点
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    if (pca.eigenvectors.empty() || pca.eigenvectors.rows < 3) {
        std::cerr << "Error: PCA failed to compute eigenvectors." << std::endl;
        return -1;
    }

    // 平面法向量 = 第3个特征向量（最小特征值方向）
    cv::Point3f normal(
        pca.eigenvectors.at<float>(2, 0),
        pca.eigenvectors.at<float>(2, 1),
        pca.eigenvectors.at<float>(2, 2));

    if (cv::norm(normal) < 1e-6f) { // Check for zero normal vector
        std::cerr << "Error: PCA resulted in a near-zero normal vector for the plane." << std::endl;
        // This can happen if points are collinear or coincident
        // Fallback or error handling needed
        // For now, let's try to use a default normal if this happens, or exit
        // normal = cv::Point3f(0,0,1); // Example fallback, but likely indicates bad input data
        return -1;
    }
    if (normal.z > 0) normal *= -1;  // 朝向相机的方向

    // 主方向 = 第1个特征向量（最大特征值方向）
    cv::Point3f mainDir(
        pca.eigenvectors.at<float>(0, 0),
        pca.eigenvectors.at<float>(0, 1),
        pca.eigenvectors.at<float>(0, 2));
    
    if (cv::norm(mainDir) < 1e-6f) { // Check for zero main direction
         std::cerr << "Error: PCA resulted in a near-zero main direction vector." << std::endl;
         // This can happen if points are coincident or form a very thin line not aligned with an axis
         // mainDir = cv::Point3f(1,0,0); // Example fallback
         return -1;
    }


    // 确保都是单位向量
    normal /= cv::norm(normal);
    mainDir /= cv::norm(mainDir);

    cv::Mat R;
    cv::Vec3f eulerZYX_deg;
    cv::Vec4f quaternion_wxyz; // Declare quaternion

    computeRotationMatrixAndEulerZYX(normal, mainDir, R, eulerZYX_deg, quaternion_wxyz);

    std::cout << "Rotation Matrix R:\n" << R << std::endl << std::endl;

    std::cout << "欧拉角 (ZYX顺序):\n"
          << "Roll (X): " << eulerZYX_deg[0] << "°\n"
          << "Pitch(Y): " << eulerZYX_deg[1] << "°\n"
          << "Yaw  (Z): " << eulerZYX_deg[2] << "°\n\n";

    std::cout << "四元数 (w, x, y, z):\n"
              << "(" << quaternion_wxyz[0] << ", "
              << quaternion_wxyz[1] << ", "
              << quaternion_wxyz[2] << ", "
              << quaternion_wxyz[3] << ")\n";
    
    // Verify quaternion norm (should be close to 1)
    float q_norm = std::sqrt(
        quaternion_wxyz[0]*quaternion_wxyz[0] +
        quaternion_wxyz[1]*quaternion_wxyz[1] +
        quaternion_wxyz[2]*quaternion_wxyz[2] +
        quaternion_wxyz[3]*quaternion_wxyz[3]);
    std::cout << "Quaternion Norm: " << q_norm << std::endl;


    return 0; // Added return 0 for successful execution
}