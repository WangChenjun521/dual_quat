#include <vector>
#include "dual_quat_cu.hpp"

using namespace Tbx;

/**
 * Function to deform a mesh with dual quaternions.
 *
 * @note Originally this was CUDA code. Aside from this deformer function
 * which uses std::vector every other classes and methods should be readily
 * convertible to CUDA code just by adding __host__ __device__ flags before
 * their definitions.
 *
 * @param in_verts : vector of mesh vertices
 * @param in_normals : vector of mesh normals (same order as 'in_verts')
 * @param out_verts : deformed vertices with dual quaternions
 * @param out_normals : deformed normals with dual quaternions
 * @param dual_quat : list of dual quaternions transformations per joints
 * @param weights : list of influence weights for each vertex
 * @param joints_id : list of joints influence fore each vertex (same order as 'weights')
 */
void dual_quat_deformer(const std::vector<Point3>& in_verts,
                        const std::vector<Vec3>& in_normals,
                        std::vector<Vec3>& out_verts,
                        std::vector<Vec3>& out_normals,
                        const std::vector<Dual_quat_cu>& dual_quat,
                        const std::vector< std::vector<float> >& weights,
                        const std::vector< std::vector<int> >& joints_id)
{
    for(unsigned v = 0; v < in_verts.size(); ++v)
    {
        const int nb_joints = weights[v].size(); // Number of joints influencing vertex p

        // Init dual quaternion with first joint transformation
        int   k0 = -1;
        float w0 = 0.f;
        Dual_quat_cu dq_blend;
        Quat_cu q0;

        if(nb_joints != 0)
        {
            k0 = joints_id[v][0];
            w0 = weights[v][0];
        }else
            dq_blend = Dual_quat_cu::identity();

        if(k0 != -1) dq_blend = dual_quat[k0] * w0;

        int pivot = k0;

        q0 = dual_quat[pivot].rotation();
        // Look up the other joints influencing 'p' if any
        for(int j = 1; j < nb_joints; j++)
        {
            const int k = joints_id[v][j];
            float w = weights[v][j];
            const Dual_quat_cu& dq = (k == -1) ? Dual_quat_cu::identity() : dual_quat[k];

            if( dq.rotation().dot( q0 ) < 0.f )
                w *= -1.f;

            dq_blend = dq_blend + dq * w;
        }

        // Compute animated position
        Vec3 vi = dq_blend.transform( in_verts[v] ).to_vec3();
        out_verts[v] = vi;
        // Compute animated normal
        out_normals[v] = dq_blend.rotate( in_normals[v] );
    }
}



// #include <iostream>
// #include <open3d/Open3D.h>
// int main(){

//     using namespace std;
//     cout<<"test!"<<endl;
//     Dual_quat_cu dq_blend;
//     dq_blend=Dual_quat_cu::identity();

//     cout<<dq_blend.get_non_dual_part().w()<<endl;
//     cout<<dq_blend.get_non_dual_part().i()<<endl;
//     cout<<dq_blend.get_non_dual_part().j()<<endl;
//     cout<<dq_blend.get_non_dual_part().k()<<endl;

//     cout<<dq_blend.get_dual_part().w()<<endl;
//     cout<<dq_blend.get_dual_part().i()<<endl;
//     cout<<dq_blend.get_dual_part().j()<<endl;
//     cout<<dq_blend.get_dual_part().k()<<endl;



//     return 0;
// }


#include <iostream>
#include <memory>
#include <thread>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > Visualizer [mesh|spin|slowspin|pointcloud|rainbow|image|depth|editing|editmesh] [filename]");
    utility::LogInfo("    > Visualizer [animation] [filename] [trajectoryfile]");
    utility::LogInfo("    > Visualizer [rgbd] [color] [depth] [--rgbd_type]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string option(argv[1]);
    if (option == "mesh") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        visualization::DrawGeometries({mesh_ptr}, "Mesh", 1600, 900);
    } else if (option == "editmesh") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        visualization::DrawGeometriesWithVertexSelection(
                {mesh_ptr}, "Edit Mesh", 1600, 900);
    } else if (option == "spin") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        visualization::DrawGeometriesWithAnimationCallback(
                {mesh_ptr},
                [&](visualization::Visualizer *vis) {
                    vis->GetViewControl().Rotate(10, 0);
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    return false;
                },
                "Spin", 1600, 900);
    } else if (option == "slowspin") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        visualization::DrawGeometriesWithKeyCallbacks(
                {mesh_ptr},
                {{GLFW_KEY_SPACE,
                  [&](visualization::Visualizer *vis) {
                      vis->GetViewControl().Rotate(10, 0);
                      std::this_thread::sleep_for(
                              std::chrono::milliseconds(30));
                      return false;
                  }}},
                "Press Space key to spin", 1600, 900);
    } else if (option == "pointcloud") {
        auto cloud_ptr = std::make_shared<geometry::PointCloud>();
        if (io::ReadPointCloud(argv[2], *cloud_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        cloud_ptr->NormalizeNormals();
        visualization::DrawGeometries({cloud_ptr}, "PointCloud", 1600, 900);
    } else if (option == "rainbow") {
        auto cloud_ptr = std::make_shared<geometry::PointCloud>();
        if (io::ReadPointCloud(argv[2], *cloud_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        cloud_ptr->NormalizeNormals();
        cloud_ptr->colors_.resize(cloud_ptr->points_.size());
        double color_index = 0.0;
        double color_index_step = 0.05;

        auto update_colors_func = [&cloud_ptr](double index) {
            auto color_map_ptr = visualization::GetGlobalColorMap();
            for (auto &c : cloud_ptr->colors_) {
                c = color_map_ptr->GetColor(index);
            }
        };
        update_colors_func(1.0);

        visualization::DrawGeometriesWithAnimationCallback(
                {cloud_ptr},
                [&](visualization::Visualizer *vis) {
                    color_index += color_index_step;
                    if (color_index > 2.0) color_index -= 2.0;
                    update_colors_func(fabs(color_index - 1.0));
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    return true;
                },
                "Rainbow", 1600, 900);
    } else if (option == "image") {
        auto image_ptr = std::make_shared<geometry::Image>();
        if (io::ReadImage(argv[2], *image_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        visualization::DrawGeometries({image_ptr}, "Image", image_ptr->width_,
                                      image_ptr->height_);
    } else if (option == "rgbd") {
        if (argc < 4) {
            PrintHelp();
            return 1;
        }

        int rgbd_type =
                utility::GetProgramOptionAsInt(argc, argv, "--rgbd_type", 0);
        auto color_ptr = std::make_shared<geometry::Image>();
        auto depth_ptr = std::make_shared<geometry::Image>();

        if (io::ReadImage(argv[2], *color_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }

        if (io::ReadImage(argv[3], *depth_ptr)) {
            utility::LogInfo("Successfully read {}", argv[3]);
        } else {
            utility::LogWarning("Failed to read {}", argv[3]);
            return 1;
        }

        std::shared_ptr<geometry::RGBDImage> (*CreateRGBDImage)(
                const geometry::Image &, const geometry::Image &, bool);
        if (rgbd_type == 0)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromRedwoodFormat;
        else if (rgbd_type == 1)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromTUMFormat;
        else if (rgbd_type == 2)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromSUNFormat;
        else if (rgbd_type == 3)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromNYUFormat;
        else
            CreateRGBDImage = &geometry::RGBDImage::CreateFromRedwoodFormat;
        auto rgbd_ptr = CreateRGBDImage(*color_ptr, *depth_ptr, false);
        visualization::DrawGeometries({rgbd_ptr}, "RGBD", depth_ptr->width_ * 2,
                                      depth_ptr->height_);

    } else if (option == "depth") {
        auto image_ptr = io::CreateImageFromFile(argv[2]);
        camera::PinholeCameraIntrinsic camera;
        camera.SetIntrinsics(640, 480, 575.0, 575.0, 319.5, 239.5);
        auto pointcloud_ptr =
                geometry::PointCloud::CreateFromDepthImage(*image_ptr, camera);
        visualization::DrawGeometries(
                {pointcloud_ptr},
                "geometry::PointCloud from Depth geometry::Image", 1920, 1080);
    } else if (option == "editing") {
        auto pcd = io::CreatePointCloudFromFile(argv[2]);
        visualization::DrawGeometriesWithEditing({pcd}, "Editing", 1920, 1080);
    } else if (option == "animation") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        if (argc == 3) {
            visualization::DrawGeometriesWithCustomAnimation(
                    {mesh_ptr}, "Animation", 1920, 1080);
        } else {
            visualization::DrawGeometriesWithCustomAnimation(
                    {mesh_ptr}, "Animation", 1600, 900, 50, 50, argv[3]);
        }
    }

    utility::LogInfo("End of the test.");

    return 0;
}