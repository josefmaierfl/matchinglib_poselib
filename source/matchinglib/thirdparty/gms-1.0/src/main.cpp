#include "ProcessImagePairs.h"

using namespace std;

int main(int argc, char** argv)
{
    string name(argv[0]);
    size_t pos = name.find_last_of("/\\");
    if(pos != string::npos)
        name = name.substr(pos + 1);
    string usage("Usage: " + name + " <image path> [--imageIdx <image index of start frame>] [--imageOffset <image offset>] [--numFeatures <number of ORB features>] [--imageHeight <image height of scaled image>]");
    if (argc < 2)
    {
        printf("Error: too few arguments!\n%s\n\n", usage.c_str());
        return -1;
    }

    unsigned int imageIdx = 0;
    unsigned int imageOffset = 10;
    unsigned int numFeatures = 1000;
    unsigned int imageHeight = 480;
    string inputImagePath = argv[1];

    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "--imageIdx") == 0)
        {
            if (argc < i + 1)
            {
                printf("Error: value missing for imageIdx!\n%s\n\n", usage.c_str());
                return -1;
            }

            imageIdx = static_cast<unsigned int>(atoi(argv[++i]));
        }
        else if (strcmp(argv[i], "--imageOffset") == 0)
        {
            if (argc < i + 1)
            {
                printf("Error: value missing for imageOffset!\n%s\n\n", usage.c_str());
                return -1;
            }

            imageOffset = static_cast<unsigned int>(atoi(argv[++i]));
        }
        else if (strcmp(argv[i], "--numFeatures") == 0)
        {
            if (argc < i + 1)
            {
                printf("Error: value missing for numFeatures!\n%s\n\n", usage.c_str());
                return -1;
            }

            numFeatures = static_cast<unsigned int>(atoi(argv[++i]));
        }
        else if (strcmp(argv[i], "--imageHeight") == 0)
        {
            if (argc < i + 1)
            {
                printf("Error: value missing for imageHeight!\n%s\n\n", usage.c_str());
                return -1;
            }

            imageHeight = static_cast<unsigned int>(atoi(argv[++i]));
        }
        else
        {
            printf("Error: invalid argument: %s\n%s\n\n", argv[i], usage.c_str());
            return -1;
        }
    }

#ifdef USE_GPU
    int flag = cv::cuda::getCudaEnabledDeviceCount();
    if (flag != 0)
        cv::cuda::setDevice(0);
#endif

    ProcessImagePairs processImagePairs(inputImagePath, imageIdx, imageOffset, numFeatures, imageHeight);

    processImagePairs.computeMatch();
}
