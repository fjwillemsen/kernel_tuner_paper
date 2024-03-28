#include <CL/cl.h>
#include <cstdlib>
#include <iostream>
#include <array>

using namespace std;

int main(void)
{
    cl_int status;
    size_t str_info_size;
    char *str_info;
    cl_uint uint_info;
    size_t t_info;
    array<size_t, 3> ta_info;
    cl_ulong ulong_info;

    cl_uint num_platforms;
    cl_platform_id *platforms;

    cl_uint num_devices;
    cl_device_id *devices;

    status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS)
    {
        cerr << "Failed to get the number of platforms." << endl;
        return -1;
    }

    if (num_platforms > 0)
    {
        cout << "* The number of platforms: " << num_platforms << endl;
    }
    else
    {
        cerr << "The number of platforms is zero." << endl;
        return -1;
    }

    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    for (int i = 0; i < num_platforms; ++i)
    {
        cout << "  -----------------------------------------------" << endl;
        cout << "  Platform-" << i + 1 << endl;

        clGetPlatformInfo(
            platforms[i], CL_PLATFORM_NAME, 0, NULL, &str_info_size);
        str_info = (char *)malloc(str_info_size);
        clGetPlatformInfo(
            platforms[i], CL_PLATFORM_NAME, str_info_size, str_info, NULL);
        cout << "  * Platform: " << str_info << endl;
        free(str_info);

        clGetPlatformInfo(
            platforms[i], CL_PLATFORM_PROFILE, 0, NULL, &str_info_size);
        str_info = (char *)malloc(str_info_size);
        clGetPlatformInfo(
            platforms[i], CL_PLATFORM_PROFILE, str_info_size, str_info, NULL);
        cout << "  * Profile: " << str_info << endl;
        free(str_info);

        clGetPlatformInfo(
            platforms[i], CL_PLATFORM_VERSION, 0, NULL, &str_info_size);
        str_info = (char *)malloc(str_info_size);
        clGetPlatformInfo(
            platforms[i], CL_PLATFORM_VERSION, str_info_size, str_info, NULL);
        cout << "  * " << str_info << endl;
        free(str_info);

        status = clGetDeviceIDs(
            platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (status != CL_SUCCESS)
        {
            cerr << "Failed to get the number of devices in the platform "
                 << i + 1 << '.' << endl;
            continue;
        }

        if (num_devices > 0)
        {
            cout << "  * Number of devices: " << num_devices << endl;
        }
        else
        {
            cerr << "Platform-" << i + 1 << " has no devices." << endl;
            continue;
        }

        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(
            platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (int j = 0; j < num_devices; ++j)
        {
            cout << "    ================================================="
                 << endl;
            cout << "    Device-" << j + 1 << endl;

            clGetDeviceInfo(
                devices[j], CL_DEVICE_NAME, 0, NULL, &str_info_size);
            str_info = (char *)malloc(str_info_size);
            clGetDeviceInfo(
                devices[j], CL_DEVICE_NAME, str_info_size, str_info, NULL);
            cout << "    * Name: " << str_info << endl;
            free(str_info);

            clGetDeviceInfo(devices[j],
                            CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(uint_info),
                            &uint_info,
                            NULL);
            cout << "    * Max Compute Units: " << uint_info << endl;

            clGetDeviceInfo(devices[j],
                            CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(uint_info),
                            &uint_info,
                            NULL);
            cout << "    * Max Clock freq: " << uint_info << endl;

            clGetDeviceInfo(devices[j],
                            CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(uint_info),
                            &uint_info,
                            NULL);
            cout << "    * Max work item dimensions: " << uint_info << endl;

            clGetDeviceInfo(devices[j],
                            CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            sizeof(ta_info),
                            &ta_info,
                            NULL);
            cout << "    * Max work item sizes: [";
            for (size_t i = 0; i < uint_info; i++)
            {
                cout << ta_info[i];
                if (i != uint_info - 1)
                {
                    cout << ", ";
                }
            }
            cout << "]" << endl;

            clGetDeviceInfo(devices[j],
                            CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof(t_info),
                            &t_info,
                            NULL);
            cout << "    * Max work group size: " << t_info << endl;

            clGetDeviceInfo(devices[j],
                            CL_DEVICE_LOCAL_MEM_SIZE,
                            sizeof(ulong_info),
                            &ulong_info,
                            NULL);
            cout << "    * Max local memory size (bytes): " << ulong_info << endl;

            cout << "    ================================================="
                 << endl;
        }
        cout << "  -----------------------------------------------" << endl;
    }

    free(devices);
    free(platforms);
    return 0;
}