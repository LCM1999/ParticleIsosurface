#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <cuda_def.h>
#include <cuda_runtime.h>

namespace cstoneOctree
{

template<class T>
struct Vec3
{
	T x;
	T y;
	T z;

	HOST_DEVICE Vec3() : x(0.0f), y(0.0f), z(0.0f) {}

	HOST_DEVICE Vec3(T x, T y, T z): x(x), y(y), z(z) {}

    HOST_DEVICE Vec3 operator+(const Vec3& other) const
    {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }


	HOST_DEVICE Vec3 operator+(const T other) const
    {
        return Vec3(x + other, y + other, z + other);
	}

    HOST_DEVICE void operator+=(const Vec3& other)
    {
        x += other.x;
        y += other.y;
		z += other.z;
        return;
    }


    HOST_DEVICE void operator+=(const T other)
    {
        x += other;
        y += other;
		z += other;
        return;
    }

	HOST_DEVICE Vec3 operator-(const Vec3& other) const
    {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
	HOST_DEVICE Vec3 operator-(const T other) const
    {
        return Vec3(x - other, y - other, z - other);
	}

    HOST_DEVICE void operator-=(const Vec3& other)
    {
        x -= other.x;
		y -= other.y;
		z -= other.z;
        return;
    }


    HOST_DEVICE void operator-=(const T other)
    {
        x -= other;
        y -= other;
		z -= other;
        return;
    }

	HOST_DEVICE Vec3 operator*(const Vec3& other) const
    {
        return Vec3(x * other.x, y * other.y, z * other.z);
	}


	HOST_DEVICE Vec3 operator*(const T other) const
    {
        return Vec3(x * other, y * other, z * other);
	}

	HOST_DEVICE void operator*=(const Vec3& other)
    {
        x *= other.x;
        y *= other.y;
		z *= other.z;
        return;
    }


    HOST_DEVICE void operator*=(const T other)
    {
        x *= other;
        y *= other;
		z *= other;
        return;
    }

	HOST_DEVICE Vec3 operator/(const Vec3& other) const
    {
        assert(other.x == 0 || other.y == 0 || other.z == 0);
        return Vec3(x / other.x, y / other.y, z / other.z);
	}


	HOST_DEVICE Vec3 operator/(const T other) const
    {
        assert(other == 0);
        return Vec3(x / other, y / other, z / other);
	}

	HOST_DEVICE void operator/=(const Vec3& other)
    {
        assert(other.x == 0 || other.y == 0 || other.z == 0);
        x /= other.x;
        y /= other.y;
		z /= other.z;
        return;
    }


    HOST_DEVICE void operator/=(const T other)
    {
        assert(other == 0);
        x /= other;
        y /= other;
		z /= other;
        return;
    }

    HOST_DEVICE Vec3& operator=(const Vec3& other) {
        if (this != &other) { 
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }


    HOST_DEVICE bool operator<(const Vec3& other) const
    {
        return (x < other.x) || ((x == other.x) && ((y < other.y) || (y == other.y) && (z < other.z)));
    }

    // HOST_DEVICE Vec3 abs()
    // {
    //     return Vec3(std::abs(x), std::abs(y), std::abs(z));
    // }


	HOST_DEVICE float squaredNorm() const
    {
        return x * x + y * y + z * z;
	}

	HOST_DEVICE float norm() const
	{
        return sqrt(squaredNorm());
	}

	HOST_DEVICE bool isZero() const
	{
        return (x == 0 && y == 0 && z == 0);
	}

	HOST_DEVICE void setZero()
	{
        x = 0;
        y = 0;
        z = 0;
	}
    
};

using Vec3f = Vec3<float>;

// struct Vec3f
// {
// 	float x;
// 	float y;
// 	float z;

// 	HOST_DEVICE Vec3f() : x(0.0f), y(0.0f), z(0.0f) {}

// 	HOST_DEVICE Vec3f(float x, float y, float z): x(x), y(y), z(z) {}

// 	// HOST_DEVICE Vec3f(XYZIdx xyz) : x(float(xyz.x_idx)), y(float(xyz.y_idx)), z(float(xyz.z_idx)) {}

// 	HOST_DEVICE Vec3f(Eigen::Vector3f vec) : x(vec.x()), y(vec.y()), z(vec.z()) {}

//     HOST_DEVICE Vec3f operator+(const Vec3f& other) const
//     {
//         return Vec3f(x + other.x, y + other.y, z + other.z);
//     }

// 	HOST_DEVICE Vec3f operator+(const Eigen::Vector3f& other) const
//     {
//         return Vec3f(x + other.x(), y + other.y(), z + other.z());
//     }

// 	HOST_DEVICE Vec3f operator+(const float other) const
//     {
//         return Vec3f(x + other, y + other, z + other);
// 	}

// 	// 重载加法运算符
//     HOST_DEVICE void operator+=(const Vec3f& other)
//     {
//         x += other.x;
//         y += other.y;
// 		z += other.z;
//         return;
//     }

// 	HOST_DEVICE void operator+=(const Eigen::Vector3f& other)
//     {
//         x += other.x();
//         y += other.y();
//         z += other.z();
//         return;
//     }

//     HOST_DEVICE void operator+=(const float other)
//     {
//         x += other;
//         y += other;
// 		z += other;
//         return;
//     }

// 	HOST_DEVICE Vec3f operator-(const Vec3f& other) const
//     {
//         return Vec3f(x - other.x, y - other.y, z - other.z);
//     }
	
// 	HOST_DEVICE Vec3f operator-(const Eigen::Vector3f& other) const
//     {
//         return Vec3f(x - other.x(), y - other.y(), z - other.z());
//     }

// 	HOST_DEVICE Vec3f operator-(const float other) const
//     {
//         return Vec3f(x - other, y - other, z - other);
// 	}

//     HOST_DEVICE void operator-=(const Vec3f& other)
//     {
//         x -= other.x;
// 		y -= other.y;
// 		z -= other.z;
//         return;
//     }

// 	HOST_DEVICE void operator-=(const Eigen::Vector3f& other)
//     {
//         x -= other.x();
//         y -= other.y();
//         z -= other.z();
//         return;
//     }

//     HOST_DEVICE void operator-=(const float other)
//     {
//         x -= other;
//         y -= other;
// 		z -= other;
//         return;
//     }

// 	HOST_DEVICE Vec3f operator*(const Vec3f& other) const
//     {
//         return Vec3f(x * other.x, y * other.y, z * other.z);
// 	}

// 	HOST_DEVICE Vec3f operator*(const Eigen::Vector3f& other) const
//     {
//         return Vec3f(x * other.x(), y * other.y(), z * other.z());
//     }

// 	HOST_DEVICE Vec3f operator*(const float other) const
//     {
//         return Vec3f(x * other, y * other, z * other);
// 	}

// 	HOST_DEVICE void operator*=(const Vec3f& other)
//     {
//         x *= other.x;
//         y *= other.y;
// 		z *= other.z;
//         return;
//     }

// 	HOST_DEVICE void operator*=(const Eigen::Vector3f& other)
//     {
//         x *= other.x();
//         y *= other.y();
//         z *= other.z();
//         return;
//     }

//     HOST_DEVICE void operator*=(const float other)
//     {
//         x *= other;
//         y *= other;
// 		z *= other;
//         return;
//     }

// 	HOST_DEVICE Vec3f operator/(const Vec3f& other) const
//     {
//         assert(other.x == 0 || other.y == 0 || other.z == 0);
//         return Vec3f(x / other.x, y / other.y, z / other.z);
// 	}

// 	HOST_DEVICE Vec3f operator/(const Eigen::Vector3f& other) const
//     {
//         assert(other.x() == 0 || other.y() == 0 || other.z() == 0);
//         return Vec3f(x / other.x(), y / other.y(), z / other.z());
//     }

// 	HOST_DEVICE Vec3f operator/(const float other) const
//     {
//         assert(other == 0);
//         return Vec3f(x / other, y / other, z / other);
// 	}

// 	HOST_DEVICE void operator/=(const Vec3f& other)
//     {
//         assert(other.x == 0 || other.y == 0 || other.z == 0);
//         x /= other.x;
//         y /= other.y;
// 		z /= other.z;
//         return;
//     }

// 	HOST_DEVICE void operator/=(const Eigen::Vector3f& other)
//     {
//         assert(other.x() == 0 || other.y() == 0 || other.z() == 0);
//         x /= other.x();
//         y /= other.y();
//         z /= other.z();
//         return;
//     }

//     HOST_DEVICE void operator/=(const float other)
//     {
//         assert(other == 0);
//         x /= other;
//         y /= other;
// 		z /= other;
//         return;
//     }

//     HOST_DEVICE Vec3f& operator=(const Vec3f& other) {
//         if (this != &other) { // 防止自我赋值
//             x = other.x;
//             y = other.y;
//             z = other.z;
//         }
//         return *this;
//     }

// 	HOST_DEVICE Vec3f& operator=(const Eigen::Vector3f vec){
// 		x = vec.x();
// 		y = vec.y();
// 		z = vec.z();
// 		return *this;
// 	}

//     HOST_DEVICE bool operator<(const Vec3f& other) const
//     {
//         return (x < other.x) || ((x == other.x) && ((y < other.y) || (y == other.y) && (z < other.z)));
//     }

//     HOST_DEVICE Vec3f abs()
//     {
//         return Vec3f(std::abs(x), std::abs(y), std::abs(z));
//     }

// 	HOST_DEVICE Eigen::Vector3f toEigen() const
//     {
//         return Eigen::Vector3f(x, y, z);
// 	}

// 	HOST_DEVICE float squaredNorm() const
//     {
//         return x * x + y * y + z * z;
// 	}

// 	HOST_DEVICE float norm() const
// 	{
//         return sqrt(squaredNorm());
// 	}

// 	HOST_DEVICE bool isZero() const
// 	{
//         return (x == 0 && y == 0 && z == 0);
// 	}

// 	HOST_DEVICE void setZero()
// 	{
//         x = 0;
//         y = 0;
//         z = 0;
// 	}
// };

struct Mat3f
{
    float data[9] = {0};
    
	HOST_DEVICE Mat3f()
    {
    }

	HOST_DEVICE Mat3f(float a, float b, float c,
		float d, float e, float f,
		float g, float h, float i)
    {
        data[0] = a;
        data[1] = b;
        data[2] = c;
        data[3] = d;
        data[4] = e;
        data[5] = f;
        data[6] = g;
        data[7] = h;
        data[8] = i;
    }

    HOST_DEVICE Mat3f(const Eigen::Matrix3f &mat)
	{
        for (size_t i = 0; i < 9; i++)
        {
            data[i] = mat.data()[i];
		}
    }

    HOST_DEVICE Mat3f operator+(const Mat3f& other) const
    {
        return Mat3f(
			data[0] + other.data[0], data[1] + other.data[1], 
			data[2] + other.data[2], data[3] + other.data[3],
            data[4] + other.data[4], data[5] + other.data[5], 
			data[6] + other.data[6], data[7] + other.data[7],
            data[8] + other.data[8]);
    }

    HOST_DEVICE Mat3f operator+(const float other) const
    {
        return Mat3f(data[0] + other, data[1] + other, data[2] + other, data[3] + other,
                     data[4] + other, data[5] + other, data[6] + other, data[7] + other,
                     data[8] + other);
    }

    HOST_DEVICE Mat3f operator-(const Mat3f& other) const
    {
        return Mat3f(
			data[0] - other.data[0], data[1] - other.data[1], 
			data[2] - other.data[2], data[3] - other.data[3],
            data[4] - other.data[4], data[5] - other.data[5], 
			data[6] - other.data[6], data[7] - other.data[7],
            data[8] - other.data[8]);
    }

    HOST_DEVICE Mat3f operator-(const float other) const
    {
        return Mat3f(data[0] - other, data[1] - other, data[2] - other, data[3] - other, data[4] - other,
                     data[5] - other, data[6] - other, data[7] - other, data[8] - other);
    }

    HOST_DEVICE Vec3f operator*(const Vec3f& other) const
    {
        return Vec3f(data[0] * other.x + data[1] * other.y + data[2] * other.z, 
			data[3] * other.x + data[4] * other.y + data[5] * other.z, 
			data[6] * other.x + data[7] * other.y + data[8] * other.z);
    }

    HOST_DEVICE Mat3f operator*(const float other) const
    {
        return Mat3f(data[0] * other, data[1] * other, data[2] * other, data[3] * other, data[4] * other,
                     data[5] * other, data[6] * other, data[7] * other, data[8] * other);
    }

    HOST_DEVICE Mat3f operator/(const Mat3f& other) const
    {
        assert(other.data[0] == 0 || other.data[1] == 0 || other.data[2] == 0 || 
			other.data[3] == 0 || other.data[4] == 0 || other.data[5] == 0 || 
			other.data[6] == 0 || other.data[7] == 0 || other.data[8]);
        return Mat3f(data[0] / other.data[0], data[1] / other.data[1],
					data[2] / other.data[2], data[3] / other.data[3],
					data[4] / other.data[4], data[5] / other.data[5],
					data[6] / other.data[6], data[7] / other.data[7],
					data[8] / other.data[8]);
    }

    HOST_DEVICE Mat3f operator/(const float other) const
    {
        assert(other == 0);
        return Mat3f(data[0] / other, data[1] / other, data[2] / other, data[3] / other, data[4] / other,
                     data[5] / other, data[6] / other, data[7] / other, data[8] / other);
    }

    HOST_DEVICE Mat3f& operator=(const Mat3f& other)
    {
        if (this != &other)
        {  // 防止自我赋值
            for (size_t i = 0; i < 9; i++)
            {
                data[i] = other.data[i];
			}
        }
        return *this;
    }

    HOST_DEVICE Mat3f& operator=(const Eigen::Matrix3f mat)
    {
		for (size_t i = 0; i < 9; i++)
		{
            data[i] = mat.data()[i];
		}
        return *this;
    } 

	HOST_DEVICE Eigen::Matrix3f toEigen() const
    {
        Eigen::Matrix3f r;
        r << data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8];
        return r;
    }

};

}