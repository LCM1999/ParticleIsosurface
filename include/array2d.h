#pragma once

#include "vect.h"


template <int N, class T>
struct Array2D
{
	Array2D(){}

	template<class S>
	Array2D(Array2D<N, S>& a)
	{
		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				data[i][j] = (T)a.data[i][j];
			}
		}
	}

	~Array2D(){}

	T data[N][N] = {0};

	vect<N, T> row(int i)
	{
		vect<N, T> r;
		for (size_t j = 0; j < N; j++)
		{
			r[j] = data[i][j];
		}
		return r;
	}

	vect<N, T> col(int j)
	{
		vect<N, T> r;
		for (size_t i = 0; i < N; i++)
		{
			r[i] = data[i][j];
		}
		return r;
	}

	bool isScaleMat()
	{
		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				if (i == j && data[i][j] == 0)
				{
					return false;
				}
				else if (i != j && data[i][j] != 0)
				{
					return false;
				}
				else
				{
					continue;
				}
			}
		}
		return true;
	}

	void makeScaleMat(float scale)
	{
		for (int i = 0; i < N; i++)
		{
			data[i][i] = scale;
		}
	}

	void makeScaleMat(float x_scale, float y_scale, float z_scale)
	{
		if (N > 0)
			data[0][0] = x_scale;
		if (N > 1)
			data[1][1] = y_scale;
		if (N > 2)
			data[2][2] = z_scale;
	}

	template<int M, class S>
	void makeScaleMat(const vect<M, S>& v)
	{
		int n = min(N, M);
		for (int i = 0; i < n; i++)
		{
			data[i][i] = v.v[i];
		}
	}

	T &operator()(int x, int y)
	{
		assert(x >= 0 && x < N&& y >= 0 && y < N);
		return data[x][y];
	}

	template<int M, class S>
	void operator=(const Array2D<M, S>& a)
	{
		int n = min(N, M);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				data[i][j] = a.data[i][j];
			}
		}
	}

	template<class S>
	Array2D<N, T> operator+(const S a)
	{
		Array2D<N, T> r;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				r.data[i][j] = data[i][j] + a;
			}
		}
		return r;
	}

	template<class S>
	Array2D<N, T> operator-(const S a)
	{
		Array2D<N, T> r;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				r.data[i][j] = data[i][j] - a;
			}
		}
		return r;
	}

	template<class S>
	vect<N, T> operator*(const vect<N, S>& v)
	{
		vect<N, T> r;
		r.zero();
		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				r[i] += data[i][j] * v.v[j];
			}
		}
		return r;
	}

	Array2D<N, T> operator*(Array2D<N, T>& a)
	{
		Array2D<N, T> r;
		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				r.data[i][j] = row(i) * a.col(j);
			}
		}
		return r;
	}

	Array2D<N, T> operator*(const T a)
	{
		Array2D<N, T> r;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				r.data[i][j] = data[i][j] * a;
			}
		}
		return r;
	}

	Array2D<N, T> operator/(const T a)
	{
		Array2D<N, T> r;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				r.data[i][j] = data[i][j] / a;
			}
		}
		return r;
	}

	template<int M, class S>
	Array2D<N, T>& operator-(const Array2D<M, S>& a)
	{
		int n = min(N, M);
		Array2D<N, T> r;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				r.data[i][j] = data[i][j] - a.data[i][j];
			}
		}
		return r;
	}

	template<int M, class S>
	Array2D<N, T>& operator+(const Array2D<M, S>& a)
	{
		int n = min(N, M);
		Array2D<N, T> r;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				r.data[i][j] = data[i][j] + a.data[i][j];
			}
		}
		return r;
	}

	template<int M, class S>
	void operator+=(const Array2D<M, S>& a)
	{
		int n = min(N, M);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				data[i][j] += a.data[i][j];
			}
		}
	}

	void operator/=(const T a)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				data[i][j] /= a;
			}
		}
	}

	Array2D<N, T> transposed()
	{
		Array2D<N, T> r;
		for (size_t i = 0; i < N; i++)
		{
			r.data[i][i] = data[i][i];
			for (size_t j = i + 1; j < N; j++)
			{
				r.data[i][j] = data[j][i];
				r.data[j][i] = data[i][j];
			}
		}
		return r;
	}

	void clear()
	{
		width = height = 0;
		data.clear();
	}
};
