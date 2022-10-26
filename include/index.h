#pragma once

/// Convert between x,y,z <-> i
struct Index
{
	Index()
	{
		v = 0;
	}
	Index(const int v_)
	{
		v = v_;
	}
	Index(const unsigned char x_, const unsigned char y_, const unsigned char z_)
	{
		v = x_ | (y_ << 1) | (z_ << 2);
	}

	void operator=(int v_)
	{
		v = v_;
	}

	void operator++()
	{
		v++;
	}
	void operator++(int blah)
	{
		v++;
	}

	bool operator<(int a)
	{
		return v < a;
	}

	operator int()
	{
		return v;
	}

	union
	{
		struct
		{
			unsigned int x : 1;
			unsigned int y : 1;
			unsigned int z : 1;
		};
		unsigned char v;
	};
};
