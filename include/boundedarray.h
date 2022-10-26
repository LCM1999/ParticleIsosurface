#pragma once

#include "assert.h"

template <class T, int elem_max = 9>
struct BoundedArray
{
	BoundedArray(){elem_size = 0;}

	int elem_size;
	T elem[elem_max];

	void push_back(const T &e){elem[elem_size++] = e;	assert(elem_size <= elem_max);}
	T &operator[](int i){return elem[i]; assert(i >= 0 && i < elem_size);}
	int size(){return elem_size;}
	void resize(int s){elem_size = s; assert(elem_size <= elem_max);}
	void clear(){elem_size = 0;}
	void reserve(int s){}
};
