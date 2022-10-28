#pragma once

template <class Type, int n>
class QEFNormal
{
public:
	Type data [ ( n + 1 ) * ( n + 2 ) / 2 ];

	void zero ( void )
	{
		for (int i = 0; i < ( n + 1 ) * ( n + 2 ) / 2; i++ )
			data [ i ] = 0;
	}

	void combineSelf ( Type *eqn );
};

template <class Type, int n>
void QEFNormal<Type, n>::combineSelf ( Type *eqn )
{
	int i, j;
	int index;

	index = 0;
	for ( i = 0; i < n + 1; i++ )
	{
		for ( j = i; j < n + 1; j++ )
		{
			data [ index ] += eqn [ i ] * eqn [ j ];
			index++;
		}
	}
}
