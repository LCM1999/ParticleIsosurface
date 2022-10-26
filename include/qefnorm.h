#pragma once

template <class Type, int n>
struct ArrayWrapper
{
	Type data [ n ] [ n ];
};

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
	void calcPoint (Type rvalue[], Type *w, ArrayWrapper<Type, n> &u) const;
	void calcPointDC (Type rvalue[], Type *mid) const;
};


//template <class Type, int n>
//void matInverse ( ArrayWrapper<Type, n> &mat, ArrayWrapper<Type, n> &rvalue, Type tolerance = 0.000001 );



#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);a[k][l]=h+s*(g-h*tau);

// this is the worst kind of hack, but the VC++ compiler can't do templates correctly
// In the case below, the compiler seems to be treating the variable "n" as a dynamic
// variable and not using the code transformation the template defines.
/*
template <class Type, int n>
void jacobi ( Type u[n][n], Type *d, Type v[n][n]);  <- Doesn't compile in VC++
*/

template <class Type, int n>
void jacobi ( ArrayWrapper<Type, n> &u, Type *d, ArrayWrapper<Type, n> &v )
{
	int j, iq, ip, i, k;
	Type tresh, theta, tau, t, sm, s, h, g, c;
	Type b [ n ];
	Type z [ n ];
	Type a [ n ] [ n ];

	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			a [ i ] [ j ] = u.data [ i ] [ j ];
		}
	}

	for ( ip = 0; ip < n; ip++ ) 
	{
		for ( iq = 0; iq < n; iq++ )
		{
			v.data [ ip ] [ iq ] = 0.0f;
		}
		v.data [ ip ] [ ip ] = 1.0f;
	}

	for ( ip = 0; ip < n; ip++ )
	{
		b [ ip ] = a [ ip ] [ ip ];
		d [ ip ] = b [ ip ];
		z [ ip ] = 0.0f;
	}

	for ( i = 1; i <= 50; i++ )
	{
		sm = 0.0f;
		for ( ip = 0; ip < n - 1; ip++ )
		{
			for ( iq = ip + 1; iq < n; iq++ )
			{
				sm += (Type)fabs ( a [ ip ] [ iq ] );
			}
		}

		// Problem can be happened in here
		// 0 in float could always be unaccurently
		if ( std::fabs(sm) < TOLERANCE )
		{
			// sort the stupid things and transpose

			// transpose
			for ( i = 0; i < n; i++ )
			{
				for ( j = 0; j < n; j++ )
				{
					a [ i ] [ j ] = v.data [ j ] [ i ];
				}
			}

			// stupid sort n^2... however n should always be small
			// bubble sort
			// largest first
			for ( i = 0; i < n; i++ )
			{
				for ( j = 0; j < n - i - 1; j++ )
				{
					if ( fabs ( d [ j ] ) < fabs ( d [ j + 1 ] ) )
					{
						sm = d [ j ];
						d [ j ] = d [ j + 1 ];
						d [ j + 1 ] = sm;

						for ( k = 0; k < n; k++ )
						{
							sm = a [ j ] [ k ];
							a [ j ] [ k ] = a [ j + 1 ] [ k ];
							a [ j + 1 ] [ k ] = sm;
						}
					}
				}
			}

			for ( i = 0; i < n; i++ )
			{
				for ( j = 0; j < n; j++ )
				{
					v.data [ i ] [ j ] = a [ i ] [ j ];
				}
			}

			return;
		}

		if ( i < 4 )
		{
			tresh = 0.2f * sm / ( n * n );
		}
		else
		{
			tresh = 0.0f;
		}

		for ( ip = 0; ip < n - 1; ip++ )
		{
			for ( iq = ip + 1; iq < n; iq++ ) 
			{
				g = 100.0f * (Type)fabs ( a [ ip ] [ iq ] );
				if ( i > 4 && ( fabs ( d [ ip ] ) + g ) == fabs ( d [ ip ] )
					&& ( fabs ( d [ iq ] ) + g ) == fabs ( d [ iq ] ) )
				{
					a [ ip ] [ iq ] = 0.0f;
				}
				else
				{
					if ( fabs ( a [ ip ] [ iq ] ) > tresh )
					{
						h = d [ iq ] - d [ ip ];
						if ( ( fabs ( h ) + g ) == fabs ( h ) )
						{
							t = ( a [ ip ] [ iq ] ) / h;
						}
						else
						{
							theta = 0.5f * h / ( a [ ip ] [ iq ] );
							t = 1.0f / (Type)( fabs ( theta ) + sqrt ( 1.0f + theta * theta ) );
							if ( theta < 0.0f ) 
							{
								t = -1.0f * t;
							}
						}

						c = 1.0f / (Type)sqrt ( 1 + t * t );
						s = t * c;
						tau = s / ( 1.0f + c );
						h = t * a [ ip ] [ iq ];
						z [ ip ] -= h;
						z [ iq ] += h;
						d [ ip ] -= h;
						d [ iq ] += h;
						a [ ip ] [ iq ] = 0.0f;
						for ( j = 0; j <= ip - 1; j++ )
						{
							ROTATE ( a, j, ip, j, iq )
						}
						for ( j = ip + 1; j <= iq - 1; j++ )
						{
							ROTATE ( a, ip, j, j, iq )
						}
						for ( j = iq + 1; j < n; j++ )
						{
							ROTATE ( a, ip, j, iq, j )
						}
						for ( j = 0; j < n; j++ )
						{
							ROTATE ( v.data, j, ip, j, iq )
						}
					}
				}
			}
		}

		for ( ip = 0; ip < n; ip++ )
		{
			b [ ip ] += z [ ip ];
			d [ ip ] = b [ ip ];
			z [ ip ] = 0.0f;
		}
	}

	printf ( "too many iterations in jacobi\n" );
	exit ( 1 );
}


template <class Type, int n>
void matInverse ( ArrayWrapper<Type, n> &mat, ArrayWrapper<Type, n> &rvalue, Type tolerance = 1e-9)
{
	Type w[n];
	ArrayWrapper<Type, n> u;

	// there is an implicit assumption that mat is symmetric and real
	// U and V in the SVD will then be the same matrix whose rows are the eigenvectors of mat
	// W will just be the eigenvalues of mat
	int i, j, k;

	jacobi<Type, n> ( mat, w, u );

	// pseudo invert eigenvalues
	for ( i = 1; i < n; i++ )
	{
		if ( fabs ( w [ i ] / w [ 0 ] ) < tolerance )
			w [ i ] = 0;
		else
			w [ i ] = 1.0f / w [ i ];
	}
	w [ 0 ] = 1.0f / w [ 0 ];

	// multiply back to get inverted matrix
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			rvalue.data [ i ] [ j ] = 0;
			for ( k = 0; k < n; k++ )
			{
				rvalue.data [ i ] [ j ] += w [ k ] * u.data [ k ] [ i ] * u.data [ k ] [ j ];
			}
		}
	}
}

template <class Type, int n>
void matInverse ( ArrayWrapper<Type, n> &mat, ArrayWrapper<Type, n> &rvalue, Type tolerance, Type *w, ArrayWrapper<Type, n> &u )
{
	// there is an implicit assumption that mat is symmetric and real
	// U and V in the SVD will then be the same matrix whose rows are the eigenvectors of mat
	// W will just be the eigenvalues of mat
	int i, j, k;

	jacobi<Type, n> ( mat, w, u );

	// pseudo invert eigenvalues
	for ( i = 1; i < n; i++ )
	{
		if ( fabs ( w [ i ] / w [ 0 ] ) < tolerance )
			w [ i ] = 0;
		else
			w [ i ] = 1.0f / w [ i ];
	}
	w [ 0 ] = 1.0f / w [ 0 ];

	// multiply back to get inverted matrix
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			rvalue.data [ i ] [ j ] = 0;
			for ( k = 0; k < n; k++ )
			{
				rvalue.data [ i ] [ j ] += w [ k ] * u.data [ k ] [ i ] * u.data [ k ] [ j ];
			}
		}
	}
}


template <class Type, int n>
void matInverseDC ( ArrayWrapper<Type, n> &mat, ArrayWrapper<Type, n> &rvalue, Type tolerance, Type *w, ArrayWrapper<Type, n> &u )
{
	// there is an implicit assumption that mat is symmetric and real
	// U and V in the SVD will then be the same matrix whose rows are the eigenvectors of mat
	// W will just be the eigenvalues of mat
	int i, j, k;

	jacobi<Type, n> ( mat, w, u );

	// pseudo invert eigenvalues
	for ( i = 0; i < n; i++ )
	{
		if ( fabs ( w [ i ]) < tolerance )
			w [ i ] = 0;
		else
			w [ i ] = 1.0f / w [ i ];
	}

	// multiply back to get inverted matrix
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			rvalue.data [ i ] [ j ] = 0;
			for ( k = 0; k < n; k++ )
			{
				rvalue.data [ i ] [ j ] += w [ k ] * u.data [ k ] [ i ] * u.data [ k ] [ j ];
			}
		}
	}
}


template <class Type, int n>
void QEFNormal<Type, n>::combineSelf ( Type *eqn )
{
	int i, j;
	int index;

//	eqn [ n ] *= -1;
	index = 0;
	for ( i = 0; i < n + 1; i++ )
	{
		for ( j = i; j < n + 1; j++ )
		{
			data [ index ] += eqn [ i ] * eqn [ j ];
			index++;
		}
	}
//	eqn [ n ] *= -1;
}

template <class Type, int n>
void QEFNormal<Type, n>::calcPoint (Type rvalue[], Type *w, ArrayWrapper<Type, n> &u) const
{
	ArrayWrapper<Type, n> a;
	Type b [ n ];

	for (int i = 0; i < n; i++ )
	{
		int index = ( ( 2 * n + 3 - i ) * i ) / 2;
		for (int j = i; j < n; j++ )
		{
			a.data [ i ] [ j ] = data [ index + j - i ];
			a.data [ j ] [ i ] = a.data [ i ] [ j ];
		}

		b [ i ] = -data [ index + n - i ];
	}

	ArrayWrapper<Type, n> inv;
	::matInverse<Type, n> ( a, inv, 1e-6, w, u);

	for (int i = 0; i < n; i++ )
	{
		rvalue [ i ] = 0;
		for (int j = 0; j < n; j++ )
			rvalue [ i ] += inv.data [ j ] [ i ] * b [ j ];
	}

	// sort eigenvectors in descending order (bubble sort)
	for (int i = 0; i < n-1; i++)
	{
		for (int j = 0; j < n-1; j++)
		{
			if (w[j] < w[j+1])
			{
				swap(w[j], w[j+1]);
				for (int k = 0; k < n; k++)
				{
					swap(u.data[j][k], u.data[j+1][k]);
				}
			}
		}
	}
}


template <class Type, int n>
void QEFNormal<Type, n>::calcPointDC (Type rvalue[], Type *mid) const
{
	ArrayWrapper<Type, n> a, u;
	Type b [ n ], newB[n], w[n];

	for (int i = 0; i < n; i++ )
	{
		int index = ( ( 2 * n + 3 - i ) * i ) / 2;
		for (int j = i; j < n; j++ )
		{
			a.data [ i ] [ j ] = data [ index + j - i ];
			a.data [ j ] [ i ] = a.data [ i ] [ j ];
		}

		b [ i ] = -data [ index + n - i ];
	}
	
	for (int i = 0; i < n; i++ )
	{
		newB [ i ] = b [ i ];
		for (int j = 0; j < n; j++ )
			newB [ i ] -= a.data[ i ] [ j ] * mid[ j ];
	}

	ArrayWrapper<Type, n> inv;
	::matInverseDC<Type, n> ( a, inv, .01, w, u);

	for (int i = 0; i < n; i++ )
	{
		rvalue [ i ] = mid[i];
		for (int j = 0; j < n; j++ )
			rvalue [ i ] += inv.data [ j ] [ i ] * newB [ j ];
	}
}