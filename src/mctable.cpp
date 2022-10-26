#include "mctable.h"
#include <stdio.h>

vector<vector<int> > mc_cycles[256];
int mc_edges[12][2] = {{0,1},{2,3},{6,7},{4,5},{0,2},{4,6},{5,7},{1,3},{0,4},{1,5},{3,7},{2,6}};

char getChar ( FILE *fptr )
{
	char token;
	
	if ( fread ( &token, 1, 1, fptr ) != 1 )
	{
		// shit... end of file
		return 0;
	}

	while ( token == ' ' || token == '\r' ||
		token == '\n' || token == '\t' ) {
		if ( token == '\n' ) {
		}

		if ( fread ( &token, 1, 1, fptr ) != 1 )
		{
			// shit... end of file
			return 0;
		}
	}
	
	return token;
}

vector<int> getArray ( FILE *fptr )
{
	char token;
	vector<int> rvalue;
	char buffer [ 256 ];
	int pos = 0;
	
	token = getChar ( fptr ) ;
	while ( token != '}' ) {
		pos = 0;
		memset ( buffer, 0, 256 );

		while ( token != ',' && token != '}' ) {
			buffer [ pos++ ] = token;
      token = getChar ( fptr );
		}
		buffer [ pos ] = '\0';
		
    rvalue.push_back ( atoi ( buffer ) );
		
		if ( token == ',' ) {
      token = getChar ( fptr );
		}
	}

  return rvalue;
}

void initMCTable()
{
	if (!mc_cycles[7].empty())
		return;

	FILE *fptr = fopen ( "F:\\RepoCpp\\Isosurface_smiplify\\cycleTable.txt", "rt" );
	char token;
	int i;

	if ( fptr == NULL )
	{
		fprintf ( stderr, "ERROR: could not open cycleTable.txt\n" );
		exit ( 1 );
	}

	for ( i = 0; i < 256; i++ )
	{
		token = getChar ( fptr );
		if ( token != '{' )
		{
			fprintf ( stderr, "Error: { expected\n" );
			exit ( 1 );
		}

		token = getChar ( fptr );
		while ( token == '{' )
		{
			mc_cycles [ i ].push_back ( getArray ( fptr ) );
			token = getChar ( fptr );
			if ( token == ',' )
			{
				token = getChar ( fptr );
			}
		}
	}

	fclose ( fptr );
}