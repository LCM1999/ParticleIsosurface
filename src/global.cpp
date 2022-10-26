#include "global.h"
#include "parse.h"
#include "iso_method_ours.h"

Global g;

float pi = 3.1415926535f;

Global::Global()
{
#ifdef USE_DMT

	screen = 0;
	terrain_widget = 0;
	widget = 0;

	draw_debug = true;
	draw_surf = true;
	draw_lines = false;
	cursor(0.593750000, 0.281250000, 0.250000000);

	tris_drawn = 0;
	frame_time = 100;
	running = true;
	camera_locked = false;

#ifdef _DEBUG
	fullscreen = false;
	g.scr_width = 800;
	g.scr_height = 600;
#else
	fullscreen = false;
	g.scr_width = 1500;
	g.scr_height = 950;
	//g.scr_width = 800;
	//g.scr_height = 600;
#endif
#endif // USE_DMT

	method = 0;
	ourRoot = 0;
}

