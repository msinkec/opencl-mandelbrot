
__kernel void mandelbrot(__global unsigned char* img, 
                           const int max_iteration, const int width, const int height) {

	// Global coordinates
	int xg = get_global_id(0);
    int yg = get_global_id(1);

    // Check bounds
    if (xg >= width || yg >= height) {
        return;
    }
    
    float xtemp;
    unsigned char max = 255;
    
    float x0 = (float) xg / width * (float)3.5 - (float)2.5;
    float y0 = (float) yg / height * (float)2.0 - (float)1.0;

    float x = 0;
    float y = 0;

    int iter = 0;
    while ((x*x + y*y <= 4) && (iter < max_iteration)) {
        xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iter++;
    }

    // http://linas.org/art-gallery/escape/smooth.html
    int color = 1.0 + iter - log(log(sqrt(x*x + y * y))) / log(2.0);
    color = (8 * max * color) / max_iteration;
    if (color > max)
        color = max;
   
    img[4 * yg*width + 4 * xg + 0] = color; //Blue
    img[4 * yg*width + 4 * xg + 1] = color; // Green
    img[4 * yg*width + 4 * xg + 2] = color; // Red
    img[4 * yg*width + 4 * xg + 3] = 255;   // Alpha

}
