#include "Coord_t.h"
#include "coope_fit.h"
#include "math.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#include <time.h>

// compile with cc -fPIC -shared -o ransac_detector.so ransac_detector.c -L.
// -lcoope_fit -Wl,-rpath .

#define timing(f, text)                                                        \
  clock_t start = clock();                                                     \
  f;                                                                           \
  clock_t diff = clock() - start;                                              \
  int msec = diff * 1000 / CLOCKS_PER_SEC;                                     \
  printf("%s took msecs: %d\n", text, msec);

// define structs
typedef struct {
  uint8_t r; // in [0,255]
  uint8_t g; // in [0,255]
  uint8_t b; // in [0,255]
} Color_t;

typedef struct {
  double h_min;   // hue as an angle between 0 and 359
  double h_max;   // hue tolerance (+-) as an angle between 0 and 359
  double sat_min; // a fraction between 0 and 1
  double val_min; // a fraction between 0 and 1
} Ball_t;

typedef struct {
  double r; // a fraction between 0 and 1
  double g; // a fraction between 0 and 1
  double b; // a fraction between 0 and 1
} rgb_t;

typedef struct {
  double h; // angle in degrees
  double s; // a fraction between 0 and 1
  double v; // a fraction between 0 and 1
} hsv_t;

typedef struct {
  int x;
  int y;
} IntCoord_t;

typedef struct {
  size_t length;
  size_t allocated;
  IntCoord_t *coords;
} IntCoords_t;

typedef struct {
  size_t length;
  size_t allocated;
  Coord_t *coords;
} Coords_t;

typedef struct {
  size_t length;
  size_t allocated;
  int *indexes;
} Indexes_t;

// define constants
#define R 0
#define G 1
#define B 2

#define NONE 255

#define NEW_GROUP_LEN 32
#define INVALID_INTCOORD -666666

// any other resolution seems really bad
#define COLOR_RESOLUTION 256

#define MIN_VALUE 0.5

// define macros
#define min(a, b) ((a < b) ? a : b)
#define max(a, b) ((a > b) ? a : b)

#define image_index(x, y) ((y * width) + x)
#define pixel(im, x, y) (im + image_index(x, y) * 3)

// define global variables
// this is here so that you can't accidently ask Python for more colors
// than we have values for
uint8_t number_of_colors = 0;

// translates an RGB value to table index of said value
int index_from_rgb(Color_t pixel) {
  return pixel.r * COLOR_RESOLUTION * COLOR_RESOLUTION /
             (256 / COLOR_RESOLUTION) +
         pixel.g * COLOR_RESOLUTION / (256 / COLOR_RESOLUTION) +
         pixel.b / (256 / COLOR_RESOLUTION);
}

// converts RGB color to HSV
hsv_t rgb2hsv(rgb_t in) {
  hsv_t out;
  double min, max, delta;

  min = in.r < in.g ? in.r : in.g;
  min = min < in.b ? min : in.b;

  max = in.r > in.g ? in.r : in.g;
  max = max > in.b ? max : in.b;

  out.v = max; // v
  delta = max - min;
  if (delta < 0.00001) {
    out.s = 0;
    out.h = 0; // undefined, maybe nan?
    return out;
  }
  if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
    out.s = (delta / max); // s
  } else {
    // if max is 0, then r = g = b = 0
    // s = 0, h is undefined
    out.s = 0.0;
    out.h = NAN; // its now undefined
    return out;
  }
  if (in.r >= max)                 // > is bogus, just keeps compiler happy
    out.h = (in.g - in.b) / delta; // between yellow & magenta
  else if (in.g >= max)
    out.h = 2.0 + (in.b - in.r) / delta; // between cyan & yellow
  else
    out.h = 4.0 + (in.r - in.g) / delta; // between magenta & cyan

  out.h *= 60.0; // degrees

  if (out.h < 0.0)
    out.h += 360.0;

  return out;
}

void init_table(uint8_t *rgb_to_balls_map, Ball_t *ball_params, int param_len) {
  /**
   * @brief generates a table of which RGB values belong to which ball, while
   * balls are defined in HSV (see the struct)
   * @param rgb_to_balls_map a preallocated array of 256x256x256 (or less,
   * depending on COLOR_RESOLUTION) that will map RGB to ball index (255 if no
   * ball fits)
   * @param ball_params ball definitions
   * @param param_len number of balls in the previous array
   * @retval None
   */
  for (int r = 0; r < COLOR_RESOLUTION; r++) {
    for (int g = 0; g < COLOR_RESOLUTION; g++) {
      for (int b = 0; b < COLOR_RESOLUTION; b++) {
        // convert RGB to a double between 0 and 1, only take every i.e. fourth
        // value
        double rd = 256 / COLOR_RESOLUTION * (double)r / 256;
        double gd = 256 / COLOR_RESOLUTION * (double)g / 256;
        double bd = 256 / COLOR_RESOLUTION * (double)b / 256;
        rgb_t pixelrgb = {rd, gd, bd};
        hsv_t pixelhsv = rgb2hsv(pixelrgb);
        Color_t pixel = {r, g, b};
        // for each ball, check if pixel color is within the limits and if so,
        // set its color in the table
        uint8_t found = 0;
        for (int c = 0; c < param_len; c++) {
          Ball_t ball = ball_params[c];
          if (((ball.h_min < pixelhsv.h && ball.h_max > pixelhsv.h) ||
               (ball.h_min < pixelhsv.h + 360 &&
                ball.h_max > pixelhsv.h + 360)) &&
              ball.sat_min < pixelhsv.s && pixelhsv.v > ball.val_min) {
            if (c == 1) {
              // printf("1");
            }
            rgb_to_balls_map[index_from_rgb(pixel)] = c;
            found = 1;
            break;
          }
        }
        if (!found) {
          rgb_to_balls_map[r * COLOR_RESOLUTION * COLOR_RESOLUTION +
                           g * COLOR_RESOLUTION + b] = NONE;
        }
      }
    }
  }
  number_of_colors = param_len;
  printf("C program received %d colors:\n", number_of_colors);
  for (int i = 0; i < number_of_colors; ++i) {
    printf("Color %d: H_MIN: %f, H_MAX:%f, SAT_MIN:%f, VAL_MIN: %f\n", i,
           ball_params[i].h_min, ball_params[i].h_max, ball_params[i].sat_min,
           ball_params[i].val_min);
  }
}

void *possibly_resize_ptr(void *ptr, size_t *ptr_size, size_t min_size,
                          size_t type_size) {
  /**
   * @brief resizes pointer if it's smaller than min_size
   * @note
   * @retval (possibly resized) pointer
   */

  if (*ptr_size < min_size) {
    *ptr_size = min_size;
    ptr = realloc(ptr, min_size * type_size);
  }
  return ptr;
}

void *possibly_double_ptr_size(void *ptr, size_t *ptr_size, size_t min_size,
                               size_t type_size) {
  /**
   * @brief doubles pointer size if it's not bigger than min_size
   * @note
   * @retval (possibly resized) pointer
   */
  if (*ptr_size <= min_size) {
    *ptr_size *= 2;
    ptr = realloc(ptr, (*ptr_size) * type_size);
  }
  return ptr;
}

void get_segmentation_mask(uint8_t *img, int width, int height, uint8_t *mask,
                           Ball_t *ball) {

  /**
   * @brief generates a segmentation mask for the ball specified
   * @param mask preallocated mask of the same dimensions as img, where the
   * result will be
   * @note as of right now, the img is already 3*8bit HSV, ball is also
   * specified that way
   * @retval None
   */

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      Color_t *c = (Color_t *)pixel(img, x, y); // 3x8bit HSV
      hsv_t hsv={((float)c->r)/256*360,((float)c->g)/256,((float)c->b)/256};
      if (((ball->h_min < hsv.h && ball->h_max > hsv.h) ||
           (ball->h_min < hsv.h + 256 && ball->h_max > hsv.h + 256)) &&
          ball->sat_min < hsv.s && hsv.v > ball->val_min) {
        mask[image_index(x, y)] = 0;
      } else {
        mask[image_index(x, y)] = NONE;
      }
    }
  }
}

void get_ball_pixels(uint8_t *img, int width, int height, int number_of_colors,
                     uint8_t *rgb_to_balls_map, int step, uint8_t *mask,
                     IntCoords_t *ball_pixels) {
  /**
   * @brief finds where balls is found in the image
   * @param step allows to only look at every n-th pixel (to save time)
   * @param mask preallocated array of the same size as img (width*height),
   * where result is written
   * @param ball_pixels an array of intcoords (one per color) where coords are
   * stored
   * @retval None
   */
  for (int i = 0; i < number_of_colors; ++i) {
    ball_pixels[i].coords =
        possibly_resize_ptr(ball_pixels[i].coords, &ball_pixels[i].allocated,
                            NEW_GROUP_LEN, sizeof(IntCoord_t));
    ball_pixels[i].length = 0;
  }

  // walk through the picture
  for (int y = 0; y < height; y = y + step) {
    for (int x = 0; x < width; x = x + step) {
      // set the pixel in segmentation mask to its corresponding ball (or NONE)
      Color_t *c = (Color_t *)pixel(img, x, y);
      // printf("R: %d, G: %d, B: %d\n",c->r, c->g, c->b);
      int color_index = index_from_rgb(*c);
      int ball_color = rgb_to_balls_map[color_index];
      mask[image_index(x, y)] = ball_color;
      if (ball_color != NONE) {
        // printf("here\n");
        IntCoords_t *current = ball_pixels + ball_color;
        // make sure enough space is available and save the coords
        current->coords =
            possibly_double_ptr_size(current->coords, &current->allocated,
                                     current->length, sizeof(IntCoord_t));
        current->coords[current->length].x = x;
        current->coords[current->length].y = y;
        current->length++;
        // printf("%d\n",current->length);
      }
    }
  }
  // for (int i = 0; i < number_of_colors; ++i) {
  //   printf("%d\n",ball_pixels[i].length);
  //   for (int j = 0; j < ball_pixels[i].length; ++j) {
  //     printf("(%d,%d) ==> [%d,%d]\n", i, j, ball_pixels[i].coords[j].x,
  //            ball_pixels[i].coords[j].y);
  //   }
  // }
}

uint8_t *get_neighbour_values(uint8_t *segmentation_mask, int width, int height,
                              int x, int y, int step) {
  /**
   * @brief finds the values of the neighbours
   * @param x x-coord of the pixel
   * @param y y-coord of the pixel
   * @param step should be the same as in get_ball_pixels
   * @retval array[9] values of all the neighbouring pixels in the segmentation
   * mask
   */
  static uint8_t neighbours[9]; // static to we can return it
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      int x_index = x + dx * step;
      int y_index = y + dy * step;
      int n_index = (dx + 1) * 3 + (dy + 1);
      if (x_index < 0 || x_index >= width || y_index < 0 || y_index >= height) {
        neighbours[n_index] = NONE;
      } else {
        neighbours[n_index] = segmentation_mask[image_index(x_index, y_index)];
      }
    }
  }
  return neighbours;
}

float distance_f2(float x0, float y0, float x1, float y1) {
  /**
   * @brief
   * @note
   * @retval ||(x0,y0)-(x1,y1)||^2
   */
  float dx = x0 - x1;
  float dy = y0 - y1;
  return dx * dx + dy * dy;
}

float distance_coords2(Coord_t *p1, Coord_t *p2) {
  /**
   * @brief
   * @note
   * @retval ||p1-p2||^2
   */
  return distance_f2(p1->x, p1->y, p2->x, p2->y);
}

bool valid_coord(Coord_t c) { return !isnan(c.x) && !isnan(c.y); }

bool valid_intcoord(IntCoord_t c) {
  return c.x != INVALID_INTCOORD && c.y != INVALID_INTCOORD;
}

bool is_border_pixel(uint8_t *neighbours) {
  /**
   * @brief decides whether pixel is a border pixel
   * @param neighbours array[9], while [4] is understood to be in the middle
   * @note   assumes neighbours generated by get_neighbour_values
   * @retval if the pixel is from the border
   */
  int pixel_value = neighbours[4];
  bool same_exists = false;
  bool different_exists = false;
  for (int j = 0; j < 9; ++j) {
    if (j == 4) {
      continue;
    } // don't consider the pixel itself
    same_exists = same_exists || neighbours[j] == pixel_value;
    different_exists = different_exists || neighbours[j] != pixel_value;
  }
  return same_exists && different_exists;
}

void get_border(uint8_t *segmentation_mask, int width, int height,
                IntCoords_t *ball_pixels, Coords_t *prev_pos, int step,
                float max_dx2, uint8_t *border_mask, uint8_t *group_mask,
                Indexes_t **groups, IntCoords_t *border) {
  /**
   * @brief finds border pixels, i.e. pixels with both background and balls next
   *to them
   * @details
   * @param prev_pos if valid values provided, the algorithm tries to group the
   *pixels to make ransac faster
   * @param max_dx2 determines how far to look from previous position for new
   *candidates (squared)
   * @param groups a list of arrays (one per valid prev_pos) of possible
   *candidates for new positions of each of the balls, saved as indexes of
   *border_mask
   * @note the function assumes that the number of balls == len(prev_pos)
   * @retval None
   */

  // first set up the arrays appropriately
  if (border->allocated == 0) {
    border->coords = malloc(NEW_GROUP_LEN * sizeof(IntCoord_t));
    border->allocated = NEW_GROUP_LEN;
  }
  border->length = 0;
  for (int i = 0; i < prev_pos->length; ++i) {
    (*groups)[i].length = 0;
    (*groups)[i].indexes =
        possibly_resize_ptr((*groups)[i].indexes, &(*groups)[i].allocated,
                            NEW_GROUP_LEN, sizeof(int));
  }

  for (int i = 0; i < ball_pixels->length; ++i) {
    int x = ball_pixels->coords[i].x;
    int y = ball_pixels->coords[i].y;
    uint8_t *neighbours =
        get_neighbour_values(segmentation_mask, width, height, x, y, step);
    if (is_border_pixel(neighbours)) { // border pixel
      // increase array size if necessary and copy x,y to boorder_coords
      border->coords =
          possibly_double_ptr_size(border->coords, &border->allocated,
                                   border->length, sizeof(IntCoord_t));
      border->coords[border->length].x = x;
      border->coords[border->length].y = y;
      if (border_mask) {
        border_mask[image_index(x, y)] = 1;
      }

      // add to possible candidates for each of the balls (based on previous
      // position) to speed up RANSAC
      for (int ball_index = 0; ball_index < prev_pos->length; ++ball_index) {
        // skip if previous pos not provided
        if (!valid_coord(prev_pos->coords[ball_index])) {
          continue;
        }
        float distance2 = distance_f2(x, y, prev_pos->coords[ball_index].x,
                                      prev_pos->coords[ball_index].y);
        // let's suppose that the ball moved at max by sqrt(max_dx2)
        // (if it did move more, it will have to be found without speedup)
        if (groups && distance2 < max_dx2) {
          Indexes_t *current = (*groups) + ball_index;
          // make sure there is available memory
          current->indexes =
              possibly_double_ptr_size(current->indexes, &current->allocated,
                                       current->length, sizeof(int));

          current->indexes[current->length++] = border->length;
          if (group_mask) {
            // if more balls are possible at the same place, this will only save
            // the last one but oh well...
            group_mask[image_index(x, y)] = ball_index;
          }
        }
      }
      border->length++;
    }
  }
}

int rand_lim(int limit) {
  /**
   * @brief random number between 0 and limit inclusive.
   * @note from
   * https://stackoverflow.com/questions/2999075/generate-a-random-number-within-range/2999130#2999130
   * @param  limit: upper limit
   * @retval generated number
   */

  int divisor = RAND_MAX / (limit + 1);
  int retval;

  do {
    retval = rand() / divisor;
  } while (retval > limit);

  return retval;
}

void circle_intersections(int x0, int y0, int r0, int x1, int y1, int r1,
                          Coord_t *centers_ret) {
  /**
   * @brief finds intersections of the two circles specified
   * @note  from
   * https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
   * (adapted for C and this task) this also makes sure that the picked circles
   * are not close enough, so THIS IS NOT A GENERAL ALGORITHM AND WILL NOT WORK
   * WELL IF USED FOR DIFFERENT TASKS circle 1: (x0, y0), radius r0; circle 2:
   * (x1, y1), radius r1
   * @retval none
   */

  float d2 = distance_f2(x0, y0, x1, y1);

  bool circles_too_far = d2 > pow(r0 + r1, 2);
  bool one_circle_within_other = d2 < pow(r0 - r1, 2);
  bool coincident_circles = d2 == 0 && r0 == r1;
  // points too close to each other to reliably estimate the circle
  // as a rule of thumb, we want them at least 20 deg apart
  bool too_close_for_estimate = d2 < pow(r0 / 3, 2);

  if (!circles_too_far && !one_circle_within_other && !coincident_circles &&
      !too_close_for_estimate) {
    float d = sqrt(d2);
    float a = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
    float h = sqrt(r0 * r0 - a * a);
    float x2 = x0 + a * (x1 - x0) / d;
    float y2 = y0 + a * (y1 - y0) / d;

    centers_ret[0].x = x2 + h * (y1 - y0) / d;
    centers_ret[0].y = y2 - h * (x1 - x0) / d;
    centers_ret[1].x = x2 - h * (y1 - y0) / d;
    centers_ret[1].y = y2 + h * (x1 - x0) / d;
  }
  return;
}

void find_center_candidates(IntCoords_t *border, float r,
                            Coord_t *centers_ret) {
  /**
   * @brief  finds possible center candidate
   * @param centers_ret preallocated array of length 2, where the result will be
   * stored
   * @note
   * @retval None
   */
  centers_ret[0].x = NAN;
  centers_ret[0].y = NAN;
  centers_ret[1].x = NAN;
  centers_ret[1].y = NAN;
  int max_iterations = 500;
  int failed_to_find_intersection = 0;
  while (!valid_coord(centers_ret[0]) &&
         failed_to_find_intersection < max_iterations) {
    int index1 = rand_lim(border->length - 1);
    int index2 = index1;
    // no point in picking the same index twice
    while (index2 == index1) {
      index2 = rand_lim(border->length - 1);
    }
    circle_intersections(border->coords[index1].x, border->coords[index1].y, r,
                         border->coords[index2].x, border->coords[index2].y, r,
                         centers_ret);
    failed_to_find_intersection++;
  }
}

void ransac(IntCoords_t *border, float r, float min_dist, float max_dist,
            int max_iter, int confidence_thrs, bool verbose,
            Coord_t *best_model) {
  /**
   * @brief  finds ball center using RANSAC
   * @param border ball border
   * @param r ball radius
   * @param min_dist specifies the minimum distance from center for a border
   * pixel to be considered part of the model
   * @param max_dist specifies the maximum distance from center for a border
   * pixel to be considered part of the model
   * @param max_iter maximum number of iterations to be performed, if
   * confidence_thrs is not reached
   * @param confidence_thrs the amount of pixels that have to be modeled for the
   * algorithm to stop sooner than max_iter is reached
   * @param best_model return value
   * @note
   * @retval None
   */
  best_model->x = NAN;
  best_model->y = NAN;
  // no point in looking for the ball if there are not too many pixels
  if (border->length < confidence_thrs) {
    if (verbose) {
      printf("Out of balls!\n");
    }
    return;
  }

  int best_inliers = 0;
  static Coord_t centers[2];
  for (int iteration = 0; iteration < max_iter; ++iteration) {
    find_center_candidates(border, r, centers);
    bool same_centers = distance_f2(centers[0].x, centers[0].y, centers[1].x,
                                    centers[1].y) < 0.1;
    for (int i = 0; i < (same_centers ? 1 : 2); ++i) {
      int inliers = 0;
      for (int j = 0; j < border->length; ++j) {
        float distance2 = distance_f2(border->coords[j].x, border->coords[j].y,
                                      centers[i].x, centers[i].y);
        if (distance2 < pow(max_dist, 2) && distance2 > pow(min_dist, 2)) {
          inliers++;
        }
      }
      if (inliers > best_inliers) {
        best_inliers = inliers;
        *best_model = centers[i];
      }
    }
    if (best_inliers >= confidence_thrs) {
      if (verbose) {
        printf("Reached confidence threshold!\n");
      }
      break;
    }
  }
  if (verbose) {
    printf("Best inliers: %d\n", best_inliers);
  }
}

bool in_group(Coord_t *model, Coord_t *previous_center, float max_dx2,
              float max_dist) {
  /**
   * @brief  decides whether all modeled pixels can by found in the previously
   * filtered group or it is necessary to look in all border coords
   * @note
   * @retval if all pixels will be in groups
   */
  bool new_center_close = sqrt(distance_coords2(model, previous_center)) <
                          (sqrt(max_dx2) - max_dist);
  return valid_coord(*previous_center) && new_center_close;
}

void find_modeled_pixels(Coord_t *model, float min_dist, float max_dist,
                         IntCoords_t *set_to_check, Indexes_t *modeled) {
  /**
   * @brief find all the pixels the are modeled
   * @param min_dist specifies the minimum distance from center for a border
   * pixel to be considered part of the model
   * @param max_dist specifies the maximum distance from center for a border
   * pixel to be considered part of the model
   * @param set_to_check pixels from this set will be inspected
   * @param modeled array where indexes of those pixels that are modeled are
   * saved
   * @note
   * @retval None
   */
  modeled->length = 0;
  modeled->indexes = possibly_resize_ptr(modeled->indexes, &modeled->allocated,
                                         set_to_check->length, sizeof(int));
  for (int i = 0; i < set_to_check->length; ++i) {
    float distance2 =
        distance_f2((float)set_to_check->coords[i].x,
                    (float)set_to_check->coords[i].y, model->x, model->y);
    if (distance2 > pow(min_dist, 2) && distance2 < pow(max_dist, 2)) {
      modeled->indexes[modeled->length++] = i;
    }
  }
}

void remove_pixels(IntCoords_t *border, Indexes_t *group, Indexes_t *modeled,
                   bool only_group) {
  /**
   * @brief  removes the pixels specified from border
   * @param modeled indexes of the pixels to be removed
   * @param only_group if true, indexes are understood to be border indexes, if
   * false, the true indexes are first obtained using group
   * @note
   * @retval None
   */
  for (int i = 0; i < modeled->length; ++i) {
    IntCoord_t *modeled_pixel_address;
    if (only_group) {
      modeled_pixel_address =
          border->coords + group->indexes[modeled->indexes[i]];
    } else {
      modeled_pixel_address = border->coords + modeled->indexes[i];
    }
    modeled_pixel_address->x = INVALID_INTCOORD;
    modeled_pixel_address->y = INVALID_INTCOORD;
  }
}

Coord_t lsq_on_modeled(IntCoords_t *coords, Indexes_t *modeled) {
 /**
  * @brief  tries to find a better fit using least squares
  * @note   computed using eigen (C++ library)
  * @retval lsq fit
  */
  static Coord_t *modeled_for_cpp;
  static size_t modeled_al = 0;
  modeled_for_cpp = possibly_resize_ptr(modeled_for_cpp, &modeled_al,
                                        modeled->length, sizeof(Coord_t));

  for (int i = 0; i < modeled->length; ++i) {
    modeled_for_cpp[i].x = (float)coords->coords[modeled->indexes[i]].x;
    modeled_for_cpp[i].y = (float)coords->coords[modeled->indexes[i]].y;
  }
  return coope_fit(modeled_for_cpp, (int)modeled->length);
}

void filter_group_coords(IntCoords_t *border, Indexes_t *group,
                         IntCoords_t *filtered) {
  /**
   * @brief  filters out coords that are not valid
   * @param group if not NULL, only those pixels listed in group will be
   * considered
   * @retval None
   */
  if (group) {
    filtered->coords =
        possibly_resize_ptr(filtered->coords, &filtered->allocated,
                            group->length, sizeof(IntCoord_t));
    for (int i = 0; i < group->length; ++i) {
      IntCoord_t group_pixel = border->coords[group->indexes[i]];
      if (valid_intcoord(group_pixel)) {
        filtered->coords[filtered->length++] = group_pixel;
      }
    }
  } else {
    filtered->coords =
        possibly_resize_ptr(filtered->coords, &filtered->allocated,
                            border->length, sizeof(IntCoord_t));
    for (int i = 0; i < border->length; ++i) {
      IntCoord_t group_pixel = border->coords[i];
      if (valid_intcoord(group_pixel)) {
        filtered->coords[filtered->length++] = group_pixel;
      }
    }
  }
}

void detect_ball(IntCoords_t *border, Indexes_t *group, Coord_t *prev_pos,
                 Coord_t *center_ransac, Coord_t *center_coope, float max_dx2,
                 float r, float min_dist, float max_dist, int max_iter,
                 int confidence_thrs, bool verbose) {
  /**
   * @brief  finds a ball in border (possibly group)
   * @note not a pure function, border will be changed (modeled pixels will be
   * removed)
   * @param border where to look for
   * @param group indexes of pixels in border to be considered (or none)
   * @param prev_pos the previous position of the ball (if known - otherwise
   * invalid)
   * @param center_ransac the ransac result will be stored there
   * @param center_coope the coope (least squares) center, which usually is more
   * precise, is stored there
   * @param r ball radius
   * @param min_dist specifies the minimum distance from center for a border
   * pixel to be considered part of the model
   * @param max_dist specifies the maximum distance from center for a border
   * pixel to be considered part of the model
   * @param max_iter maximum number of iterations to be performed, if
   * confidence_thrs is not reached
   * @param confidence_thrs the amount of pixels that have to be modeled for the
   * algorithm to stop sooner than max_iter is reached
   * @retval None
   */

  // declare statics and make sure to have enough memory
  static IntCoords_t filtered = {0, 0, NULL};
  filtered.length = 0;

  // find the ball and remove old pixels
  // it is not necessary to look among all the pixels, if previous position was
  // known and the ball has not moved too much
  filter_group_coords(border, group, &filtered);

  ransac(&filtered, r, min_dist, max_dist, max_iter, confidence_thrs, verbose,
         center_ransac);

  if (valid_coord(*center_ransac)) {
    // find out whether it is necessary to look in previously created group or
    // in all border coords
    bool only_group =
        group && in_group(center_ransac, prev_pos, max_dx2, max_dist);
    IntCoords_t *set_to_check;
    if (only_group) {
      set_to_check = &filtered;
    } else {
      set_to_check = border;
    }
    static Indexes_t modeled = {0, 0, NULL};

    find_modeled_pixels(center_ransac, min_dist, max_dist, set_to_check,
                        &modeled);
    *center_coope = lsq_on_modeled(set_to_check, &modeled);
    remove_pixels(border, group, &modeled, only_group);
  }
}

int compute_number_of_colors(int *ball_colors, int nr_of_balls) {
  int number_of_colors = 0;
  for (int i = 0; i < nr_of_balls; ++i) {
    if (ball_colors[i] >= number_of_colors) {
      number_of_colors = ball_colors[i] + 1;
    }
  }
  return number_of_colors;
}

void detect_balls(uint8_t *rgb_to_balls_map, uint8_t *img, int width,
                  int height, int step, Coords_t *prev_pos, int *ball_colors,
                  float max_dx2, float r, float min_dist, float max_dist,
                  int max_iter, int confidence_thrs, bool verbose,
                  Coord_t *centers_ransac, Coord_t *centers_coope) {
  /**
   * @brief finds balls using ransac and least squares
   * @param rgb_to_balls_map a preallocated array of 256x256x256 (or less,
   * depending on COLOR_RESOLUTION) that will map RGB to ball index (255 if no
   * ball fits)
   * @note nr of balls is defined by the length of prev_pos (invalid if not
   * known)
   * @param step allows to only look at every n-th pixel (to save time)
   * @param prev_pos the previous positions of the ball (if known - otherwise
   * invalid with NANs) - valid values will improve speed
   * @param ball_colors specifies what color each of the balls is
   * (e.g. for 2 balls of color 0, 1 of color 1 and 3 of color 2, the array
   *would be [0,0,1,2,2,2])
   * @param max_dx2 determines how far to look from previous position for new
   *candidates (squared)
   * @param r ball radius
   * @param min_dist specifies the minimum distance from center for a border
   * pixel to be considered part of the model
   * @param max_dist specifies the maximum distance from center for a border
   * pixel to be considered part of the model
   * @param max_iter maximum number of iterations to be performed, if
   * confidence_thrs is not reached
   * @param confidence_thrs the amount of pixels that have to be modeled for the
   * algorithm to stop sooner than max_iter is reached
   * @param center_ransac the ransac result will be stored there
   * @param center_coope the coope (least squares) center, which usually is more
   * precise, is stored there
   * @retval None
   */

  int number_of_colors =
      compute_number_of_colors(ball_colors, prev_pos->length);

  static uint8_t *segmentation_mask = NULL;
  static size_t segmentation_mask_al = 0;

  // alocate memory
  int image_size = width * height;
  segmentation_mask = possibly_resize_ptr(
      segmentation_mask, &segmentation_mask_al, image_size, sizeof(uint8_t));

  // find ball_pixels
  static IntCoords_t *ball_pixels = NULL;
  static size_t ball_pixels_l = 0;

  if (ball_pixels_l < number_of_colors) {
    ball_pixels = realloc(ball_pixels, number_of_colors * sizeof(IntCoords_t));
    for (int i = ball_pixels_l; i < number_of_colors; ++i) {
      ball_pixels[i].length = 0;
      ball_pixels[i].coords = malloc(NEW_GROUP_LEN * sizeof(IntCoord_t));
      ball_pixels[i].allocated = NEW_GROUP_LEN;
      ball_pixels_l = number_of_colors;
    }
  }

  get_ball_pixels(img, width, height, number_of_colors, rgb_to_balls_map, step,
                  segmentation_mask, ball_pixels);

  // find border
  static IntCoords_t border = {0, 0, NULL};
  static Indexes_t *groups = NULL;
  static int groups_l = 0;
  if (groups_l < prev_pos->length) {
    // create index list for each new ball of length NEW_GROUP_LEN
    groups = realloc(groups, prev_pos->length * sizeof(Indexes_t));
    for (int i = groups_l; i < prev_pos->length; ++i) {
      groups[i].indexes = malloc(NEW_GROUP_LEN * sizeof(int));
      groups[i].allocated = NEW_GROUP_LEN;
      groups[i].length = 0;
    }
    groups_l = prev_pos->length;
  }

  // now let's find all the balls
  static bool *skipped = NULL;
  static size_t skipped_al = 0;
  skipped =
      possibly_resize_ptr(skipped, &skipped_al, prev_pos->length, sizeof(bool));
  memset(skipped, true, prev_pos->length * sizeof(bool));
  for (int c = 0; c < number_of_colors; ++c) {
    // only create groups for balls of the same color
    // because groups aren't created for invalid previous coords, this is an
    // easy way to do so without sacrificing ball-group indexing
    Coords_t filtered_prev_pos = *prev_pos;
    for (int i = 0; i < number_of_colors; ++i) {
      if (ball_colors[i] != c) {
        filtered_prev_pos.coords[i].x = NAN;
        filtered_prev_pos.coords[i].y = NAN;
      }
    }

    get_border(segmentation_mask, width, height, ball_pixels + c,
               &filtered_prev_pos, step, max_dx2, NULL, NULL, &groups, &border);
    // printf("color %d, border length %d\n",c,border.length);
    // first, only balls with known previous positions are searched
    for (int i = 0; i < prev_pos->length; ++i) {
      if (!valid_coord(prev_pos->coords[i]) || !skipped[i] ||
          ball_colors[i] != c) {
        continue;
      }
      detect_ball(&border, groups + i, prev_pos->coords + i, centers_ransac + i,
                  centers_coope + i, max_dx2, r, min_dist, max_dist, max_iter,
                  confidence_thrs, verbose);
      if (valid_coord(centers_ransac[i])) {
        skipped[i] = false;
      }
    }
    // now (try to) find the rest
    for (int i = 0; i < prev_pos->length; ++i) {
      if (!skipped[i] || ball_colors[i] != c) {
        continue;
      }
      detect_ball(&border, NULL, prev_pos->coords + i, centers_ransac + i,
                  centers_coope + i, max_dx2, r, min_dist, max_dist, max_iter,
                  confidence_thrs, verbose);
      if (valid_coord(centers_ransac[i])) {
        skipped[i] = false;
      }
    }
  }

  // if the ball wasn't found, return NAN coords
  Coord_t not_found = {NAN, NAN};
  for (int i = 0; i < prev_pos->length; ++i) {
    if (skipped[i]) {
      *centers_ransac = not_found;
      *centers_coope = not_found;
    }
  }
}