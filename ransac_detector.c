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

// generates a table of which RGB values belong to which ball, while balls are
// defined in HSV
void init_table(uint8_t *rgb_to_balls_map, Ball_t *ball_params, int param_len) {
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
  printf("\n");
  number_of_colors = param_len;
  printf("C program received %d colors:\n", number_of_colors);
  for (int i = 0; i < number_of_colors; ++i) {
    printf("Color %d: H_MIN: %f, H_MAX:%f, SAT_MIN:%f, VAL_MIN: %f\n", i,
           ball_params[i].h_min, ball_params[i].h_max, ball_params[i].sat_min,
           ball_params[i].val_min);
  }
}

// resizes pointer if it's smaller than min_size
void *possibly_resize_ptr(void *ptr, size_t *ptr_size, size_t min_size,
                          size_t type_size) {
  if (*ptr_size < min_size) {
    *ptr_size = min_size;
    ptr = realloc(ptr, min_size * type_size);
  }
  return ptr;
}

// doubles pointer if it's not bigger than min_size
void *possibly_double_ptr_size(void *ptr, size_t *ptr_size, size_t min_size,
                               size_t type_size) {
  if (*ptr_size <= min_size) {
    *ptr_size *= 2;
    ptr = realloc(ptr, (*ptr_size) * type_size);
  }
  return ptr;
}

void get_segmentation_mask(uint8_t *img, int width, int height, uint8_t *mask,
                           Ball_t *ball) {
  // as of right now, the img is already 3*8bit HSV, ball is also specified that
  // way
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      Color_t *c = (Color_t *)pixel(img, x, y);
      if (((ball->h_min < c->r && ball->h_max > c->r) ||
           (ball->h_min < c->r + 360 && ball->h_max > c->r + 360)) &&
          ball->sat_min < c->g && c->b > ball->val_min) {
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
  float dx = x0 - x1;
  float dy = y0 - y1;
  return dx * dx + dy * dy;
}

float distance_coords2(Coord_t *p1, Coord_t *p2) {
  return distance_f2(p1->x, p1->y, p2->x, p2->y);
}

bool valid_coord(Coord_t c) { return !isnan(c.x) && !isnan(c.y); }

bool valid_intcoord(IntCoord_t c) {
  return c.x != INVALID_INTCOORD && c.y != INVALID_INTCOORD;
}

bool is_border_pixel(uint8_t *neighbours) {
  // neighbours is expected implicitly to be of len 9 (i.e. all directions
  // including the pixel)
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

  /*
   *Returns border pixels, i.e. pixels with both background and balls next to
   *them. 'max_dx2' determines how far to look from previous position for new
   *candidates. In case previous positions are provided, the algorithm tries to
   *group the pixels to make ransac faster.
   * all coord lengths are //2, ie [[0,0],[1,2]]~[0,0,1,2] has length 2
   * return border_coords len
   */

  /*
 group indexes is an array of arrays as follows: [[0,1,2],[2,5,4,9],[],[2]]
 for the above array, values of the variables below would be
 group_index: [&[0,1.2],&[2,5,4,9],NULL,&[2]]
 group_index_als: [NEW_GROUP_LEN, NEW_GROUP_LEN, NEW_GROUP_LEN, NEW_GROUP_LEN]
 group_index_ls: [3,4,0,1]
 groups_l: 4
 */
  border->length = 0;
  for (int i = 0; i < ball_pixels->length; ++i) {
    int x = ball_pixels->coords[i].x;
    int y = ball_pixels->coords[i].y;
    // printf("looking at pixel %d [%d,%d]\n", i, x, y);
    uint8_t *neighbours =
        get_neighbour_values(segmentation_mask, width, height, x, y, step);
    if (is_border_pixel(neighbours)) { // border pixel
      // printf("is border_pixel\n");
      // increase array size if necessary and copy x,y to boorder_coords
      border->coords =
          possibly_double_ptr_size(border->coords, &border->allocated,
                                   border->length, sizeof(IntCoord_t));
      // printf("succesfully changed (or not) size to %d\n",border->allocated);
      border->coords[border->length].x = x;
      border->coords[border->length].y = y;
      if (border_mask) {
        border_mask[image_index(x, y)] = 1;
      }
      // printf("add found border\n");

      // add to possible candidates for each of the balls (based on previous
      // position) to speed up RANSAC
      for (int ball_index = 0; ball_index < prev_pos->length; ++ball_index) {
        // printf("looking at ball %d\n", ball_index);
        // skip if previous pos not provided
        if (!valid_coord(prev_pos->coords[ball_index])) {
          // printf("invalid coord!\n");
          continue;
        }
        // printf("valid coord!\n");
        float distance2 = distance_f2(x, y, prev_pos->coords[ball_index].x,
                                      prev_pos->coords[ball_index].y);
        // let's suppose that the ball moved at max by sqrt(max_dx2)
        // (if it did move more, it will have to be found without speedup)
        if (groups && distance2 < max_dx2) {
          // printf("adding pixel %d to ball %d\n", i, ball_index);
          Indexes_t *current = (*groups) + ball_index;
          // printf("current allocated %d\n",current->allocated);
          // make sure there is available memory
          current->indexes =
              possibly_double_ptr_size(current->indexes, &current->allocated,
                                       current->length, sizeof(int));
          // printf("added pixel %d to ball %d\n", i, ball_index);

          current->indexes[current->length++] = border->length;
          if (group_mask) {
            // if more balls are possible at the same place, this will only save
            // the last one but oh well...
            group_mask[image_index(x, y)] = ball_index;
          }
        }
      }
      border->length++;
      // printf("%d\n", border->length);
    }
  }
  // printf("border length out: %d\n", border->length);
}

int rand_lim(int limit) {
  /* return a random number between 0 and limit inclusive.
  from
  https://stackoverflow.com/questions/2999075/generate-a-random-number-within-range/2999130#2999130
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
  /* finds intersections of the two circles specified
 https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
 (adapted for c) circle 1: (x0, y0), radius r0 circle 2: (x1, y1), radius r1*/
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
  best_model->x = NAN;
  best_model->y = NAN;
  // printf("bcl: %d, c: %d\n", border->length, confidence_thrs);
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
  // decides whether all modeled pixels can by found in previously filtered
  // group or it is necessary to look in all border coords
  bool new_center_close = sqrt(distance_coords2(model, previous_center)) <
                          (sqrt(max_dx2) - max_dist);
  return valid_coord(*previous_center) && new_center_close;
}

// returns the length of modeled indexes
void find_modeled_pixels(Coord_t *model, Coord_t *previous_center,
                         float max_dx2, float min_dist, float max_dist,
                         IntCoords_t *set_to_check, Indexes_t *modeled) {
  // check if it is possible to only search in group coords

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
  static Coord_t *modeled_for_cpp;
  static size_t modeled_al = 0;
  modeled_for_cpp = possibly_resize_ptr(modeled_for_cpp, &modeled_al,
                                        modeled->length, sizeof(Coord_t));

  for (int i = 0; i < modeled->length; ++i) {
    // printf("[%f, %f]-", coords[i].x, coords[i].y);
    modeled_for_cpp[i].x = (float)coords->coords[modeled->indexes[i]].x;
    modeled_for_cpp[i].y = (float)coords->coords[modeled->indexes[i]].y;
  }
  return coope_fit(modeled_for_cpp, (int)modeled->length);
}

void filter_group_coords(IntCoords_t *border, Indexes_t *group,
                         IntCoords_t *filtered) {
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
      // printf("[%f, %f]\n", border_coords[i].x, border_coords[i].y);
      if (valid_intcoord(group_pixel)) {
        filtered->coords[filtered->length++] = group_pixel;
      }
    }
  }
}

void parse_to_float_coords(IntCoords_t *coords_in, Coords_t *coords_ret) {
  static Coords_t parsed_coords = {0, 0, NULL};
  parsed_coords.coords =
      possibly_resize_ptr(parsed_coords.coords, &parsed_coords.allocated,
                          coords_in->length, sizeof(Coord_t));
  for (int i = 0; i < coords_in->length; ++i) {
    parsed_coords.coords[i].x = (float)coords_in->coords[i].x;
    parsed_coords.coords[i].y = (float)coords_in->coords[i].y;
  }
  parsed_coords.length = coords_in->length;
  *coords_ret = parsed_coords;
}

void detect_ball(IntCoords_t *border, Indexes_t *group, Coord_t *prev_pos,
                 Coord_t *center_ransac, Coord_t *center_coope, float max_dx2,
                 float r, float min_dist, float max_dist, int max_iter,
                 int confidence_thrs, bool verbose) {
  // declare statics and make sure they have enough memory
  // static int *modeled_indexes = NULL;
  // static size_t modeled_indexes_al = 0;
  // static Coord_t *filtered_coords = NULL;
  // static size_t filtered_coords_al = 0;

  // filtered_coords = possibly_resize_ptr(filtered_coords, &filtered_coords_al,
  //                                       border_coords_l, sizeof(Coord_t));

  static IntCoords_t filtered = {0, 0, NULL};
  filtered.length = 0;

  // find the ball and remove old pixels
  // it is not necessary to look among all the pixels, if previous position was
  // known and the ball has not moved too much
  // printf("about to filter group coords\n");
  filter_group_coords(border, group, &filtered);

  // printf("filtered group coords\n");

  ransac(&filtered, r, min_dist, max_dist, max_iter, confidence_thrs, verbose,
         center_ransac);

  // printf("done ransac\n");

  if (valid_coord(*center_ransac)) {
    // find out whether it is necessary to look in previously created group or
    // in all border coords
    // printf("valid coord\n");
    bool only_group =
        group && in_group(center_ransac, prev_pos, max_dx2, max_dist);
    IntCoords_t *set_to_check;
    if (only_group) {
      set_to_check = &filtered;
    } else {
      set_to_check = border;
    }
    static Indexes_t modeled = {0, 0, NULL};
    modeled.length = 0;
    modeled.indexes = possibly_resize_ptr(modeled.indexes, &modeled.allocated,
                                          set_to_check->length, sizeof(int));

    find_modeled_pixels(center_ransac, prev_pos, max_dx2, min_dist, max_dist,
                        set_to_check, &modeled);
    // printf("found modeled pixels\n");
    *center_coope = lsq_on_modeled(set_to_check, &modeled);
    remove_pixels(border, group, &modeled, only_group);
    // printf("removed pixels\n");
  }
  // printf("leaving\n");
}

void detect_balls(uint8_t *rgb_to_balls_map, uint8_t *img, int width,
                  int height, int step, Coords_t *prev_pos, int *ball_colors,
                  float max_dx2, float r, float min_dist, float max_dist,
                  int max_iter, int confidence_thrs, bool verbose,
                  Coord_t *centers_ransac, Coord_t *centers_coope) {
  // nr of balls is defined by the length of prev_pos (nan if not known)
  // ball colors is an array that for each ball in previous positions has a
  // color index, that is the same as in the rgb_to_balls_map - probably defined
  // by the order of colors passed to init table
  for (int p = 0; p < 1000; ++p) {

    int number_of_colors = 0;
    for (int i = 0; i < prev_pos->length; ++i) {
      if (ball_colors[i] >= number_of_colors) {
        number_of_colors = ball_colors[i] + 1;
      }
    }

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
      // printf("incereasing size to %d\n", number_of_colors);
      ball_pixels =
          realloc(ball_pixels, number_of_colors * sizeof(IntCoords_t));
      for (int i = ball_pixels_l; i < number_of_colors; ++i) {
        ball_pixels[i].length = 0;
        ball_pixels[i].coords = malloc(NEW_GROUP_LEN * sizeof(IntCoord_t));
        ball_pixels[i].allocated = NEW_GROUP_LEN;
        ball_pixels_l = number_of_colors;
      }
    }

    for (int i = 0; i < ball_pixels_l; ++i) {
      ball_pixels[i].length = 0;
    }

    get_ball_pixels(img, width, height, number_of_colors, rgb_to_balls_map,
                    step, segmentation_mask, ball_pixels);

    // for (int i = 0; i < number_of_colors; ++i) {
    //   printf("found %d ball pixels for ball %d\n", ball_pixels[i].length, i);
    // }

    // for (int i = 0; i < number_of_colors; ++i) {
    //   for (int j = 0; j < ball_pixels[i].length; ++j) {
    //     printf("(%d,%d) ==> [%d,%d]\n", i, j, ball_pixels[i].coords[j].x,
    //            ball_pixels[i].coords[j].y);
    //   }
    // }

    static bool *skipped = NULL;
    static size_t skipped_al = 0;
    skipped = possibly_resize_ptr(skipped, &skipped_al, prev_pos->length,
                                  sizeof(bool));
    memset(skipped, true, prev_pos->length * sizeof(bool));

    // find border
    static IntCoords_t border = {0, 0, NULL};
    // first make sure there is enough memory allocated (result of malloc is not
    // constant at compile time)
    if (border.allocated == 0) {
      border.coords = malloc(NEW_GROUP_LEN * sizeof(IntCoord_t));
      border.allocated = NEW_GROUP_LEN;
    }

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

    for (int c = 0; c < number_of_colors; ++c) {
      // printf("color %d\n",c);
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

      for (int i = 0; i < prev_pos->length; ++i) {
        groups[i].length = 0;
      }
      get_border(segmentation_mask, width, height, ball_pixels + c,
                 &filtered_prev_pos, step, max_dx2, NULL, NULL, &groups,
                 &border);

      // printf("gotten border %d, length %d\n", c, border.length);
      // for (int i = 0; i < border.length; ++i) {
      //   printf("[%d,%d]\n", border.coords[i].x, border.coords[i].y);
      // }

      // printf("%s,
      // %s\n",(skipped[0]?"true":"false"),(skipped[1]?"true":"false"));
      // printf("bc %d: %d\n", c, border_coords_l);
      // again, make sure enough memory is allocated

      // parse_to_float_coords(border, border);

      // first, only balls with known previous positions are searched
      for (int i = 0; i < prev_pos->length; ++i) {
        if (!valid_coord(prev_pos->coords[i]) || !skipped[i] ||
            ball_colors[i] != c) {
          // if (ball_colors[i] != c) {
          // printf("ball %d is not color %d\n", i, c);
          // }
          continue;
        }
        detect_ball(&border, groups + i, prev_pos->coords + i,
                    centers_ransac + i, centers_coope + i, max_dx2, r, min_dist,
                    max_dist, max_iter, confidence_thrs, verbose);
        // printf("done detecting\n");
        if (valid_coord(centers_ransac[i])) {
          skipped[i] = false;
          // printf("1: detected ball color %d, index %d\n", c, i);
        }
      }
      // printf("Looking for more balls\n");
      // now (try to) find the rest
      for (int i = 0; i < prev_pos->length; ++i) {
        if (!skipped[i] || ball_colors[i] != c) {
          if (ball_colors[i] != c) {
            // printf("2: ball %d is not color %d\n", i, c);
          }
          continue;
        }
        detect_ball(&border, NULL, prev_pos->coords + i, centers_ransac + i,
                    centers_coope + i, max_dx2, r, min_dist, max_dist, max_iter,
                    confidence_thrs, verbose);
        // printf("done detecting2\n");
        if (valid_coord(centers_ransac[i])) {
          skipped[i] = false;
          // printf("2: detected ball color %d, index %d\n", c, i);
        }
      }
    }

    Coord_t not_found = {NAN, NAN};
    for (int i = 0; i < prev_pos->length; ++i) {
      if (skipped[i]) {
        *centers_ransac = not_found;
        *centers_coope = not_found;
      }
    }
    // exit(108);
  }
}