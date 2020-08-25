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

// compile with cc -fPIC -shared -o ransac_detector.so ransac_detector.c -L.
// -lcoope_fit -Wl,-rpath .

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

// define constants
#define R 0
#define G 1
#define B 2

#define NONE 255

#define NEW_GROUP_LEN 32

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
  if (*ptr_size < min_size) {
    *ptr_size = min_size;
    ptr = realloc(ptr, min_size * type_size);
  }
  return ptr;
}

void *possibly_double_ptr_size(void *ptr, size_t *ptr_size, size_t min_size,
                               size_t type_size) {
  // printf("%d vs %d\n",*ptr_size,min_size);
  if (*ptr_size <= min_size) {
    // printf("incerasing\n");
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
  // printf("done segmenting\n");
}

size_t get_ball_pixels(uint8_t *img, int width, int height,
                       uint8_t *rgb_to_balls_map, int step, uint8_t *mask,
                       int **ball_pixels_ret) {
  /* returns the length of mask_coords*/
  static size_t ball_pixels_al = 0;
  static int *ball_pixels = NULL;

  int ball_pixel_l = 0;
  for (int y = 0; y < height; y = y + step) {
    for (int x = 0; x < width; x = x + step) {
      // set the pixel in segmentation mask to its corresponding ball (or NONE)
      Color_t *c = (Color_t *)pixel(img, x, y);
      // printf("R: %d, G: %d, B: %d\n",c->r, c->g, c->b);
      int color_index = index_from_rgb(*c);
      int ball_color = rgb_to_balls_map[color_index];
      mask[image_index(x, y)] = ball_color;
      if (ball_color != NONE) {
        // make sure enough space is available
        if (ball_pixel_l + 2 >= ball_pixels_al) {
          ball_pixels_al = 2 * (ball_pixels_al + 1);
          ball_pixels = realloc(ball_pixels, 2 * ball_pixels_al * sizeof(int));
        }
        // save the coords
        ball_pixels[2 * ball_pixel_l] = x;
        ball_pixels[2 * ball_pixel_l + 1] = y;
        ball_pixel_l++;
      }
    }
  }
  *ball_pixels_ret = ball_pixels;
  return ball_pixel_l;
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

bool is_border_pixel(uint8_t *neighbours) {
  // neighbours is expected implicitly to be of len 9 (i.e. all directions
  // including the pixel)
  bool zero_exists = false;
  bool one_exists = false;
  for (int j = 0; j < 9; ++j) {
    // if(j==4){continue;}//the pixel itself
    one_exists = one_exists || neighbours[j];
    zero_exists = zero_exists || !neighbours[j];
  }
  return one_exists && zero_exists;
}

size_t get_border_coords(uint8_t *segmentation_mask, int width, int height,
                         int *ball_pixel_coords, size_t ball_pixel_coords_l,
                         Coord_t *prev_pos, size_t prev_pos_l, int step,
                         float max_dx2, uint8_t *border_mask,
                         uint8_t *group_mask, int ***group_index_ret,
                         size_t **group_index_ls_ret, int **border_coords_ret) {

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
 group_index_l: 4
 */
  static int **group_index = NULL;
  static size_t *group_index_als = NULL; // allocated size
  static size_t *group_index_ls = NULL;  // filled size
  static int group_index_l = 0;

  static int *border_coords = NULL;
  static size_t border_coords_al = 0;

  // first make sure there is enough memory allocated
  // printf("before\n");
  if (!border_coords) {
    border_coords = malloc(64 * sizeof(int));
    border_coords_al = 64;
  }
  // printf("before2\n");
  if (group_index_l < prev_pos_l) {
    // create index list for each new ball of length NEW_GROUP_LEN
    group_index = realloc(group_index, prev_pos_l * sizeof(int *));
    for (int i = group_index_l; i < prev_pos_l; ++i) {
      group_index[i] = malloc(NEW_GROUP_LEN * sizeof(int));
    }
    group_index_als = realloc(group_index_als, prev_pos_l * sizeof(size_t));
    for (int i = group_index_l; i < prev_pos_l; ++i) {
      group_index_als[i] = NEW_GROUP_LEN;
    }
    group_index_ls = realloc(group_index_ls, prev_pos_l * sizeof(size_t));
    group_index_l = prev_pos_l;
  }
  memset(group_index_ls, 0, prev_pos_l * sizeof(size_t));
  // printf("after2\n");

  size_t border_coords_l = 0;
  // printf("before3\n");
  for (int i = 0; i < ball_pixel_coords_l; ++i) {
    int x = ball_pixel_coords[2 * i];
    int y = ball_pixel_coords[2 * i + 1];
    // printf("before4\n");
    uint8_t *neighbours =
        get_neighbour_values(segmentation_mask, width, height, x, y, step);
    // printf("after4\n");
    if (is_border_pixel(neighbours)) { // border pixel
      // printf("is border_pixel\n");
      // increase array size if necessary and copy x,y to boorder_coords
      border_coords =
          possibly_double_ptr_size(border_coords, &border_coords_al,
                                   2 * border_coords_l + 2, sizeof(int));
      border_coords[2 * border_coords_l] = x;
      border_coords[2 * border_coords_l + 1] = y;
      // printf("hh\n");
      if (border_mask) {
        border_mask[image_index(x, y)] = 1;
      }

      // add to possible candidates for each of the balls (based on previous
      // position) to speed up RANSAC
      // printf("before5\n");
      for (int ball_index = 0; ball_index < prev_pos_l; ++ball_index) {
        // skip if previous pos not provided
        if (!valid_coord(prev_pos[ball_index])) {
          continue;
        }
        size_t *this_group_index = group_index_ls + ball_index;
        float distance =
            distance_f2(x, y, prev_pos[ball_index].x, prev_pos[ball_index].y);
        // let's suppose that the ball moved at max by sqrt(max_dx2) (if it did,
        // it will have to be found without speedup)
        if (distance < max_dx2) {
          // make sure there is available memory
          group_index[ball_index] = possibly_double_ptr_size(
              group_index[ball_index], group_index_als + ball_index,
              *this_group_index, sizeof(int));

          group_index[ball_index][*this_group_index] = border_coords_l;
          if (group_mask) {
            // if more balls are possible at the same place, this will only save
            // the last one but oh well...
            group_mask[image_index(x, y)] = ball_index;
          }
          *this_group_index += 1;
        }
      }
      // printf("after5\n");
      border_coords_l++;
    }
  }

  // printf("after3\n");
  *group_index_ret = group_index;
  *group_index_ls_ret = group_index_ls;
  *border_coords_ret = border_coords;
  return border_coords_l;
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

void find_center_candidates(Coord_t *border_coords, int border_coords_l, int r,
                            Coord_t *centers_ret) {
  centers_ret[0].x = NAN;
  centers_ret[0].y = NAN;
  centers_ret[1].x = NAN;
  centers_ret[1].y = NAN;
  int max_iterations = 500;
  int failed_to_find_intersection = 0;
  while (!valid_coord(centers_ret[0]) &&
         failed_to_find_intersection < max_iterations) {
    int index1 = rand_lim(border_coords_l - 1);
    int index2 = index1;
    // no point in picking the same index twice
    while (index2 == index1) {
      index2 = rand_lim(border_coords_l - 1);
    }
    circle_intersections(border_coords[index1].x, border_coords[index1].y, r,
                         border_coords[index2].x, border_coords[index2].y, r,
                         centers_ret);
    failed_to_find_intersection++;
  }
}

void ransac(Coord_t *border_coords, int border_coords_l, float r,
            float min_dist, float max_dist, int max_iter, int confidence_thrs,
            bool verbose, Coord_t *best_model_ret) {
  best_model_ret->x = NAN;
  best_model_ret->y = NAN;
  if (border_coords_l < confidence_thrs / 4) {
    if (verbose) {
      printf("Out of balls!\n");
    }
    return;
  }

  int best_inliers = 0;
  static Coord_t centers[2];
  for (int iteration = 0; iteration < max_iter; ++iteration) {
    find_center_candidates(border_coords, border_coords_l, r, centers);
    bool same_centers = distance_f2(centers[0].x, centers[0].y, centers[1].x,
                                    centers[1].y) < 0.1;
    for (int i = 0; i < (same_centers ? 1 : 2); ++i) {
      int inliers = 0;
      for (int j = 0; j < border_coords_l; ++j) {
        float distance = distance_f2(border_coords[j].x, border_coords[j].y,
                                     centers[i].x, centers[i].y);
        if (distance < max_dist && distance > min_dist) {
          inliers++;
        }
      }
      if (inliers > best_inliers) {
        best_inliers = inliers;
        *best_model_ret = centers[i];
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
size_t find_modeled_pixels(Coord_t *model, Coord_t *previous_center,
                           float max_dx2, float min_dist, float max_dist,
                           Coord_t *set_to_check, size_t set_length,
                           int *modeled_indexes) {
  // check if it is possible to only search in group coords
  memset(modeled_indexes, 0, set_length * sizeof(int));
  size_t modeled_nr = 0;
  for (int i = 0; i < set_length; ++i) {
    float distance2 = distance_coords2(set_to_check + i, model);
    if (distance2 > pow(min_dist, 2) && distance2 < pow(max_dist, 2)) {
      modeled_indexes[modeled_nr++] = i;
    }
  }
  return modeled_nr;
}

void remove_pixels(Coord_t *border_coords, size_t border_coords_l,
                   int *group_indexes, size_t group_indexes_l,
                   int *modeled_indexes, size_t modeled_indexes_l,
                   bool only_group) {
  for (int i = 0; i < modeled_indexes_l; ++i) {
    Coord_t *modeled_pixel_address;
    if (only_group) {
      modeled_pixel_address = border_coords + group_indexes[modeled_indexes[i]];
    } else {
      modeled_pixel_address = border_coords + modeled_indexes[i];
    }
    modeled_pixel_address->x = NAN;
    modeled_pixel_address->y = NAN;
  }
}

Coord_t lsq_on_modeled(Coord_t *coords, size_t coords_l, int *modeled_indexes,
                       size_t modeled_indexes_l) {
  static Coord_t *modeled = NULL;
  static size_t modeled_al = 0;
  modeled = possibly_resize_ptr(modeled, &modeled_al, modeled_indexes_l,
                                sizeof(Coord_t));

  for (int i = 0; i < modeled_indexes_l; ++i) {
    // printf("[%f, %f]-", coords[i].x, coords[i].y);
    modeled[i] = coords[modeled_indexes[i]];
  }
  return coope_fit(modeled, (int)modeled_indexes_l);
}

size_t filter_group_coords(Coord_t *border_coords, size_t border_coords_l,
                           int *group_indexes, size_t group_indexes_l,
                           Coord_t *filtered_coords_ret) {
  size_t j = 0;
  if (group_indexes) {
    for (int i = 0; i < group_indexes_l; ++i) {
      Coord_t group_pixel = border_coords[group_indexes[i]];
      if (valid_coord(group_pixel)) {
        filtered_coords_ret[j++] = group_pixel;
      }
    }
  } else {
    for (int i = 0; i < border_coords_l; ++i) {
      Coord_t group_pixel = border_coords[i];
      // printf("[%f, %f]\n", border_coords[i].x, border_coords[i].y);
      if (valid_coord(group_pixel)) {
        filtered_coords_ret[j++] = group_pixel;
      }
    }
  }
  return j;
}

void parse_to_coords(int *coords, size_t coords_l, Coord_t *coords_ret) {
  for (int i = 0; i < coords_l; ++i) {
    coords_ret[i].x = (float)coords[2 * i];
    coords_ret[i].y = (float)coords[2 * i + 1];
  }
}

void detect_ball(Coord_t *border_coords, size_t border_coords_l,
                 int *group_indexes, size_t group_indexes_l, Coord_t *prev_pos,
                 Coord_t *center_ransac, Coord_t*center_coope, float max_dx2, float r, float min_dist,
                 float max_dist, int max_iter, int confidence_thrs,
                 bool verbose) {
  // declare statics and make sure they have enough memory
  static int *modeled_indexes = NULL;
  static size_t modeled_indexes_al = 0;
  static Coord_t *filtered_coords = NULL;
  static size_t filtered_coords_al = 0;

  filtered_coords = possibly_resize_ptr(filtered_coords, &filtered_coords_al,
                                        border_coords_l, sizeof(Coord_t));

  // find the ball and remove old pixels
  // it is not necessary to look among all the pixels, if previous position was
  // known and the ball has not moved too much
  size_t filtered_coords_l =
      filter_group_coords(border_coords, border_coords_l, group_indexes,
                          group_indexes_l, filtered_coords);
  // for (int i = 0; i < filtered_coords_l; ++i) {
  //   printf("[%f, %f]\n",filtered_coords[i].x,filtered_coords[i].y);
  // }

  ransac(filtered_coords, filtered_coords_l, r, min_dist, max_dist, max_iter,
         confidence_thrs, verbose, center_ransac);

  if (valid_coord(*center_ransac)) {
  // find out whether it is necessary to look in previously created group or in
  // all border coords
  bool only_group = in_group(center_ransac, prev_pos, max_dx2, max_dist);
  Coord_t *modeled_set;
  size_t modeled_set_l;
  if (only_group) {
    modeled_set = filtered_coords;
    modeled_set_l = filtered_coords_l;
  } else {
    modeled_set = border_coords;
    modeled_set_l = border_coords_l;
  }
  modeled_indexes = possibly_resize_ptr(modeled_indexes, &modeled_indexes_al,
                                        modeled_set_l, sizeof(int));
  if (modeled_indexes_al < modeled_set_l) {
    modeled_indexes_al = modeled_set_l;
    modeled_indexes =
        realloc(modeled_indexes, modeled_indexes_al * sizeof(int));
  }

    size_t modeled_indexes_l =
        find_modeled_pixels(center_ransac, prev_pos, max_dx2, min_dist, max_dist,
                            modeled_set, modeled_set_l, modeled_indexes);
    *center_coope = lsq_on_modeled(modeled_set, modeled_indexes_l,
                                 modeled_indexes, modeled_indexes_l);
    remove_pixels(border_coords, border_coords_l, group_indexes,
                  group_indexes_l, modeled_indexes, modeled_indexes_l,
                  only_group);
  }
}

void detect_balls(uint8_t *rgb_to_balls_map, uint8_t *img, int width,
                  int height, int step, Coord_t *prev_pos, size_t prev_pos_l,
                  float max_dx2, float r, float min_dist, float max_dist,
                  int max_iter, int confidence_thrs, bool verbose,
                  Coord_t *centers_ransac, Coord_t*centers_coope) {
  for (int l = 0; l < 1; ++l) { // for benchmarking
    static uint8_t *segmentation_mask = NULL;
    static size_t segmentation_mask_al = 0;
    static Coord_t *border_coords = NULL;
    static size_t border_coords_al = 0;

    // alocate memory
    int image_size = width * height;
    segmentation_mask = possibly_resize_ptr(
        segmentation_mask, &segmentation_mask_al, image_size, sizeof(uint8_t));
    // if (segmentation_mask_al < image_size) {
    //   segmentation_mask =
    //       realloc(segmentation_mask, image_size * sizeof(uint8_t));
    //   segmentation_mask_al = image_size;
    // }
    int *ball_pixels;
    uint8_t *group_mask;
    int **group_index;
    size_t *group_index_ls;
    int *border_pixels;

    // find border
    size_t ball_pixels_l =
        get_ball_pixels(img, width, height, rgb_to_balls_map, step,
                        segmentation_mask, &ball_pixels);
    // printf("bp%d\n",ball_pixels_l);
    size_t border_coords_l = get_border_coords(
        segmentation_mask, width, height, ball_pixels, ball_pixels_l, prev_pos,
        prev_pos_l, step, max_dx2, NULL, NULL, &group_index, &group_index_ls,
        &border_pixels);
    //  printf("bc%d\n",border_coords_l);
    // again, make sure enough memory is allocated
    border_coords = possibly_resize_ptr(border_coords, &border_coords_al,
                                        border_coords_l, sizeof(Coord_t));
    if (border_coords_al < border_coords_l) {
      border_coords_al = border_coords_l;
      border_coords =
          realloc(border_coords, border_coords_al * sizeof(Coord_t));
    }

    parse_to_coords(border_pixels, border_coords_l, border_coords);

    static bool *skipped = NULL;
    static size_t skipped_al = 0;
    skipped =
        possibly_resize_ptr(skipped, &skipped_al, prev_pos_l, sizeof(bool));
    memset(skipped, 0, prev_pos_l * sizeof(bool));

    // first, only balls with known previous positions are searched
    for (int i = 0; i < prev_pos_l; ++i) {
      if (!valid_coord(prev_pos[i])) {
        skipped[i] = true;
        continue;
      }
      detect_ball(border_coords, border_coords_l, group_index[i],
                  group_index_ls[i], prev_pos + i, centers_ransac + i,centers_coope+i, max_dx2, r,
                  min_dist, max_dist, max_iter, confidence_thrs, verbose);
      if (!valid_coord(centers_ransac[i])) {
        skipped[i] = true;
      }
    }
    // now (try to) find the rest
    for (int i = 0; i < prev_pos_l; ++i) {
      if (!skipped[i]) {
        continue;
      }
      detect_ball(border_coords, border_coords_l, NULL, 0, prev_pos + i,
                  centers_ransac + i,centers_coope+i, max_dx2, r, min_dist, max_dist, max_iter,
                  confidence_thrs, verbose);
    }
  }
  // exit(108);
}