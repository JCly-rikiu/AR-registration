#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Please provide the image file!" << std::endl;
    return EXIT_FAILURE;
  }
  std::string image_filename = argv[1];
  cv::Mat image;
  image = cv::imread(image_filename, cv::IMREAD_COLOR);
  if (!image.data) {
    std::cerr << "Could not open or find the image!" << std::endl;
    return EXIT_FAILURE;
  }

  auto rows = image.rows;
  auto cols = image.cols;

  cv::Vec3b A(0, 0, 255);
  cv::Vec3b B(0, 100, 255);
  cv::Vec3b C(255, 0, 0);

  std::vector<int> abc_pos(6);

  for (int i = 0; i != rows; i++)
    for (int j = 0; j != cols; j++) {
      auto pixel = image.at<cv::Vec3b>(i, j);
      if (pixel == A) {
        std::cout << "A i: " << i << " j: " << j << std::endl;
        abc_pos[0] = i;
        abc_pos[1] = j;
      }
      if (pixel == B) {
        std::cout << "B i: " << i << " j: " << j << std::endl;
        abc_pos[2] = i;
        abc_pos[3] = j;
      }
      if (pixel == C) {
        std::cout << "C i: " << i << " j: " << j << std::endl;
        abc_pos[4] = i;
        abc_pos[5] = j;
      }
    }

  auto iAB = abc_pos[2] - abc_pos[0];
  auto jAB = abc_pos[3] - abc_pos[1];
  auto iAC = abc_pos[4] - abc_pos[0];
  auto jAC = abc_pos[5] - abc_pos[1];
  auto a = iAB + iAC, b = jAB + jAC;
  auto angle = std::acos(b / std::sqrt(a * a + b * b)) * 180 / 3.14;
  if (angle > 135) {
    for (int p = 0; p < 6; p += 2) {
      abc_pos[p] = rows - abc_pos[p];
      abc_pos[p + 1] = cols - abc_pos[p + 1];
    }
  } else if (angle > 45) {
    for (int p = 0; p < 6; p += 2) {
      auto i = abc_pos[p], j = abc_pos[p + 1];
      if (a > 0) {
        abc_pos[p] = cols - j;
        abc_pos[p + 1] = i;
      } else {
        abc_pos[p] = j;
        abc_pos[p + 1] = rows - i;
      }
    }

    auto temp = rows;
    rows = cols;
    cols = temp;
  }

  std::stringstream ss;
  ss << "python3 registration.py " << rows << " " << cols;
  for (auto p : abc_pos) ss << " " << p;

  [[maybe_unused]] auto ignored = std::system(ss.str().c_str());

  return EXIT_SUCCESS;
}
