/*
  Author: Yanting Lai
  Date: 2024-10-14
  CS 5330 Computer Vision
*/

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>

/**
 * @brief 
 * Computes the sum of squared differences (SSD) for the 7x7 center region 
 * of two images.
 * 
 * @param target The target image.
 * @param compare The image to compare with the target.
 * 
 * @return The SSD value between the two 7x7 regions.
 */
double sevenSquare(const cv::Mat& target, const cv::Mat& compare);

/**
 * @brief Checks if the file has a valid image extension (e.g., .jpg, .png).
 * 
 * @param filename The file name to check.
 * @return True if the file is an image, false otherwise.
 */
bool isImageFile(const std::string& filename);

/**
 * @brief Extracts the RGB histogram feature vector from an image.
 * 
 * @param image The image.
 * @param featureVector Output vector to store the histogram values.
 * @param bins The number of bins to use for the histogram.
 * @return 0 for success.
 */
int rgbHistfeature(const cv::Mat &image, std::vector<float> &featureVector, int bins);

/**
 * @brief Computes the histogram intersection between two histograms.
 * 
 * @param hist1 First histogram.
 * @param hist2 Second histogram.
 * @return The intersection value (higher is more similar).
 */
double histIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2);

/**
 * @brief Computes the similarity between two images based on multi-histogram matching
 *        for different parts of the image.
 * 
 * @param targetImage The target image.
 * @param compareImage The comparison image.
 * @param bins Number of bins for the histograms.
 * @return The similarity value between the two images.
 */
double multiHistogramSimilarity(const cv::Mat& targetImage, const cv::Mat& compareImage, int bins);

/**
 * @brief Computes a combined similarity using texture and color histograms.
 * 
 * @param targetImage The target image.
 * @param compareImage The comparison image.
 * @param bins Number of bins for the histograms.
 * @param textureWeight Weight for the texture similarity.
 * @param colorWeight Weight for the color similarity.
 * @return The combined similarity score.
 */
double computeTextureAndColorSimilarity(const cv::Mat& targetImage, const cv::Mat& compareImage, int bins, double textureWeight, double colorWeight);

/**
 * @brief Extracts the texture histogram from an image using orientation and magnitude.
 * 
 * @param image The image.
 * @param featureVector Output vector to store the texture histogram.
 * @return 0 for success.
 */
int textureHistfeature(const cv::Mat &image, std::vector<float> &featureVector);

/**
 * @brief Normalizes a feature vector to have unit length.
 * 
 * @param vec The vector to normalize.
 */
void normalizeVector(std::vector<float> &vec);

/**
 * @brief Computes the cosine distance between two feature vectors.
 * 
 * @param feaVec1 First feature vector.
 * @param feaVec2 Second feature vector.
 * @return The cosine distance (lower is more similar).
 */
double cosinedistance(const std::vector<float> &feaVec1, const std::vector<float> &feaVec2);

/**
 * @brief Computes a combined similarity score using DNN embeddings, color, and texture.
 * 
 * @param targetImage The target image.
 * @param compareImage The comparison image.
 * @param targetEmbedding The target image's DNN embedding.
 * @param compareEmbedding The comparison image's DNN embedding.
 * @param bins Number of bins for color and texture histograms.
 * @param dnnWeight Weight for DNN similarity.
 * @param colorWeight Weight for color histogram similarity.
 * @param textureWeight Weight for texture histogram similarity.
 * @return The combined similarity score.
 */
double combinedSimilarity(const cv::Mat& targetImage, const cv::Mat& compareImage,
                          const std::vector<float>& targetEmbedding, const std::vector<float>& compareEmbedding,
                          int bins, double dnnWeight, double colorWeight, double textureWeight);

#endif  // FEATURES_H
