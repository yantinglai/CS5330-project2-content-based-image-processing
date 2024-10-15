/*
  Author: Yanting Lai
  Date: 2024-10-14
  CS 5330 Computer Vision
*/

#include "features.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

// Calculate the SSD (Sum of Squared Differences) for the central 7x7 region
double calculateSSD(const Mat &target, const Mat &compare) {
    int startRow = (target.rows - 7) / 2;
    int startCol = (target.cols - 7) / 2;

    double ssd = 0.0;

    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
            for (int channel = 0; channel < 3; ++channel) {
                uchar targetPixel = target.at<cv::Vec3b>(startRow + r, startCol + c)[channel];
                uchar comparePixel = compare.at<cv::Vec3b>(startRow + r, startCol + c)[channel];

                // Calculate the pixel difference and accumulate SSD
                double diff = static_cast<double>(targetPixel) - static_cast<double>(comparePixel);
                ssd += diff * diff;
            }
        }
    }
    return ssd;
}

// Extract RGB histogram features
int extractRGBHist(const Mat &image, std::vector<float> &featureVector, int bins) {
    // Clear and resize the vector to hold RGB histogram bins
    featureVector.clear();
    featureVector.resize(bins * bins * bins, 0);

    float totalPixels = image.rows * image.cols; // Total number of pixels

    for (int r = 0; r < image.rows; ++r) {
        for (int c = 0; c < image.cols; ++c) {
            Vec3b pixel = image.at<Vec3b>(r, c); // Get the pixel

            float blue = pixel[0];
            float green = pixel[1];
            float red = pixel[2];

            // Normalize the RGB values
            float divisor = red + green + blue;
            divisor = divisor > 0.0 ? divisor : 1.0;

            float rValue = red / divisor;
            float gValue = green / divisor;
            float bValue = blue / divisor;

            // Calculate bin indices for each channel
            int rIndex = static_cast<int>(rValue * (bins - 1) + 0.5);
            int gIndex = static_cast<int>(gValue * (bins - 1) + 0.5);
            int bIndex = static_cast<int>(bValue * (bins - 1) + 0.5);

            featureVector[rIndex * bins * bins + gIndex * bins + bIndex] += 1.0 / totalPixels; // Normalize by total pixels
        }
    }
    return 0;
}

// Calculate histogram intersection similarity
double histIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2) {
    double intersection = 0.0;
    for (size_t i = 0; i < hist1.size(); ++i) {
        intersection += std::min(hist1[i], hist2[i]); // Sum minimum bin values
    }
    return intersection;
}

// Calculate histogram correlation
double histCorrelation(const std::vector<float> &hist1, const std::vector<float> &hist2) {
    Mat histMat1 = Mat(hist1).clone().reshape(1, 1);
    Mat histMat2 = Mat(hist2).clone().reshape(1, 1);
    
    return compareHist(histMat1, histMat2, HISTCMP_CORREL);
}

// Multi-histogram similarity computation using different regions
double multiRegionHistSimilarity(const Mat& targetImage, const Mat& compareImage, int bins = 16) {
    // Weights for different regions
    double wholeImageWeight = 0.4;
    double centerImageWeight = 0.4;
    double topBottomWeight = 0.2;

    vector<float> targetWholeHist, targetCenterHist, targetTopHist, targetBottomHist;
    vector<float> compareWholeHist, compareCenterHist, compareTopHist, compareBottomHist;

    // Whole image histogram
    extractRGBHist(targetImage, targetWholeHist, bins);
    extractRGBHist(compareImage, compareWholeHist, bins);

    // Center region histogram
    int startRow = targetImage.rows / 3;
    int startCol = targetImage.cols / 3;
    Rect centerRegion(startCol, startRow, targetImage.cols / 3, targetImage.rows / 3);
    
    Mat targetCenter = targetImage(centerRegion);
    Mat compareCenter = compareImage(centerRegion);
    
    extractRGBHist(targetCenter, targetCenterHist, bins);
    extractRGBHist(compareCenter, compareCenterHist, bins);

    // Top and bottom region histograms
    Rect topRegion(0, 0, targetImage.cols, targetImage.rows / 2);
    Rect bottomRegion(0, targetImage.rows / 2, targetImage.cols, targetImage.rows / 2);

    Mat targetTop = targetImage(topRegion);
    Mat compareTop = compareImage(topRegion);
    extractRGBHist(targetTop, targetTopHist, bins);
    extractRGBHist(compareTop, compareTopHist, bins);

    Mat targetBottom = targetImage(bottomRegion);
    Mat compareBottom = compareImage(bottomRegion);
    extractRGBHist(targetBottom, targetBottomHist, bins);
    extractRGBHist(compareBottom, compareBottomHist, bins);

    // Calculate correlations and combine using weights
    double wholeCorrelation = histCorrelation(targetWholeHist, compareWholeHist);
    double centerCorrelation = histCorrelation(targetCenterHist, compareCenterHist);
    double topCorrelation = histCorrelation(targetTopHist, compareTopHist);
    double bottomCorrelation = histCorrelation(targetBottomHist, compareBottomHist);

    return (wholeImageWeight * wholeCorrelation) + 
           (centerImageWeight * centerCorrelation) + 
           (topBottomWeight * (topCorrelation + bottomCorrelation) / 2);
}

// Extract texture histogram features
int textureHist(const Mat &image, std::vector<float> &featureVector) {
    int magnitudeBins = 16;
    int orientationBins = 18;

    featureVector.clear();
    featureVector.resize(orientationBins * magnitudeBins, 0);

    Mat grayImage, x, y;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Sobel(grayImage, x, CV_32F, 1, 0);
    cv::Sobel(grayImage, y, CV_32F, 1, 0);

    Mat mag, ori;
    cv::cartToPolar(x, y, mag, ori, true);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);

    for (int r = 0; r < image.rows; ++r) {
        for (int c = 0; c < image.cols; ++c) {
            float magValue = mag.at<float>(r, c);
            float oriValue = ori.at<float>(r, c);

            int magIndex = static_cast<int>((magValue / 255) * (magnitudeBins - 1) + 0.5);
            int oriIndex = static_cast<int>((oriValue / 360) * (orientationBins - 1) + 0.5);

            featureVector[magIndex * magnitudeBins + oriIndex] += 1.0; // Increment histogram bin
        }
    }
    return 0;
}

// Compute combined texture and color similarity
double textureAndColorSimilarity(const Mat& targetImage, const Mat& compareImage, int bins = 8, double textureWeight = 0.5, double colorWeight = 0.5) {
    std::vector<float> targetTexture, targetColor;
    std::vector<float> compareTexture, compareColor;

    extractRGBHist(targetImage, targetColor, bins);
    extractRGBHist(compareImage, compareColor, bins);

    textureHist(targetImage, targetTexture);
    textureHist(compareImage, compareTexture);

    double textureSimilarity = histIntersection(targetTexture, compareTexture);
    double colorSimilarity = histIntersection(targetColor, compareColor);

    return (textureWeight * textureSimilarity) + (colorWeight * colorSimilarity);
}

// Normalize feature vector
void normalize(std::vector<float> &vec) {
    double norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
    if (norm > 0) {
        for (auto &v : vec) {
            v /= norm;
        }
    }
}

// Compute cosine similarity between two feature vectors
double cosineSimilarity(const std::vector<float> &vec1, const std::vector<float> &vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Feature vectors must be of the same size.");
    }

    double dotProduct = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
    double norm1 = std::inner_product(vec1.begin(), vec1.end(), vec1.begin(), 0.0);
    double norm2 = std::inner_product(vec2.begin(), vec2.end(), vec2.begin(), 0.0);

    if (norm1 == 0 || norm2 == 0) {
        return 1.0; // Max cosine distance when one vector is zero
    }

    return 1 - dotProduct / (std::sqrt(norm1) * std::sqrt(norm2));
}

// Combined similarity: DNN embeddings, color, and texture
double combinedSimilarity(const Mat& targetImage, const Mat& compareImage,
                          const vector<float>& targetEmbedding, const vector<float>& compareEmbedding,
                          int bins = 8, double dnnWeight = 0.25, double colorWeight = 0.25, double textureWeight = 0.5) {

    std::vector<float> normTargetEmbedding = targetEmbedding;
    std::vector<float> normCompareEmbedding = compareEmbedding;
    normalize(normTargetEmbedding);
    normalize(normCompareEmbedding);
    double dnnSimilarity = cosineSimilarity(normTargetEmbedding, normCompareEmbedding);

    std::vector<float> targetColorHist, compareColorHist;
    extractRGBHist(targetImage, targetColorHist, bins);
    extractRGBHist(compareImage, compareColorHist, bins);
    double colorSimilarity = histIntersection(targetColorHist, compareColorHist);

    std::vector<float> targetTextureHist, compareTextureHist;
    textureHist(targetImage, targetTextureHist);
    textureHist(compareImage, compareTextureHist);
    double textureSimilarity = histIntersection(targetTextureHist, compareTextureHist);

    return dnnWeight * dnnSimilarity + colorWeight * colorSimilarity + textureWeight * textureSimilarity;
}
