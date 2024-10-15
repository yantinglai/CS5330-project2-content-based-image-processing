/*
  Author: Yanting Lai
  Date: 2024-10-14
  CS 5330 Computer Vision
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "features.h"  // Contains sevenSquare, rgbHistfeature, histIntersection, and combinedSimilarity functions
#include "csv_util.h"  // Contains append_image_data_csv function
#include "readfiles.h" // Contains readImageFiles function

using namespace std;
using namespace cv;

// Function to extract the filename from a full path
string getFileName(const string& filePath) {
    size_t pos = filePath.find_last_of("/\\");
    return (pos == string::npos) ? filePath : filePath.substr(pos + 1);
}

string trim(const string& str) {
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

// Read the embedding values from a CSV file
map<string, vector<float>> readCSV(const string& csvFilePath) {
    map<string, vector<float>> imageEmbeddings;
    ifstream file(csvFilePath);
    string line, word;

    while (getline(file, line)) {
        stringstream ss(line);
        string fileName;
        vector<float> embedding(512);

        // Read the filename (first column)
        getline(ss, fileName, ',');
        fileName = trim(fileName);  // Trim any leading/trailing whitespace

        // Read the 512 embedding values
        for (int i = 0; i < 512; ++i) {
            if (!getline(ss, word, ',')) {
                cerr << "Error: Incorrect number of embedding values for " << fileName << endl;
                break;
            }
            embedding[i] = stof(word);  // Convert to float
        }

        if (embedding.size() == 512) {
            imageEmbeddings[fileName] = embedding;
        } else {
            cerr << "Error: Embedding vector for " << fileName << " is incomplete." << endl;
        }
    }

    return imageEmbeddings;
}

// Modified computeSimilarity to call combinedSimilarity
double computeSimilarity(const Mat& targetImage, const Mat& compareImage, const string& method,
                         const vector<float>& targetEmbedding, const map<string, vector<float>>& embeddings,
                         const string& compareImageFileName, int bins = 8) {
    string compareFileName = getFileName(compareImageFileName);  // Extract just the filename for comparison
    if (method == "SSD") {
        return sevenSquare(targetImage, compareImage);  // Traditional SSD matching
    } else if (method == "HIST") {
        vector<float> targetHist, compareHist;
        rgbHistfeature(targetImage, targetHist, bins);
        rgbHistfeature(compareImage, compareHist, bins);
        return histIntersection(targetHist, compareHist);  // Histogram intersection
    } else if (method == "MULTI_HIST") {
        return multiHistogramSimilarity(targetImage, compareImage, bins);  // Multi-histogram similarity
    } else if (method == "TEXTURE_COLOR") {
        return computeTextureAndColorSimilarity(targetImage, compareImage, bins, 0.5, 0.5);  // Texture and color similarity
    } else if (method == "COSINE") {
        auto it = embeddings.find(compareFileName);  // Use extracted filename for comparison
        if (it == embeddings.end()) {
            cerr << "Error: Could not find embedding for comparison image: " << compareFileName << endl;
            return -1;
        }
        const vector<float>& compareEmbedding = it->second;

        // Normalize both target and comparison embeddings
        std::vector<float> normalizedTargetEmbedding = targetEmbedding;
        std::vector<float> normalizedCompareEmbedding = compareEmbedding;
        normalizeVector(normalizedTargetEmbedding);
        normalizeVector(normalizedCompareEmbedding);

        return cosinedistance(normalizedTargetEmbedding, normalizedCompareEmbedding);
    } else if (method == "COMBINED") {
        // Use the combinedSimilarity function when the method is "COMBINED"
        auto it = embeddings.find(compareFileName);
        if (it == embeddings.end()) {
            cerr << "Error: Could not find embedding for comparison image: " << compareFileName << endl;
            return -1;
        }
        const vector<float>& compareEmbedding = it->second;
        return combinedSimilarity(targetImage, compareImage, targetEmbedding, compareEmbedding, bins, 0.5, 0.25, 0.25);
    } else {
        cerr << "Invalid method. Use 'SSD', 'HIST', 'MULTI_HIST', 'TEXTURE_COLOR', or 'COMBINED'." << endl;
        exit(-1);
    }
    return 0;
}

int main(int argc, char* argv[]) {
    // Ensure correct usage with at least 5 arguments
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <directory path> <target image> <N> <method (COSINE, SSD, HIST, MULTI_HIST, TEXTURE_COLOR, COMBINED)> <csv file>" << endl;
        return -1;
    }

    string directory = argv[1];             // Directory containing images
    string targetImageFileName = getFileName(argv[2]);   // Extract only the filename for the target image
    string method = argv[4];                // Similarity method (COSINE, SSD, etc.)
    int N = stoi(argv[3]);                  // Number of top matches to retrieve

    string targetImagePath = directory + "/" + targetImageFileName;  // Full path of the target image

    // Load the target image from the specified path
    Mat targetImage = imread(targetImagePath, IMREAD_COLOR);
    if (targetImage.empty()) {
        cerr << "Error: Cannot load target image from " << targetImagePath << endl;
        return -1;
    }

    // Load embeddings for the COSINE/COMBINED method (from CSV file)
    map<string, vector<float>> embeddings;
    vector<float> targetEmbedding;

    if (method == "COSINE" || method == "COMBINED") {
        string csvFilePath = argv[5];  // CSV file path for embedding-based methods
        embeddings = readCSV(csvFilePath);  // Load embeddings from the CSV file

        // Extract the embedding for the target image
        auto it = embeddings.find(targetImageFileName);
        if (it == embeddings.end()) {
            cerr << "Error: Could not find embedding for the target image: " << targetImageFileName << endl;
            return -1;
        }
        targetEmbedding = it->second;  // Get the embedding vector for the target image
    }

    // Read all image file paths from the specified directory
    vector<string> filepaths;
    readImageFiles(directory, filepaths);  // Function to list all image files in the directory

    multimap<double, string> distances;  // To store similarity values and corresponding image paths

    // Compare each image in the directory with the target image and calculate similarity
    for (const auto& filepath : filepaths) {
        Mat compareImg = imread(filepath, IMREAD_COLOR);  // Load the comparison image
        if (!compareImg.empty()) {
            // Compute similarity based on the chosen method
            double similarity = computeSimilarity(targetImage, compareImg, method, targetEmbedding, embeddings, filepath);
            distances.emplace(similarity, filepath);  // Store the result in the multimap
        }
    }

    // Output the top N matching images
    cout << "Top " << N << " matching images based on " << method << ":\n";
    int count = 0;

    // For SSD method: lower values are more similar, so we print from the beginning (ascending order)
    if (method == "SSD") {
        for (const auto& [similarity, filename] : distances) {
            if (count++ >= N) break;
            cout << "Image: " << filename << ", SSD Value: " << similarity << endl;
        }
    } 
    // For other methods like COSINE/COMBINED: higher values are more similar, so we print from the end (descending order)
    else {
        for (auto it = distances.rbegin(); it != distances.rend() && count < N; ++it) {
            cout << "Image: " << it->second << ", Similarity Value: " << it->first << endl;
            count++;
        }
    }

    // Output the bottom N least similar images (for methods like COSINE/COMBINED)
    cout << "Bottom " << N << " least similar images based on " << method << ":\n";
    count = 0;

    // For SSD method: higher values are less similar, so we print from the end (descending order)
    if (method == "SSD") {
        for (auto it = distances.rbegin(); it != distances.rend() && count < N; ++it) {
            cout << "Image: " << it->second << ", SSD Value: " << it->first << endl;
            count++;
        }
    }
    // For other methods like COSINE/COMBINED: lower values are less similar, so we print from the beginning (ascending order)
    else {
        for (const auto& [similarity, filename] : distances) {
            if (count++ >= N) break;
            cout << "Image: " << filename << ", Similarity Value: " << similarity << endl;
        }
    }

    return 0;
}
