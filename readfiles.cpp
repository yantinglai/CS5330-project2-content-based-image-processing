/*
  Author: Yanting Lai
  Date: 2024-10-14
  CS 5330 Computer Vision
*/

#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include "readfiles.h"

#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include "readfiles.h"

using namespace std;

// Reads all file names in the specified directory.
void readImageFiles(const string& directory, vector<string>& filepaths) {
    DIR *dirp;
    struct dirent *dp;

    // Opens the specified directory.
    dirp = opendir(directory.c_str());
    if (!dirp) {
        cerr << "Cannot open directory: " << directory << endl;
        return;
    }

    // Iterates through the files in the directory.
    while ((dp = readdir(dirp)) != NULL) {
        string filename = dp->d_name;
        // Only processes .jpg files.
        if (filename.find(".jpg") != string::npos) {
            // Adds the full path of the .jpg file to the list.
            filepaths.push_back(directory + "/" + filename);
        }
    }

    // Closes the directory.
    closedir(dirp);
}
