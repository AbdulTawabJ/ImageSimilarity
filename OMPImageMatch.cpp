#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
namespace fs = std::filesystem;

// Function to compute the LBP histogram of an image
vector<double> computeLBP(const cv::Mat& img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat lbp = cv::Mat::zeros(gray.size(), CV_8UC1);

    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            unsigned char center = gray.at<unsigned char>(y, x);
            unsigned char code = 0;

            code |= (gray.at<unsigned char>(y - 1, x - 1) > center) << 7;
            code |= (gray.at<unsigned char>(y - 1, x) > center) << 6;
            code |= (gray.at<unsigned char>(y - 1, x + 1) > center) << 5;
            code |= (gray.at<unsigned char>(y, x + 1) > center) << 4;
            code |= (gray.at<unsigned char>(y + 1, x + 1) > center) << 3;
            code |= (gray.at<unsigned char>(y + 1, x) > center) << 2;
            code |= (gray.at<unsigned char>(y + 1, x - 1) > center) << 1;
            code |= (gray.at<unsigned char>(y, x - 1) > center) << 0;

            lbp.at<unsigned char>(y, x) = code;
        }
    }

    vector<double> histogram(256, 0.0);

    for (int y = 0; y < lbp.rows; ++y) {
        for (int x = 0; x < lbp.cols; ++x) {
            histogram[lbp.at<unsigned char>(y, x)]++;
        }
    }

    double total = lbp.rows * lbp.cols;
    for (double& value : histogram) {
        value /= total;
    }

    return histogram;
}

// Function to compare two LBP histograms
double compareHistograms(const vector<double>& hist1, const vector<double>& hist2) {
    double similarity = 0.0;
    for (size_t i = 0; i < hist1.size(); ++i) {
        similarity += min(hist1[i], hist2[i]);
    }
    return similarity;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <image_directory> <test_image>" << endl;
        return 1;
    }

    string imageDirectory = argv[1];
    string testImagePath = argv[2];

    // Load the test image and compute its LBP histogram
    cv::Mat testImage = cv::imread(testImagePath);
    if (testImage.empty()) {
        cerr << "Error: Unable to load test image!" << endl;
        return 1;
    }
    vector<double> testLBP = computeLBP(testImage);

    // Load all images from the directory
    vector<pair<string, vector<double>>> localLBPs;
    for (const auto& entry : fs::directory_iterator(imageDirectory)) {
        if (entry.is_regular_file()) {
            string imagePath = entry.path().string();
            cv::Mat img = cv::imread(imagePath);
            if (!img.empty()) {
                localLBPs.emplace_back(imagePath, computeLBP(img));
            }
        }
    }

    double globalBestSimilarity = -1;
    string globalBestPath = "";

    // Parallelize the comparison
    #pragma omp parallel
    {
        double localBestSimilarity = -1;
        string localBestPath = "";

        #pragma omp for
        for (size_t i = 0; i < localLBPs.size(); ++i) {
            const auto& [path, histogram] = localLBPs[i];
            double similarity = compareHistograms(testLBP, histogram);

            if (similarity > localBestSimilarity) {
                localBestSimilarity = similarity;
                localBestPath = path;
            }
        }

        // Update global results
        #pragma omp critical
        {
            if (localBestSimilarity > globalBestSimilarity) {
                globalBestSimilarity = localBestSimilarity;
                globalBestPath = localBestPath;
            }
        }
    }

    // Output the best match
    cout << "Best match: " << globalBestPath << " with similarity: " << (double)globalBestSimilarity*100 << "%" << endl;

    return 0;
}

