#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <cmath>
#include <cstring>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Function to convert an image to grayscale
Mat preprocessImage(const Mat &image) {
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    return grayImage;
}

// Function to compute the LBP histogram of an image
vector<int> computeLBP(const Mat &image) {
    Mat lbpImage(image.size(), CV_8UC1);
    vector<int> histogram(256, 0);

    for (int i = 1; i < image.rows - 1; ++i) {
        for (int j = 1; j < image.cols - 1; ++j) {
            uchar center = image.at<uchar>(i, j);
            uchar code = 0;
            code |= (image.at<uchar>(i - 1, j - 1) >= center) << 7;
            code |= (image.at<uchar>(i - 1, j) >= center) << 6;
            code |= (image.at<uchar>(i - 1, j + 1) >= center) << 5;
            code |= (image.at<uchar>(i, j + 1) >= center) << 4;
            code |= (image.at<uchar>(i + 1, j + 1) >= center) << 3;
            code |= (image.at<uchar>(i + 1, j) >= center) << 2;
            code |= (image.at<uchar>(i + 1, j - 1) >= center) << 1;
            code |= (image.at<uchar>(i, j - 1) >= center);
            lbpImage.at<uchar>(i, j) = code;
            histogram[code]++;
        }
    }

    return histogram;
}

// Function to compute histogram intersection
double compareHistograms(const vector<int> &h1, const vector<int> &h2) {
    double sum = 0;
    for (size_t i = 0; i < h1.size(); ++i) {
        sum += min(h1[i], h2[i]);
    }
    return sum; // Higher value means more similar
}

// Struct to hold the result of similarity comparison
struct Result {
    double similarity;
    char path[256];
};



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <num_procs> ./image_processing_and_matching <input_dir> <test_image>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string inputDir = argv[1];
    string testImagePath = argv[2];

    vector<string> imagePaths;
    if (rank == 0) {
        // Gather all image file paths from the input directory
        for (const auto &entry : fs::directory_iterator(inputDir)) {
            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                imagePaths.push_back(entry.path().string());
            }
        }
    }

    // Serialize image paths for scattering
    vector<int> sendcounts(size, 0); // Make sure each process has an initialized sendcounts array
    vector<int> displs(size, 0); // Make sure each process has an initialized displacements array
    string serializedPaths;

    if (rank == 0) {
        for (const string &path : imagePaths) {
            serializedPaths += path + '\n';
        }

        int totalLength = serializedPaths.size();
        int base = totalLength / size;
        int remainder = totalLength % size;

        int offset = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = base + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    // Debug: Check sendcounts and displacements
    //if (rank == 0) {
        //cout << "Sendcounts: ";
        //for (int i = 0; i < size; ++i) {
        //    cout << sendcounts[i] << " ";
        //}
        //cout << endl;

        //cout << "Displacements: ";
        //for (int i = 0; i < size; ++i) {
        //    cout << displs[i] << " ";
        //}
        //cout << endl;
    //}

    // Scatter serialized paths
    int localSize = sendcounts[rank]; // Correctly calculate local size for each process
    vector<char> localBuffer(localSize);

    MPI_Scatterv(serializedPaths.data(), sendcounts.data(), displs.data(), MPI_CHAR,
                 localBuffer.data(), localSize, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Debug: Check localBuffer after scatter
    //if (rank == 0) {
    //    cout << "Serialized paths (rank " << rank << "): " << serializedPaths << endl;
    //}

    // Deserialize paths in local buffer
    stringstream ss(string(localBuffer.begin(), localBuffer.end()));
    vector<string> localPaths;
    string path;

    while (getline(ss, path, '\n')) {
        localPaths.push_back(path);
    }

    // Debug: Check localPaths
    //cout << "Rank " << rank << " Local Paths: " << endl;
    //for (const auto &p : localPaths) {
    //    cout << p << endl;
    //}

    // Compute local histograms
    vector<pair<string, vector<int>>> localLBPs;
    for (const string &path : localPaths) {
        Mat image = imread(path);
        if (image.empty()) {
            cerr << "Error: Could not read image: here" << path << endl;
            MPI_Finalize();  // Clean up MPI resources
            return 1;        // Return with error code
        }
        Mat grayImage = preprocessImage(image);
        vector<int> lbpHistogram = computeLBP(grayImage);
        localLBPs.push_back({path, lbpHistogram});
    }

    // Read and compute LBP for test image
    vector<int> testLBP(256);
    if (rank == 0) {
        Mat testImage = imread(testImagePath);
        if (testImage.empty()) {
            cerr << "Error: Could not read image: there" << testImagePath << endl;
            MPI_Finalize();  // Clean up MPI resources
            return 1;        // Return with error code
        }
        Mat grayTestImage = preprocessImage(testImage);
        testLBP = computeLBP(grayTestImage);
    }

    // Broadcast test LBP histogram
    MPI_Bcast(testLBP.data(), 256, MPI_INT, 0, MPI_COMM_WORLD);

    // Compare local histograms with test LBP
    // Local result: Store best match from this rank
double bestMatch = -1;
string bestImage = "";
for (const auto &[path, histogram] : localLBPs) {
    double similarity = compareHistograms(testLBP, histogram);
    if (similarity > bestMatch) {
        bestMatch = similarity;
        bestImage = path;
    }
}

// Local result
Result localResult = {bestMatch, ""};
strncpy(localResult.path, bestImage.c_str(), 255);
localResult.path[255] = '\0'; // Ensure null-termination

//cout << "Rank " << rank << " local result: " << localResult.path << " local similarity: " << localResult.similarity << endl;

double globalBestSimilarity = 0.0; // Initialize with a small value
MPI_Reduce(&localResult.similarity, &globalBestSimilarity, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
char gatheredPaths[size][256]; // Array to store all paths at rank 0
double gatheredSimilarities[size]; // Array to store similarities at rank 0
MPI_Gather(&localResult.similarity, 1, MPI_DOUBLE, gatheredSimilarities, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Gather(localResult.path, 256, MPI_CHAR, gatheredPaths, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
    int bestRank = -1;
    double highestSimilarity = -1;
    
    // Compare all ranks' similarities (gathered locally)
    for (int i = 0; i < size; i++) {
        if (gatheredSimilarities[i] > highestSimilarity) {
            highestSimilarity = gatheredSimilarities[i];
            bestRank = i;
        }
    }
    
    // Output the best match from rank with highest similarity
    cout << "Best match: " << gatheredPaths[bestRank] << " with similarity: " << (double)(highestSimilarity/9604)*100 << "%" << endl;
}

    MPI_Finalize();

    return 0;
}

