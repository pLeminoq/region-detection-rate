/*
 * Copyright 2019 <Tamino Huxohl>
 */
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "measures.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
    int threshold;

    po::options_description desc("Compute measures for binary maps");
    desc.add_options()
        ("help,h", "produce help message")
        ("input-file", po::value<std::string>(), "input file containing a mapping from a ground truths to a (non-)binary maps")
        ("threshold,t", po::value<int>(&threshold)->default_value(100), "threshold which is applied to deal with non-binary maps")
        ("verbose,v", "verboseize statistics and decisions made by the region detection rate");
    po::positional_options_description pdesc;
    pdesc.add("input-file",  -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    if (!vm.count("input-file")) {
        std::cout << "Input file is missing!\n" << desc << std::endl;
        return 1;
    }

    const bool verbose = vm.count("verbose");

    std::ifstream inputFile(vm["input-file"].as<std::string>());
    std::string groundTruthFilename, foregroundMapFilename;
    measure::pixelwise::Quantities pixelwise;
    measure::rdr::Quantities rdr;
    while (inputFile >> groundTruthFilename >> foregroundMapFilename) {
        cv::Mat groundTruth = cv::imread(groundTruthFilename, cv::IMREAD_GRAYSCALE);
        cv::Mat foregroundMap = cv::imread(foregroundMapFilename, cv::IMREAD_GRAYSCALE);

        std::cout << "Evaluate: " << groundTruthFilename << "\t" << foregroundMapFilename << std::endl;

        cv::Mat binaryMap;
        cv::threshold(foregroundMap, binaryMap, vm["threshold"].as<int>(), 255, cv::THRESH_BINARY);

        pixelwise += measure::pixelwise::Quantities(binaryMap, groundTruth);

        std::vector<int>* detectedRegionLabels = new std::vector<int>();
        std::vector<int>* falsePredictionLabels = new std::vector<int>();
        measure::rdr::Stats st(binaryMap, groundTruth);
        rdr += measure::rdr::Quantities(st, 0.5, 0.25, detectedRegionLabels, falsePredictionLabels);

        if (!verbose) {
            continue;
        }

        std::cout << "Ground Truths:" << std::endl;
        std::cout << "Label\tArea\tLeft\tTop\tWidth\tHeight" << std::endl;
        for (const auto &entry : st.getGTStats()) {
            std::cout << entry.first << "\t";
            std::cout << entry.second.area << "\t";
            std::cout << entry.second.left << "\t";
            std::cout << entry.second.top << "\t";
            std::cout << entry.second.width << "\t";
            std::cout << entry.second.height << "\t";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Predictions:" << std::endl;
        std::cout << "Label\tArea\tLeft\tTop\tWidth\tHeight" << std::endl;
        for (const auto &entry : st.getPredictionStats()) {
            std::cout << entry.first << "\t";
            std::cout << entry.second.area << "\t";
            std::cout << entry.second.left << "\t";
            std::cout << entry.second.top << "\t";
            std::cout << entry.second.width << "\t";
            std::cout << entry.second.height << "\t";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "GT\tPrediction\tIntersection" << std::endl;
        for (const auto &entry1 : st.getIntersecions()) {
            for (const auto &entry2 : entry1.second) {
                std::cout << entry1.first << "\t";
                std::cout << entry2.first << "\t";
                std::cout << entry2.second << "\t";
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;

        cv::namedWindow("Detected Regions", cv::WINDOW_NORMAL);
        cv::resizeWindow("Detected Regions", 1000, (groundTruth.rows * 1000 / groundTruth.cols));
        cv::namedWindow("False Predictions", cv::WINDOW_NORMAL);
        cv::resizeWindow("False Predictions", 1000, (groundTruth.rows * 1000 / groundTruth.cols));

        measure::rdr::util::CCStats groundTruthCCStats = measure::rdr::util::connectedComponents(groundTruth);
        measure::rdr::util::CCStats predictionCCStats = measure::rdr::util::connectedComponents(binaryMap);
        cv::Mat detectedRegionImage = cv::Mat::zeros(groundTruth.size(), CV_8UC3);
        cv::Mat falsePredictionImage = cv::Mat::zeros(binaryMap.size(), CV_8UC3);

        const cv::Vec3b blue(255, 135, 36);
        const cv::Vec3b orange(28, 136, 237);
        for (int i = 0; i < groundTruth.rows; ++i) {
            for (int j = 0; j < groundTruth.cols; ++j) {
                const int groundTruthLabel = groundTruthCCStats.labels.at<int>(i, j);
                if (groundTruthLabel > 0) {
                    if (std::find(detectedRegionLabels->begin(), detectedRegionLabels->end(), groundTruthLabel) == detectedRegionLabels->end()) {
                        detectedRegionImage.at<cv::Vec3b>(i, j) = blue;
                    } else {
                        detectedRegionImage.at<cv::Vec3b>(i, j) = orange;
                    }
                }

                const int predictionLabel = predictionCCStats.labels.at<int>(i, j);
                if (predictionLabel > 0) {
                    if (std::find(falsePredictionLabels->begin(), falsePredictionLabels->end(), predictionLabel) == falsePredictionLabels->end()) {
                        falsePredictionImage.at<cv::Vec3b>(i, j) = orange;
                    } else {
                        falsePredictionImage.at<cv::Vec3b>(i, j) = blue;
                    }
                }
            }
        }

        cv::imshow("Detected Regions", detectedRegionImage);
        cv::imshow("False Predictions", falsePredictionImage);
        if (cv::waitKey(0) == 's') {
            const std::string detectedRegionsFilename = groundTruthFilename.substr(0, groundTruthFilename.size() - 4) + "_detected_regions.png";
            const std::string falsePredictionsFilename = foregroundMapFilename.substr(0, foregroundMapFilename.size() - 4) + "_false_predictions.png";
            std::cout << "Save detected region image to: " << detectedRegionsFilename << std::endl;
            cv::imwrite(detectedRegionsFilename, detectedRegionImage);
            std::cout << "Save false prediction image to: " << falsePredictionsFilename << std::endl;
            cv::imwrite(falsePredictionsFilename, falsePredictionImage);
            std::cout << std::endl;
        }
    }

    printf("Precision: %.3f\n", measure::pixelwise::precision(pixelwise));
    printf("Recall: %.3f\n", measure::pixelwise::recall(pixelwise));
    printf("FBeta: %.3f\n", measure::pixelwise::fBeta(pixelwise));
    printf("IoU: %.3f\n", measure::pixelwise::intersectionOverUnion(pixelwise));
    printf("RDR: %.3f", measure::rdr::regionDetectionRate(rdr));
    std::cout << " from " << rdr << std::endl;
}

