/*
 * Copyright 2019 <Tamino Huxohl>
 */
#ifndef REGION_DETECTION_RATE_MEASURES_H_
#define REGION_DETECTION_RATE_MEASURES_H_

#include <opencv2/core/core.hpp>

#include <map>
#include <string>
#include <vector>

namespace measure {
namespace pixelwise {

/**
 * Class encapsulating the pixel-wise quantities: true positives, true negatives, false positive and false negatives.
 */
class Quantities {
 public:
    explicit Quantities(const double truePositives = 0, const double trueNegatives = 0, const double falsePositives = 0, const double falseNegatives = 0);
    Quantities(const cv::Mat& prediction, const cv::Mat& groundTruth);

    Quantities& operator+=(const Quantities& rhs);
    friend Quantities operator+(Quantities lhs, const Quantities &rhs);
    friend std::ostream& operator<<(std::ostream& os, const Quantities q);

    double getTruePositives() const;
    double getTrueNegatives() const;
    double getFalsePositives() const;
    double getFalseNegatives() const;

    std::string toString() const;

 private:
    double truePositives;
    double trueNegatives;
    double falsePositives;
    double falseNegatives;
};

/**
 * Compute a precision score for given pixel wise quantities.
 *
 * @param quantities entity encapsulating pixel-wise quantities.
 * @return the precision score given the quantities.
 */
double precision(const Quantities& quantities);

/**
 * Compute a recall score for given pixel wise quantities.
 *
 * @param quantities entity encapsulating pixel-wise quantities.
 * @return the recall score given the quantities.
 */
double recall(const Quantities& quantities);

/**
 * Compute the true positive rate for given pixel wise quantities.
 *
 * @param quantities entity encapsulating pixel-wise quantities.
 * @return the true positive rate given the quantities.
 */
double truePositiveRate(const Quantities& quantities);

/**
 * Compute the false positive rate for given pixel wise quantities.
 *
 * @param quantities entity encapsulating pixel-wise quantities.
 * @return the false positive rate score given the quantities.
 */
double falsePositiveRate(const Quantities& quantities);

/**
 * Compute the f beta measure given pixel-wise quantities.
 *
 * @param quantities entity encapsulating pixel-wise quantities.
 * @param betaSquared the beta squared value which defaults to 0.3
 * @return the value of the f beta measure.
 */
double fBeta(const Quantities& quantities, double betaSquared = 0.3);

/**
 * Compute the f beta measure from recall and precision scores.
 *
 * @param precision the precision score
 * @param recall the recall score
 * @param beta the beta value which defaults to 0.3
 * @return the value of the f beta measure.
 */
double fBeta(double precision, double recall, double beta = 0.3);

/**
 * Compute the intersection over union score given pixel-wise quantities.
 *
 * @param quantities entity encapsulating pixel-wise quantities.
 * @return the intersection over union score
 */
double intersectionOverUnion(const Quantities& quantities);

/**
 * Compute the mean absolute error.
 *
 * @param saliencyMap the saliency map
 * @param ground truth the ground truth
 * @return the mean absolute error
 */
double meanAbsoluteError(const cv::Mat &saliencyMap, const cv::Mat &groundTruth);

}  // namespace pixelwise

namespace rdr {

namespace util {

/**
 * Utility struct which contains all results of the openCV method connectedComponentsWithStats.
 */
struct CCStats {
    cv::Mat image, labels, stats, centroids;
    int labelCount;
};

/**
 * Utility method which saves the results of the openCV method connectedComponentsWithStats in a struct.
 */
CCStats connectedComponents(const cv::Mat& image, const int connectivity = 8, const int labelType = CV_32S);

/**
 * Find a point which has a certain label from connected components
 * computed by OpenCV.
 *
 * @param ccStats the stats struct
 * @param label the label with which to find a point
 * @return a point with the coordinates having the given label
 */ 
cv::Point findPointWithLabel(const CCStats& ccStats, const int label);

}  // namespace util

/**
 * Struct to compactly save the statistics of a region for the region detection rate.
 */
struct RegionStats {
    int area;
    int left, top, width, height;
};

/**
 * Class saving statistics for the computation of the region detection rate.
 * It contains the area and bounding rectangles for all regions in the ground truth
 * and all predicted regions.
 * In addition, it contains a mapping from ground truth region to predicted region
 * as well as the area of the intersection.
 */
class Stats {
 public:
    /**
     * Compute the stats from a binary prediction and a ground truth.
     */
    Stats(const cv::Mat& prediction, const cv::Mat& groundTruth);

    /**
     * Get a map from ground truth region label to the region statistics.
     */
    std::map<int, RegionStats> getGTStats() const;
    /**
     * Get a map from predicted region label to the region statistics.
     */
    std::map<int, RegionStats> getPredictionStats() const;
    /**
     * Get a mapping from ground truth label to intersecting prediction labels
     * as well as intersection areas.
     */
    std::map<int, std::map<int, int>> getIntersecions() const;

 private:
    std::map<int, RegionStats> groundTruthStats;
    std::map<int, RegionStats> predictionStats;
    std::map<int, std::map<int, int>> intersections;
};

/**
 * Class encapsulating the region detection rate quantities: 
 *      total number of regions, detected regions and false predictions.
 */
class Quantities {
 public:
    /**
     * Create rdr quantities by setting its values.
     */
    explicit Quantities(const double regions = 0, const double detectedRegions = 0, const double falsePredictions = 0);
    /**
     * Create rdr quantities from stats and by setting the required thresholds.
     * Labels for predicted regions and false predictions will be written into the vector parameters.
     */
    Quantities(const Stats& stats, const double regionThreshold, const double predictionThreshold, std::vector<int>* detectedRegionLabels, std::vector<int>* falsePredictionLabels);

    Quantities& operator+=(const Quantities & rhs);
    friend Quantities operator+(Quantities lhs, const Quantities &rhs);
    friend std::ostream& operator<<(std::ostream& os, const Quantities q);

    double getRegions() const;
    double getDetectedRegions() const;
    double getFalsePredictions() const;

    std::string toString() const;

 private:
    double regions;
    double detectedRegions;
    double falsePredictions;
};

/**
 * Compute the region detection rate from RDR quantities.
 *
 * @param quantities entity encapsulating region detection rate quantities.
 * @param falsePredictionWeight weight-factor for false predictions
 * @return the region detection rate
 */
double regionDetectionRate(const Quantities& quantities, const double falsePredictionWeight = 0.5);

}  // namespace rdr

}  // namespace measure

#endif  // REGION_DETECTION_RATE_MEASURES_H_
