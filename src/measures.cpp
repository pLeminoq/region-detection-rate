/*
 * Copyright 2019 <Tamino Huxohl>
 */
#include "measures.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iostream>

namespace measure {
namespace pixelwise {

Quantities::Quantities(const double truePositives, const double trueNegatives, const double falsePositives, const double falseNegatives) {
    this->truePositives = truePositives;
    this->trueNegatives = trueNegatives;
    this->falsePositives = falsePositives;
    this->falseNegatives = falseNegatives;
}

Quantities::Quantities(const cv::Mat& prediction, const cv::Mat& groundTruth) {
    const cv::Mat intersection(prediction.size(), CV_8U);
    bitwise_and(prediction, groundTruth, intersection);

    this->truePositives = cv::sum(intersection)[0] / 255.0;
    this->falsePositives = (cv::sum(prediction)[0] / 255.0) - this->truePositives;
    this->falseNegatives = (cv::sum(groundTruth)[0] / 255.0) - this->truePositives;
    this->trueNegatives = (prediction.rows * prediction.cols)- (this->truePositives + this->falsePositives + this->falseNegatives);
}

Quantities& Quantities::operator+=(const Quantities& quantities) {
    this->truePositives += quantities.getTruePositives();
    this->trueNegatives += quantities.getTrueNegatives();
    this->falsePositives += quantities.getFalsePositives();
    this->falseNegatives += quantities.getFalseNegatives();
    return *this;
}

Quantities operator+(Quantities lhs, const Quantities &rhs) {
    lhs += rhs;
    return lhs;
}

std::ostream& operator<<(std::ostream& os, const Quantities q) {
    return os << "(TP, TN, FP, FN) = (" << q.truePositives << ", " << q.trueNegatives << ", " << q.falsePositives << ", " << q.falseNegatives << ")";
}

double Quantities::getTruePositives() const {
    return truePositives;
}

double Quantities::getTrueNegatives() const {
    return trueNegatives;
}

double Quantities::getFalsePositives() const {
    return falsePositives;
}

double Quantities::getFalseNegatives() const {
    return falseNegatives;
}

std::string Quantities::toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
}

double precision(const Quantities& quantities) {
    double denominator = quantities.getTruePositives() + quantities.getFalsePositives();
    if (denominator == 0.0) {
        throw std::invalid_argument("Precision cannot be computed without TP and FP: " + quantities.toString());
    }
    return quantities.getTruePositives() / denominator;
}

double recall(const Quantities& quantities) {
    double denominator = quantities.getTruePositives() + quantities.getFalseNegatives();
    if (denominator == 0.0) {
        throw std::invalid_argument("Recall/TPR cannot be computed without TP and FN: " + quantities.toString());
    }
    return quantities.getTruePositives() / denominator;
}

double truePositiveRate(const Quantities& quantities) {
    return recall(quantities);
}

double falsePositiveRate(const Quantities& quantities) {
    const double denominator = quantities.getFalsePositives() + quantities.getTrueNegatives();
    if (denominator == 0.0) {
        throw std::invalid_argument("FPR cannot be computed without FP and TN: " + quantities.toString());
    }
    return quantities.getFalsePositives() / denominator;
}

double fBeta(const Quantities& quantities, double betaSquared) {
    return fBeta(precision(quantities), recall(quantities), betaSquared);
}

double fBeta(double precision, double recall, double betaSquared) {
    double numerator = (1 + betaSquared) * precision * recall;
    double denominator = (betaSquared) * precision + recall;
    return numerator / denominator;
}

double intersectionOverUnion(const Quantities& quantities) {
    const double denominator = quantities.getTruePositives() + quantities.getFalsePositives() + quantities.getFalseNegatives();
    if (denominator == 0.0) {
        throw std::invalid_argument("IoU cannot be computed without TP, FP and FN: " + quantities.toString());
    }
    return quantities.getTruePositives() / denominator;
}

double meanAbsoluteError(const cv::Mat &saliencyMap, const cv::Mat &groundTruth) {
    const cv::Mat absoluteDifference(saliencyMap.size(), CV_8U);
    cv::absdiff(saliencyMap, groundTruth, absoluteDifference);
    return sum(absoluteDifference)[0] / static_cast<double>(absoluteDifference.size().height * absoluteDifference.size().width * 255.0);
}

}  // namespace pixelwise

namespace rdr {

namespace util {

CCStats connectedComponents(const cv::Mat &image, const int connectivity, const int labelType) {
    CCStats stats;
    stats.image = image;
    stats.labelCount = connectedComponentsWithStats(image, stats.labels, stats.stats, stats.centroids, connectivity, labelType);
    return stats;
}

cv::Point findPointWithLabel(const CCStats& ccStats, const int label) {
    cv::Point pointWithLabel(
            ccStats.stats.at<int>(label, cv::CC_STAT_LEFT),
            ccStats.stats.at<int>(label, cv::CC_STAT_TOP));

    while (ccStats.labels.at<int>(pointWithLabel) != label) {
        pointWithLabel.x++;
    }

    return pointWithLabel;
}

}  // namespace util

Stats::Stats(const cv::Mat& prediction, const cv::Mat& groundTruth) {
    // compute connected components of ground truth
    util::CCStats groundTruthCCStats = util::connectedComponents(groundTruth);

    // compute connected components of prediction
    util::CCStats predictionCCStats = util::connectedComponents(prediction);

    // compute the intersection between ground truth and prediction
    const cv::Mat intersection(prediction.size(), CV_8U);
    bitwise_and(prediction, groundTruth, intersection);
    // compute connected components of intersection
    util::CCStats intersectionCCStats = util::connectedComponents(intersection);

    // save ground_truth stats
    for (int groundTruthLabel = 1; groundTruthLabel < groundTruthCCStats.labelCount; groundTruthLabel++) {
        this->groundTruthStats[groundTruthLabel].area = groundTruthCCStats.stats.at<int>(groundTruthLabel, cv::CC_STAT_AREA);
        this->groundTruthStats[groundTruthLabel].left = groundTruthCCStats.stats.at<int>(groundTruthLabel, cv::CC_STAT_LEFT);
        this->groundTruthStats[groundTruthLabel].top = groundTruthCCStats.stats.at<int>(groundTruthLabel, cv::CC_STAT_TOP);
        this->groundTruthStats[groundTruthLabel].width = groundTruthCCStats.stats.at<int>(groundTruthLabel, cv::CC_STAT_WIDTH);
        this->groundTruthStats[groundTruthLabel].height = groundTruthCCStats.stats.at<int>(groundTruthLabel, cv::CC_STAT_HEIGHT);
    }

    // save prediction stats
    for (int predictionLabel = 1; predictionLabel < predictionCCStats.labelCount; ++predictionLabel) {
        this->predictionStats[predictionLabel].area = predictionCCStats.stats.at<int>(predictionLabel, cv::CC_STAT_AREA);
        this->predictionStats[predictionLabel].left = predictionCCStats.stats.at<int>(predictionLabel, cv::CC_STAT_LEFT);
        this->predictionStats[predictionLabel].top = predictionCCStats.stats.at<int>(predictionLabel, cv::CC_STAT_TOP);
        this->predictionStats[predictionLabel].width = predictionCCStats.stats.at<int>(predictionLabel, cv::CC_STAT_WIDTH);
        this->predictionStats[predictionLabel].height = predictionCCStats.stats.at<int>(predictionLabel, cv::CC_STAT_HEIGHT);
    }

    // iterate over all regions in the intersection to fill the array and the map created above
    // note that this and all following loops start iteration at 1 because 0 is the background label
    for (int intersectionLabel = 1; intersectionLabel < intersectionCCStats.labelCount; intersectionLabel++) {
        cv::Point pointWithLabel = util::findPointWithLabel(intersectionCCStats, intersectionLabel);
        // because the intersection has a label at that point, the prediction and the ground truth also
        // have to have labels there
        int groundTruthLabel = groundTruthCCStats.labels.at<int>(pointWithLabel);
        int predictionLabel = predictionCCStats.labels.at<int>(pointWithLabel);
        assert(groundTruthLabel != 0);  // 0 is the background label which should not be retrieved here
        assert(predictionLabel != 0);  // 0 is the background label which should not be retrieved here

        // init as zero
        if (this->intersections[groundTruthLabel].find(predictionLabel) == this->intersections[groundTruthLabel].end()) {
            this->intersections[groundTruthLabel][predictionLabel] = 0;
        }
        // add area of intersection
        this->intersections[groundTruthLabel][predictionLabel] += intersectionCCStats.stats.at<int>(intersectionLabel, cv::CC_STAT_AREA);
    }
}

std::map<int, RegionStats> Stats::getGTStats() const {
    return groundTruthStats;
}

std::map<int, RegionStats> Stats::getPredictionStats() const {
    return predictionStats;
}
std::map<int, std::map<int, int>> Stats::getIntersecions() const {
    return intersections;
}

Quantities::Quantities(const double regions, const double detectedRegions, const double falsePredictions) {
    this->regions = regions;
    this->detectedRegions = detectedRegions;
    this->falsePredictions = falsePredictions;
}

Quantities::Quantities(const Stats& stats, const double regionThreshold, const double predictionThreshold, std::vector<int>* detectedRegionLabels, std::vector<int>* falsePredictionLabels) {
    this->regions = stats.getGTStats().size();
    this->detectedRegions = 0;
    this->falsePredictions = 0;

    std::map<int, bool> predictionContributionMap;
    for (auto const &entry : stats.getPredictionStats()) {
        predictionContributionMap[entry.first] = false;
    }

    for (const auto& entry1 : stats.getIntersecions()) {
        const int groundTruthLabel = entry1.first;
        double groundTruthCoverage = 0.0;

        for (const auto& entry2 : entry1.second) {
            const int predictionLabel = entry2.first;
            const int intersectionArea = entry2.second;

            // verify threshold
            const double predictionCoverage = intersectionArea / static_cast<double>(stats.getPredictionStats()[predictionLabel].area);
            if (predictionCoverage >= predictionThreshold) {
                groundTruthCoverage += intersectionArea / static_cast<double>(stats.getGTStats()[groundTruthLabel].area);
                predictionContributionMap[predictionLabel] = true;
            }
        }

        if (groundTruthCoverage >= regionThreshold) {
            this->detectedRegions++;
            if (NULL != detectedRegionLabels) {
                detectedRegionLabels->push_back(groundTruthLabel);
            }
        }
    }

    for (const auto &entry : predictionContributionMap) {
        if (entry.second) {
            continue;
        }
        this->falsePredictions++;
        if (NULL != falsePredictionLabels) {
            falsePredictionLabels->push_back(entry.first);
        }
    }
}

Quantities& Quantities::operator+=(const Quantities& quantities) {
    this->regions += quantities.getRegions();
    this->detectedRegions += quantities.getDetectedRegions();
    this->falsePredictions += quantities.getFalsePredictions();
    return *this;
}

Quantities operator+(Quantities lhs, const Quantities &rhs) {
    lhs += rhs;
    return lhs;
}

std::ostream& operator<<(std::ostream& os, const Quantities q) {
    return os << "(Regions, Detected-Regions, False-Predictions) = (" << q.regions << ", " << q.detectedRegions << ", " << q.falsePredictions << ")";
}

double Quantities::getRegions() const {
    return regions;
}

double Quantities::getDetectedRegions() const {
    return detectedRegions;
}

double Quantities::getFalsePredictions() const {
    return falsePredictions;
}

std::string Quantities::toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
}

double regionDetectionRate(const Quantities& rdrQuantities, const double falsePredictionWeight) {
    double denominator = rdrQuantities.getRegions() + falsePredictionWeight * rdrQuantities.getFalsePredictions();
    if (denominator == 0.0) {
        throw std::invalid_argument("RDR cannot be computed without regions and false predictions: " + rdrQuantities.toString());
    }
    return rdrQuantities.getDetectedRegions() / denominator;
}

}  // namespace rdr

}  // namespace measure
