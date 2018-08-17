//
//  main.cpp
//  grf_xcode
//
//  Created by Vitor Baisi Hadad on 8/6/18.
//  Copyright © 2018 atheylab. All rights reserved.
//

#include "sampling/SamplingOptions.h"
#include "commons/utility.h"
#include "forest/ForestPredictor.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainer.h"
#include "forest/ForestTrainers.h"
#include "utilities/ForestTestUtilities.h"

#include <iostream>
#include <string>

#include "cxxopts.hpp"

ForestOptions parse_option(cxxopts::ParseResult result) {
    ForestOptions opt(
        result["num_trees"].as<uint>(),
        result["ci_group_size"].as<uint>(),
        result["sample_fraction"].as<double>(),
        result["mtry"].as<uint>(),
        result["min_node_size"].as<uint>(),
        result["honesty"].as<bool>(),
        result["alpha"].as<double>(),
        result["imbalance_penalty"].as<double>(),
        result["num_threads"].as<uint>(),
        result["random_seed"].as<uint>(),
        std::vector<size_t>{}, // sample_clusters
        0); // samples_per_cluster
    return(opt);
};


int main(int argc, char * argv[]) {
    
    // Argument parser
    cxxopts::Options options(argv[0], " - example command line options");
    
    options.add_options()
    ("f,sample_fraction", "an apple", cxxopts::value<double>()->default_value("0.5"))
    ("n,min_node_size", "Minimum node side", cxxopts::value<uint>()->default_value("5"))  // Needs change
    ("m,mtry", "Average covariates per tree", cxxopts::value<uint>()->default_value("5"))
    ("t,num_trees", "Number of trees", cxxopts::value<uint>()->default_value("2000"))
    ("d,num_threads", "Number of threads", cxxopts::value<uint>()->default_value("4"))
    ("h,honesty", "Input", cxxopts::value<bool>()->default_value("true"))
    ("g,ci_group_size", "Confidence interval group size", cxxopts::value<uint>()->default_value("2"))
    ("a,alpha", "Maximum imbalance parameter", cxxopts::value<double>()->default_value("5"))
    ("p,imbalance_penalty", "Imbalance penalty", cxxopts::value<double>()->default_value("0"))
    ("s,random_seed", "Seed", cxxopts::value<uint>()->default_value("12345"))
    ("b,compute_oob_predictions", "Compute OOB predictions", cxxopts::value<bool>()->default_value("true"))
    ("o,outcome_index", "Outcome index", cxxopts::value<uint>())
    ("i,input_file", "Input file", cxxopts::value<std::string>())
    ;

    auto result = options.parse(argc, argv);
    ForestOptions forest_options = parse_option(result);
    std::string input_file = result.count("input_file") ? result["input_file"].as<std::string>() : throw std::runtime_error("Input file is required");
    uint outcome_index = result.count("outcome_index") ? result["outcome_index"].as<uint>() : throw std::runtime_error("Outcome index is required");
    
    // Loading data
    std::cout << "Loading data\n";
    std::unique_ptr<Data> data(load_data(input_file));
    
    std::cout << "Training\n";
    ForestTrainer trainer = ForestTrainers::regression_trainer(outcome_index);
    Forest forest = trainer.train(data.get(), forest_options);
    
    std::cout << "Predicting\n";
    ForestPredictor predictor = ForestPredictors::regression_predictor(forest_options.get_num_threads(), forest_options.get_ci_group_size());
    std::vector<Prediction> predictions = predictor.predict_oob(forest, data.get());
    
    std::cout << "Computing MSE\n";
    double mse = 0;
    auto obs = forest.get_observations();
    auto n = obs.get_num_samples();
    for (int i=0; i < n; ++i) {
        double diff = (obs.get(0, i) - predictions[i].get_predictions()[0]);
        mse += diff * diff;
    }
    mse /= n;
    std::cout << "MSE: " << mse;

    return 0;
}
