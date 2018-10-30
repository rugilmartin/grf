//
//  main.cpp
//  grf
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
#include "serialization/ForestSerializer.h"

#include <iostream>
#include <string>

#include "cxxopts.hpp"

ForestOptions parse_forest_options(cxxopts::ParseResult result) {
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

Forest load_forest_from_file(std::string filename) {
    std::ifstream infile;
    infile.open(filename, std::ios::binary);
    if (!infile.good()) throw std::runtime_error("Could not open " + filename + ".");
    ForestSerializer forest_loader;
    Forest forest = forest_loader.deserialize(infile);
    return(forest);
}

Forest train_regression_forest(std::unique_ptr<Data> data, ForestOptions forest_options, uint outcome_index) {
    ForestTrainer trainer = ForestTrainers::regression_trainer(outcome_index);
    Forest forest = trainer.train(data.get(), forest_options);
    return(forest);
}

void save_forest_to_file(std::string filename, Forest forest) {
    std::ofstream outfile;
    outfile.open(filename, std::ios::binary);
    if (!outfile.good()) throw std::runtime_error("Could not open " + filename + ".");
    ForestSerializer serial;
    serial.serialize(outfile, forest);
    outfile.close();
}

void save_predictions_to_file(std::string filename, std::vector<Prediction> predictions) {
    std::ofstream outfile;
    outfile.open(filename);
    if (!outfile.good()) throw std::runtime_error("Could not open " + filename + ".");
    for (auto&p: predictions) {
        auto yhat = p.get_predictions()[0];
        outfile << yhat << "\n";
    }
    outfile.close();
}


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
    ("y,outcome_index", "Outcome index", cxxopts::value<uint>())
    ("I,input_file", "Input file", cxxopts::value<std::string>())
    ("W,write", "Write forest to file", cxxopts::value<std::string>()->default_value(""))
    ("L,load", "Load forest from file", cxxopts::value<std::string>()->default_value(""))
    ("O,output_file", "File that will receive predictions", cxxopts::value<std::string>()->default_value(""))
    ;

    auto result = options.parse(argc, argv);
    
    // Parse forest options and other command line options
    ForestOptions forest_options = parse_forest_options(result);

    // File management
    std::string input_file = result.count("input_file") ? result["input_file"].as<std::string>() : throw std::runtime_error("Input file is required.");
    std::string output_file = result.count("output_file") ? result["output_file"].as<std::string>() : "";
    
    std::string in_serial_name = result.count("load") ? result["load"].as<std::string>() : "";
    std::string out_serial_name = result.count("write") ? result["write"].as<std::string>() : "";

    // Column indices
    uint outcome_index = result.count("outcome_index") ? result["outcome_index"].as<uint>() : throw std::runtime_error("Outcome index is required");

    // Trains forest or loads from memory data
    std::unique_ptr<Data> data(load_data(input_file));
    Forest forest = in_serial_name.size() ? load_forest_from_file(in_serial_name) : train_regression_forest(std::move(data), forest_options, outcome_index);
    
    // Serialize forest, if applicable
    if (!out_serial_name.empty()) save_forest_to_file(out_serial_name, forest);

    // Save predictions
    if (!output_file.empty()) {
        ForestPredictor predictor = ForestPredictors::regression_predictor(forest_options.get_num_threads(), forest_options.get_ci_group_size());
        std::vector<Prediction> predictions = predictor.predict_oob(forest, data.get());
        save_predictions_to_file(output_file, predictions);
    }
    
    return 0;
}
