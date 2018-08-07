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

#include <memory>
#include <iostream>
#include <string>


int main(int argc, const char * argv[]) {
    
    std::string filename = "test/forest/resources/gaussian_data.csv"; 
    std::unique_ptr<Data> data(load_data(filename));
    
    uint outcome_index = 10;
    ForestTrainer trainer = ForestTrainers::regression_trainer(outcome_index);
    ForestOptions options = ForestTestUtilities::default_honest_options();
    Forest forest = trainer.train(data.get(), options);
    
    return 0;
}
