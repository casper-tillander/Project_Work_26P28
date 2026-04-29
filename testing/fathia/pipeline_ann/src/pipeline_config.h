#ifndef PIPELINE_CONFIG_H
#define PIPELINE_CONFIG_H


// #define USE_MODEL_RF      
#define USE_MODEL_ANN         

                                     
#if defined(USE_MODEL_RF) && defined(USE_MODEL_ANN)
    #error "Define only ONE of USE_MODEL_RF or USE_MODEL_ANN"
#endif

#if !defined(USE_MODEL_RF) && !defined(USE_MODEL_ANN)
    #error "Define at least one of USE_MODEL_RF or USE_MODEL_ANN"
#endif

#endif 
