#!/bin/bash

DATASET_NAME=$1 # options: wise-east/mime-cropped, wise-east/mime-real-resized, wise-east/mime-original, wise-east/mime-real-original
VARIANT=$2 # options: all, none 
EVAL_TYPE=$3 # options: zero-shot, cot, few-shot 
EVAL_FORMAT=$4 # options: mcq, ff
MODEL_TYPE=$5 # options: gemini, qwen3b, qwen7b, openai, internvl
# MODEL_NAME=$5 # options: gemini-1.5-flash, Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, gpt-4o-mini, OpenGVLab/InternVL2_5-8B 

# if usage is not correct, print the usage 
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset_name> <variant> <eval_type> <eval_format> <model_type>"
    echo "dataset_name: wise-east/mime-cropped, wise-east/mime-real-resized, wise-east/mime-original, wise-east/mime-real-original"
    echo "variant: all, none, base+blank@0, adversarial+blank@0, woman+blank@0, base+aligned@0, base+misaligned@0, adversarial+aligned@0, adversarial+misaligned@0, base+blank@90, base+blank@180, base+blank@270"
    echo "eval_type: zero-shot, cot, few-shot"
    echo "eval_format: mcq, ff" # mcq: multiple choice question, ff: free-form question 
    echo "model_type:  qwen3b, qwen7b, internvl, openai, gemini"
    exit 1
fi

# check if the dataset name is a valid option 
if [ "$DATASET_NAME" != "wise-east/mime-cropped" ] && [ "$DATASET_NAME" != "wise-east/mime-real-resized" ] && [ "$DATASET_NAME" != "wise-east/mime-original" ] && [ "$DATASET_NAME" != "wise-east/mime-real-original" ]; then
    echo "Error: DATASET_NAME is not a valid option. It should be one of the following: wise-east/mime-cropped, wise-east/mime-real-resized, wise-east/mime-original, wise-east/mime-real-original"
    exit 1
fi

# check if the variant is a valid option 
if [ "$VARIANT" != "all" ] && [ "$VARIANT" != "none" ] && [ "$VARIANT" != "base+blank@0" ] && [ "$VARIANT" != "adversarial+blank@0" ] && [ "$VARIANT" != "woman+blank@0" ] && [ "$VARIANT" != "base+aligned@0" ] && [ "$VARIANT" != "base+misaligned@0" ] && [ "$VARIANT" != "adversarial+aligned@0" ] && [ "$VARIANT" != "adversarial+misaligned@0" ] && [ "$VARIANT" != "base+blank@90" ] && [ "$VARIANT" != "base+blank@180" ] && [ "$VARIANT" != "base+blank@270" ]; then
    echo "Error: VARIANT is not a valid option. It should be one of the following: all, none, base+blank@0, adversarial+blank@0, woman+blank@0, base+aligned@0, base+misaligned@0, adversarial+aligned@0, adversarial+misaligned@0, base+blank@90, base+blank@180, base+blank@270"
    exit 1
fi

# check if the eval type is a valid option 
if [ "$EVAL_TYPE" != "zero-shot" ] && [ "$EVAL_TYPE" != "cot" ] && [ "$EVAL_TYPE" != "few-shot" ]; then
    echo "Error: EVAL_TYPE is not a valid option. It should be one of the following: zero-shot, cot, few-shot"
    exit 1
fi

# check if the eval format is a valid option 
if [ "$EVAL_FORMAT" != "mcq" ] && [ "$EVAL_FORMAT" != "ff" ]; then
    echo "Error: EVAL_FORMAT is not a valid option. It should be one of the following: mcq, ff"
    exit 1
fi

# check if the model type is a valid option 
if [ "$MODEL_TYPE" != "gemini" ] && [ "$MODEL_TYPE" != "qwen3b" ] && [ "$MODEL_TYPE" != "qwen7b" ] && [ "$MODEL_TYPE" != "openai" ] && [ "$MODEL_TYPE" != "internvl" ]; then
    echo "Error: MODEL_TYPE is not a valid option. It should be one of the following: gemini, qwen3b, qwen7b, openai, internvl"
    exit 1
fi


function run_eval() {
        model_name=$1
        model_type=$2
        dataset_name=$3
        variant=$4
        eval_type=$5
        eval_format=$6

        # set api key according to the model type 
        if [ "$model_type" == "gemini" ]; then
            api_key=$GEMINI_API_KEY
        elif [ "$model_type" == "openai" ]; then
            api_key=$OPENAI_API_KEY
        else
            api_key=none
        fi
        
        mimeeval run $eval_format --dataset-name $dataset_name \
        --model-name $model_name --model $model_type --api-key $api_key --eval-type $eval_type --variant $variant

        if [ "$eval_format" == "ff" ]; then
            mimeeval score --predictions results/mime/$dataset_name/$eval_type/${variant}/raw_${eval_format}_${model_type}_${model_name}.json --dataset-name $dataset_name --threshold 0.5 # score free-form results 
        fi
}
        

if [ "$MODEL_TYPE" == "gemini" ]; then
    run_eval gemini-1.5-flash gemini $DATASET_NAME $VARIANT $EVAL_TYPE $EVAL_FORMAT
elif [ "$MODEL_TYPE" == "qwen3b" ]; then
    run_eval Qwen/Qwen2.5-VL-3B-Instruct qwen25vl $DATASET_NAME $VARIANT $EVAL_TYPE $EVAL_FORMAT
elif [ "$MODEL_TYPE" == "qwen7b" ]; then
    run_eval Qwen/Qwen2.5-VL-7B-Instruct qwen25vl $DATASET_NAME $VARIANT $EVAL_TYPE $EVAL_FORMAT
elif [ "$MODEL_TYPE" == "openai" ]; then
    run_eval gpt-4o-mini openai $DATASET_NAME $VARIANT $EVAL_TYPE $EVAL_FORMAT
elif [ "$MODEL_TYPE" == "internvl" ]; then
    run_eval OpenGVLab/InternVL2_5-8B internvl2_5 $DATASET_NAME $VARIANT $EVAL_TYPE $EVAL_FORMAT
fi
