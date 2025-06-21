# Zero-shot evaluation with MIME
. run_eval.sh wise-east/mime-cropped all zero-shot mcq qwen3b
. run_eval.sh wise-east/mime-cropped all zero-shot ff qwen3b

. run_eval.sh wise-east/mime-cropped all zero-shot mcq qwen7b
. run_eval.sh wise-east/mime-cropped all zero-shot ff qwen7b

. run_eval.sh wise-east/mime-cropped all zero-shot mcq internvl
. run_eval.sh wise-east/mime-cropped all zero-shot ff internvl

. run_eval.sh wise-east/mime-cropped all zero-shot mcq gemini
. run_eval.sh wise-east/mime-cropped all zero-shot ff gemini

. run_eval.sh wise-east/mime-cropped all zero-shot mcq openai
. run_eval.sh wise-east/mime-cropped all zero-shot ff openai


# Zero-shot evaluation with REAL 
. run_eval.sh wise-east/mime-real-resized none zero-shot mcq qwen3b
. run_eval.sh wise-east/mime-real-resized none zero-shot ff qwen3b

. run_eval.sh wise-east/mime-real-resized none zero-shot mcq qwen7b
. run_eval.sh wise-east/mime-real-resized none zero-shot ff qwen7b

. run_eval.sh wise-east/mime-real-resized none zero-shot mcq internvl
. run_eval.sh wise-east/mime-real-resized none zero-shot ff internvl

. run_eval.sh wise-east/mime-real-resized none zero-shot mcq gemini
. run_eval.sh wise-east/mime-real-resized none zero-shot ff gemini

. run_eval.sh wise-east/mime-real-resized none zero-shot mcq openai
. run_eval.sh wise-east/mime-real-resized none zero-shot ff openai

# COT evaluation with MIME
. run_eval.sh wise-east/mime-cropped all cot mcq qwen3b
. run_eval.sh wise-east/mime-cropped all cot ff qwen3b

. run_eval.sh wise-east/mime-cropped all cot mcq qwen7b
. run_eval.sh wise-east/mime-cropped all cot ff qwen7b

. run_eval.sh wise-east/mime-cropped all cot mcq internvl
. run_eval.sh wise-east/mime-cropped all cot ff internvl

. run_eval.sh wise-east/mime-cropped all cot mcq gemini
. run_eval.sh wise-east/mime-cropped all cot ff gemini

. run_eval.sh wise-east/mime-cropped all cot mcq openai
. run_eval.sh wise-east/mime-cropped all cot ff openai

# Few-shot evaluation with MIME
. run_eval.sh wise-east/mime-cropped all few-shot mcq qwen3b
. run_eval.sh wise-east/mime-cropped all few-shot ff qwen3b

. run_eval.sh wise-east/mime-cropped all few-shot mcq qwen7b
. run_eval.sh wise-east/mime-cropped all few-shot ff qwen7b

. run_eval.sh wise-east/mime-cropped all few-shot mcq internvl
. run_eval.sh wise-east/mime-cropped all few-shot ff internvl

. run_eval.sh wise-east/mime-cropped all few-shot mcq gemini
. run_eval.sh wise-east/mime-cropped all few-shot ff gemini

. run_eval.sh wise-east/mime-cropped all few-shot mcq openai
. run_eval.sh wise-east/mime-cropped all few-shot ff openai
