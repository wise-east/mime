import pandas as pd 
from pathlib import Path 
from argparse import ArgumentParser
from typing import List, Dict
from MimeEval.utils.compute_human_eval_results import calculate_accuracies_by_groups
import numpy as np 
from loguru import logger
from scipy.stats import ttest_ind, ttest_ind_from_stats

model_name_mapping = {
    "gemini_gemini-1.5-flash": "Gemini 1.5 Flash", 
    "gemini-few-shot": "Gemini 1.5 Flash", 
    "internvl2_5_OpenGVLab_InternVL2_5-8B": "InternVL2.5 (8B)",
    "openai_gpt-4o-mini": "GPT-4o Mini",
    "openai-few-shot": "GPT-4o Mini",
    "phi35_microsoft_Phi-3.5-vision-instruct": "Phi-3.5",
    "qwen25vl_Qwen_Qwen2.5-VL-7B-Instruct": "Qwen2.5 VL (7B)",
    "qwen25vl_Qwen_Qwen2.5-VL-3B-Instruct": "Qwen2.5 VL (3B)",
    "qwen2vl_Qwen_Qwen2-VL-7B-Instruct": "Qwen2 VL (7B)",
}


def load_results_from_folder(folder_path:Path) -> List[Dict]: 
    """
    Load evaluation results from a folder of csv files. 

    Args:
        folder_path (str): Path to the folder containing the csv files. 

    Raises:
        ValueError: If no column 'correct' or 'is_correct' is found in the csv file.

    Returns:
        list: A list of dictionaries containing the results. 
    """
    
    results_fn = sorted(folder_path.glob("*.csv"), key=lambda x: (x.stem.split("_")[2], x.stem.split("_")[1])) # sort by (model name, test format) e.g., qwen25vl, mcq
    
    results =[] 
    
    for result_fn in results_fn:
        model_name = '_'.join(result_fn.stem.split("_")[2:])
        model_name = model_name_mapping[model_name]
        test_format = result_fn.stem.split("_")[1]
        
        # data processing is different for the real data results because total count is different and no aggregation per avatar, background, angle is needed
        if "results_real_data" in str(result_fn): 
            try: 
                score_df = pd.read_csv(result_fn) 
                if "correct" not in score_df.columns and "is_correct" not in score_df.columns:
                    raise ValueError(f"No column 'correct' or 'is_correct' found in {result_fn}")
                             
                column_name = "correct" if "correct" in score_df.columns else "is_correct"
                # accuracy is number of correct over total where correct column is True
                accuracy = score_df[score_df[column_name] == True].shape[0] / score_df.shape[0] * 100 
                accuracy = round(accuracy, 2)

                results.append({
                    "model": model_name,
                    "test_format": test_format,
                    "avatar": "real",
                    "background_config": "real",
                    "angle": "real",
                    "accuracy": accuracy
                })
                continue 
                
            except Exception as e:
                print(f"Error reading {result_fn}: {e}")
        
        else: 
            score_df = pd.read_csv(result_fn) 
            if "correct" in score_df.columns:
                full_data_df["correct"] = score_df["correct"]
            elif "is_correct" in score_df.columns:
                full_data_df["correct"] = score_df["is_correct"]
            else:
                raise ValueError(f"No column 'correct' or 'is_correct' found in {result_fn}")
                        
            # aggregate results based on different configurations 
            agg = full_data_df.groupby(["avatar", "angle", "background_config"]).agg(
                accuracy = ("correct", "mean"),
                # count = ("correct", "count")
            ).reset_index()
            
            # multiply accuracy * 100 and keep it to two decimal places
            agg["accuracy"] = agg["accuracy"] * 100
            agg["accuracy"] = agg["accuracy"].round(2)
                    
            for idx, row in agg.iterrows():
                results.append({
                    "model": model_name,
                    "test_format": test_format,
                    "avatar": row["avatar"],
                    "angle": row["angle"],
                    "background_config": row["background_config"],
                    "accuracy": row["accuracy"]
                })
    
    return results 

parser = ArgumentParser()
parser.add_argument("-t", "--table_num", type=int, default=1, help="options: 1, 2, 3, 4, 5, 6. 1: mime-base vs real data, 2: mime-full results, 3: gender variation results, 4: angle variation results, 5: full improvement results, 6: per actor results")
args = parser.parse_args()

PACKAGE_DIR = Path(__file__).parent

full_data = PACKAGE_DIR / "data" / "mime_data_legacy.jsonl"
# if full_data does not exist, run `python src/MimeEval/utils/create_data_jsonl.py` to create it
if not full_data.exists():
    print(f"Error: {full_data} does not exist. Please run `python src/MimeEval/utils/create_data_jsonl.py` to create it.")
    exit()

full_data_df = pd.read_json(full_data, lines=True)

all_results= []
methods = ["zero-shot", "few-shot", "cot"]

# load MIME results 
for method in methods:
    full_results_path = PACKAGE_DIR / "results" / "mime-cropped" / method / "all"
    
    if full_results_path.exists():
        results = load_results_from_folder(full_results_path)
        # add method to results 
        for result in results:
            result["method"] = method
    
        all_results.extend(results)

# load REAL results 
real_results_path = PACKAGE_DIR / "results" / "mime-real-resized" / "zero-shot" / "none"
real_results = load_results_from_folder(real_results_path)
all_results.extend(real_results)

vlm_results_df = pd.DataFrame(all_results)

# organize results for paper 
model_order = [
    "Qwen2.5 VL (3B)",
    "Qwen2.5 VL (7B)",
    "Phi-3.5",
    "InternVL2.5 (8B)",
    "GPT-4o Mini",
    "Gemini 1.5 Flash",
]

sep = "|"

### mime-base vs real results (Table 1)
def print_table1(vlm_results_df, human_eval_mime_results, human_eval_real_results):
    method = "zero-shot"
    
    # columns: Model - mime-base mcq - mime-base ff - real mcq - real ff
    for model in model_order:
        mime_base_mcq = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "mcq") & (vlm_results_df["avatar"] == "man") & (vlm_results_df["background_config"] == "blank") & (vlm_results_df["angle"] == 0) & (vlm_results_df["method"] == method)]
        mime_base_ff = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "ff") & (vlm_results_df["avatar"] == "man") & (vlm_results_df["background_config"] == "blank") & (vlm_results_df["angle"] == 0) & (vlm_results_df["method"] == method)]
        real_mcq = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "mcq") & (vlm_results_df["avatar"] == "real") & (vlm_results_df["background_config"] == "real") & (vlm_results_df["angle"] == "real") & (vlm_results_df["method"] == method)]
        real_ff = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "ff") & (vlm_results_df["avatar"] == "real") & (vlm_results_df["background_config"] == "real") & (vlm_results_df["angle"] == "real") & (vlm_results_df["method"] == method)]
        
        print(f"{model}{sep}{mime_base_mcq['accuracy'].item():.1f}{sep}{mime_base_ff['accuracy'].item():.1f}{sep}{real_mcq['accuracy'].item():.1f}{sep}{real_ff['accuracy'].item():.1f}")
    
    human_mime_mcq = human_eval_mime_results['mcq']['man']['0']['blank']['accuracy']
    human_mime_ff = human_eval_mime_results['free_form']['man']['0']['blank']['accuracy']
    human_real_mcq = human_eval_real_results['mcq']['unknown']['unknown']['unknown']['accuracy']
    human_real_ff = human_eval_real_results['free_form']['unknown']['unknown']['unknown']['accuracy']
    print(f"Human{sep}{human_mime_mcq:.1f}{sep}{human_mime_ff:.1f}{sep}{human_real_mcq:.1f}{sep}{human_real_ff:.1f}")
    return 

### create figure 1 
def create_figure1(vlm_results_df, human_eval_mime_results, human_eval_real_results):
    """use matplotlib to create figure 1 that shows separate subplots for MCQ and FF with REAL and MIME bars"""    
    
    # get accuracies of models on real and mime 
    method = "zero-shot"
    vlm_results_df = vlm_results_df[vlm_results_df["method"] == method]
    
    # chart data 
    chart_data = {} 
    for model in model_order:
        real_results = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["avatar"] == "real") & (vlm_results_df["background_config"] == "real") & (vlm_results_df["angle"] == "real")]
        mime_results = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["avatar"] == "man") & (vlm_results_df["background_config"] == "blank") & (vlm_results_df["angle"] == 0)]
        
        chart_data[model] = {
            "real": {
                "mcq": real_results[real_results["test_format"] == "mcq"]["accuracy"].item(),
                "ff": real_results[real_results["test_format"] == "ff"]["accuracy"].item(),
            },
            "mime": {
                "mcq": mime_results[mime_results["test_format"] == "mcq"]["accuracy"].item(),
                "ff": mime_results[mime_results["test_format"] == "ff"]["accuracy"].item(),
            }
        }
        
    chart_data["Human"] = {
        "real": {
            "mcq": human_eval_real_results["mcq"]["unknown"]["unknown"]["unknown"]["accuracy"],
            "ff": human_eval_real_results["free_form"]["unknown"]["unknown"]["unknown"]["accuracy"],
        },
        "mime": {
            "mcq": human_eval_mime_results["mcq"]["man"]["0"]["blank"]["accuracy"],
            "ff": human_eval_mime_results["free_form"]["man"]["0"]["blank"]["accuracy"],
        }
    }
    
    # create separate subplots for MCQ and FF
    import matplotlib.pyplot as plt 
    
    # Set font properties for better appearance
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 14
    
    # Create figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)  # Increased height
    
    # Reverse model order and add Human
    models_with_human = (model_order + ["Human"])[::-1]
    
    # Set up y positions for grouped bars with smaller gaps
    n_models = len(models_with_human)
    bar_height = 0.3
    y_positions = np.arange(n_models) * 0.4  # Multiply by 0.8 to reduce gaps between models
    
    # Color-blind friendly colors (MIME dark, REAL light)
    real_color = '#1f77b4'   # Dark blue for REAL
    mime_color = '#ff7f0e'   # Dark orange for MIME
    
    # Helper function to format accuracy values
    def format_accuracy(value):
        if value == 100.0:
            return "100"
        else:
            return f"{value:.1f}"
    
    # Function to add bars and annotations
    def add_bars_and_text(ax, data_type, title, color_real, color_mime, show_yaxis_vals=False, show_legend=False):
        for i, model in enumerate(models_with_human):
            # Get data for this model
            real_value = chart_data[model]["real"][data_type]
            mime_value = chart_data[model]["mime"][data_type]
            
            # Create bars - REAL above MIME
            real_y_pos = y_positions[i] + bar_height/4
            mime_y_pos = y_positions[i] - bar_height/4
            
            # Draw bars
            ax.barh(real_y_pos, real_value, bar_height/2, 
                   color=color_real, alpha=0.8, label='REAL' if i == 0 and show_legend else "")
            ax.barh(mime_y_pos, mime_value, bar_height/2, 
                   color=color_mime, alpha=0.8, label='MIME' if i == 0 and show_legend else "")
            
            # Add text annotations at right end of bars
            # REAL value
            if real_value > 15:  # Inside bar if wide enough
                ax.text(real_value - 2, real_y_pos, f'{format_accuracy(real_value)}%', 
                       ha='right', va='center', fontsize=12, color='black')
            else:  # Outside bar
                ax.text(real_value + 1, real_y_pos, f'{format_accuracy(real_value)}%', 
                       ha='left', va='center', fontsize=12, color='black')
            
            # MIME value
            if mime_value > 15:  # Inside bar if wide enough
                ax.text(mime_value - 2, mime_y_pos, f'{format_accuracy(mime_value)}%', 
                       ha='right', va='center', fontsize=12, color='black')
            else:  # Outside bar
                ax.text(mime_value + 1, mime_y_pos, f'{format_accuracy(mime_value)}%', 
                       ha='left', va='center', fontsize=12, color='black')
        
        # Customize the subplot
        ax.set_xlim(0, 100)
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        # Remove top and bottom margins by setting tight y-limits
        y_min = y_positions[0] - bar_height/2 - 0.05  # Bottom of lowest bars with small padding
        y_max = y_positions[-1] + bar_height/2 + 0.05  # Top of highest bars with small padding
        ax.set_ylim(y_min, y_max)
        
        # Customize x-axis ticks
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(['0', '20', '40', '60', '80', '100'])
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add dotted line between Human (index 0) and the model above it (index 1)
        if len(models_with_human) > 1:
            # Adjust separation line for new y-positioning
            separation_y = (y_positions[0] + y_positions[1]) / 2
            ax.axhline(y=separation_y, color='gray', linestyle=':', linewidth=2, alpha=0.7)
            
    # Create MCQ subplot (with legend)
    add_bars_and_text(ax1, "mcq", "Multiple Choice (MC)", real_color, mime_color, show_yaxis_vals=True)
    
    # Create FF subplot (without legend)
    add_bars_and_text(ax2, "ff", "Free-form (FF)", real_color, mime_color)
    
    # Set y-axis labels only on left subplot with bold font
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(models_with_human, fontweight='bold')
    
    # Add shared x-axis label
    fig.text(x=0.55, y=0.1, s='Accuracy (%)', ha='center', va='center', fontweight='medium', fontsize=12)
    
    # Add custom legend to the right side of the x-axis title
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, fc=real_color, alpha=0.8, label='REAL'),
        plt.Rectangle((0,0), 1, 1, fc=mime_color, alpha=0.8, label='MIME'), 
    ]
    
    # Position legend to the right of the x-axis title
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.85, 0.05), 
               ncol=4, frameon=True)
        
    # Adjust layout and save with more bottom space
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Increased bottom margin significantly
    # save as pdf 
    plt.savefig(f'{PACKAGE_DIR}/results/figure1_separate_subplots.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    return

### mime-full results 
def print_table2(vlm_results_df, human_eval_mime_results):

    method = "zero-shot"

    configs = [
        ("man", "blank", 0),
        ("man", "aligned", 0),
        ("man", "misaligned", 0),
        ("spacesuit", "blank", 0),
        ("spacesuit", "aligned", 0), 
        ("spacesuit", "misaligned", 0),
    ]

    for model in model_order: 
        print(f"{model}", end=sep)
        for config in configs: 
            mcq = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "mcq") & (vlm_results_df["avatar"] == config[0]) & (vlm_results_df["background_config"] == config[1]) & (vlm_results_df["angle"] == config[2]) & (vlm_results_df["method"] == method)]
            ff = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "ff") & (vlm_results_df["avatar"] == config[0]) & (vlm_results_df["background_config"] == config[1]) & (vlm_results_df["angle"] == config[2]) & (vlm_results_df["method"] == method)]

            print(f"{mcq['accuracy'].item():.1f}{sep}{ff['accuracy'].item():.1f}", end=sep)
        print()
        
    print("Human", end=sep)
    for config in configs: 
        human_mcq = human_eval_mime_results['mcq'][config[0]][str(config[2])][config[1]]['accuracy']
        human_ff = human_eval_mime_results['free_form'][config[0]][str(config[2])][config[1]]['accuracy']
        
        print(f"{human_mcq:.1f}{sep}{human_ff:.1f}", end=sep)
    print()
    return 
        
### gender variation results 
def print_table3(vlm_results_df, human_eval_mime_results):
    methods = ["zero-shot", "cot", "few-shot"]
    method_print_mapping = {
        "zero-shot": "Zero-shot",
        "cot": "CoT",
        "few-shot": "Few-shot",
    }
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Method} & \multicolumn{3}{c}{MC} & \multicolumn{3}{c}{FF} \\")
    print(r" & & $\male$ & $\female$ & diff & $\male$ & $\female$ & diff \\")
    print(r"\midrule")
    for model in model_order:
        # Get all methods for this model that have results
        model_methods = [method for method in methods if not vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["method"] == method)].empty]
        n_methods = len(model_methods)
        for i, method in enumerate(model_methods):
            # MCQ
            mcq_man = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "mcq") & (vlm_results_df["avatar"] == "man") & (vlm_results_df["background_config"] == "blank") & (vlm_results_df["angle"] == 0) & (vlm_results_df["method"] == method)]["accuracy"].item()
            mcq_woman = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "mcq") & (vlm_results_df["avatar"] == "woman") & (vlm_results_df["background_config"] == "blank") & (vlm_results_df["angle"] == 0) & (vlm_results_df["method"] == method)]["accuracy"].item()
            mcq_diff = round(mcq_woman, 1) - round(mcq_man, 1)
            # FF
            ff_man = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "ff") & (vlm_results_df["avatar"] == "man") & (vlm_results_df["background_config"] == "blank") & (vlm_results_df["angle"] == 0) & (vlm_results_df["method"] == method)]["accuracy"].item()
            ff_woman = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == "ff") & (vlm_results_df["avatar"] == "woman") & (vlm_results_df["background_config"] == "blank") & (vlm_results_df["angle"] == 0) & (vlm_results_df["method"] == method)]["accuracy"].item()
            ff_diff = round(ff_woman, 1) - round(ff_man, 1)
            # Color diff
            def color_diff(val):
                if val > 0:
                    val = abs(val)
                    return f"\\textcolor{{darkgreen}}{{{val:.1f}}}"
                else:
                    val = abs(val)
                    return f"\\textcolor{{darkred}}{{{val:.1f}}}"
            mcq_diff_str = color_diff(mcq_diff)
            ff_diff_str = color_diff(ff_diff)
            # Multirow for model
            if i == 0:
                print(f"\\multirow{{{n_methods}}}{{*}}{{\\makecell[l]{{{model}}}}} & {method_print_mapping[method]} & {mcq_man:.1f} & {mcq_woman:.1f} & {mcq_diff_str} & {ff_man:.1f} & {ff_woman:.1f} & {ff_diff_str} \\\\")
            else:
                print(f" & {method_print_mapping[method]} & {mcq_man:.1f} & {mcq_woman:.1f} & {mcq_diff_str} & {ff_man:.1f} & {ff_woman:.1f} & {ff_diff_str} \\\\")
    # Human row
    mcq_man = human_eval_mime_results["mcq"]["man"]["0"]["blank"]["accuracy"]
    mcq_woman = human_eval_mime_results["mcq"]["woman"]["0"]["blank"]["accuracy"]
    mcq_diff = round(mcq_woman, 1) - round(mcq_man, 1)
    ff_man = human_eval_mime_results["free_form"]["man"]["0"]["blank"]["accuracy"]
    ff_woman = human_eval_mime_results["free_form"]["woman"]["0"]["blank"]["accuracy"]
    ff_diff = round(ff_woman, 1) - round(ff_man, 1)
    def color_diff(val):
        if val > 0:
            return f"\\textcolor{{darkgreen}}{{{val:.1f}}}"
        else:
            return f"\\textcolor{{darkred}}{{{val:.1f}}}"
    mcq_diff_str = color_diff(mcq_diff)
    ff_diff_str = color_diff(ff_diff)
    print(r"\arrayrulecolor{black} \midrule")
    print(f"Human & - & {mcq_man:.1f} & {mcq_woman:.1f} & {mcq_diff_str} & {ff_man:.1f} & {ff_woman:.1f} & {ff_diff_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

### angle variation results
def print_table4(vlm_results_df, human_eval_mime_results):
    logger.info("Printing angle variation results")
    
    method = "zero-shot"
    test_formats = ["mcq", "ff"]
    angles = [0, 90, 180, 270]
    avatar = "man"
    background_config = "blank"
    
    column_names = ["model", "test format", "0", "90", "180", "270", "avg", "std"]
    print(f"{sep.join(column_names)}")
    
    for model in model_order: 
        print(f"{model}", end=sep)
        for idx, test_format in enumerate(test_formats):
            if idx == 0:
                print(test_format, end=sep)
            else: 
                print(f"{sep}{test_format}", end=sep)
            angle_results = [] 
            for angle in angles:
                acc = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["test_format"] == test_format) & (vlm_results_df["angle"] == angle) & (vlm_results_df["method"] == method) & (vlm_results_df["avatar"] == avatar) & (vlm_results_df["background_config"] == background_config)]
                print(f"{acc['accuracy'].item():.1f}", end=sep)
                angle_results.append(acc['accuracy'].item())
            print(f"{np.mean(angle_results):.1f}{sep}{np.std(angle_results):.1f}")
        
    print(f"Human", end=sep)
    for idx, test_format in enumerate(["mcq", "free_form"]):
        if idx == 0:
            print(test_format, end=sep)
        else: 
            print(f"{sep}{test_format}", end=sep)
        angle_results = [] 
        for angle in angles:
            human_acc = human_eval_mime_results[test_format][avatar][str(angle)][background_config]['accuracy']
            print(f"{human_acc:.1f}", end=sep)
            angle_results.append(human_acc)
        print(f"{np.mean(angle_results):.1f}{sep}{np.std(angle_results):.1f}")
    return 

### full improvement results 
def print_table5(vlm_results_df, human_eval_mime_results):
    logger.info("Printing full improvement results")
    ### Note that the fine-tuning results are not included here. 
    
    methods = ["zero-shot", "cot", "few-shot"]
    test_formats = ["mcq", "ff"]
    configs = [
        ("man", "blank", 0),
        ("man", "aligned", 0),
        ("man", "misaligned", 0),
        ("spacesuit", "blank", 0),
        ("spacesuit", "aligned", 0),
        ("spacesuit", "misaligned", 0),
    ]
    
    config_names = [f"{config[0]} {config[1]} {test_format}" for config in configs for test_format in test_formats]
    column_names = ["model", "method", *config_names]
    print(f"{sep.join(column_names)}")
    
    for model in model_order:
        print(f"{model}", end=sep)
        for idx, method in enumerate(methods):
            if idx == 0:
                print(f"{method}", end=sep)
            else:
                print(sep, end=sep) 
            method_results = vlm_results_df[(vlm_results_df["model"] == model) & (vlm_results_df["method"] == method)]
            if method_results.empty:
                continue 
            
            for config in configs:
                for test_format in test_formats:
                    acc = method_results[(method_results["test_format"] == test_format) & (method_results["avatar"] == config[0]) & (method_results["background_config"] == config[1]) & (method_results["angle"] == config[2])]
                    print(f"{acc['accuracy'].item():.1f}", end=sep)
            print()
        
    print(f"Human{sep}-", end=sep)
    for config in configs:
        for test_format in ["mcq", "free_form"]:
            human_acc = human_eval_mime_results[test_format][config[0]][str(config[2])][config[1]]['accuracy']
            print(f"{human_acc:.1f}", end=sep)
    print()
    return 

human_eval_mime_fp = PACKAGE_DIR / "results" / "mime-cropped" / "human_eval.jsonl"
human_eval_real_fp = PACKAGE_DIR / "results" / "mime-real-resized" / "human_eval.jsonl"

human_eval_results_mime, overall_metrics_mime = calculate_accuracies_by_groups(
    human_eval_mime_fp, 
    similarity_threshold=0.5
) 

human_eval_results_real, overall_metrics_real = calculate_accuracies_by_groups(
    human_eval_real_fp, 
    similarity_threshold=0.5
) 

if args.table_num == 1:
    print_table1(vlm_results_df, human_eval_results_mime, human_eval_results_real)
    create_figure1(vlm_results_df, human_eval_results_mime, human_eval_results_real)

if args.table_num == 2:
    print_table2(vlm_results_df, human_eval_results_mime)

if args.table_num == 3:
    print_table3(vlm_results_df, human_eval_results_mime)

if args.table_num == 4:
    print_table4(vlm_results_df, human_eval_results_mime)

if args.table_num == 5:
    print_table5(vlm_results_df, human_eval_results_mime)


if args.table_num == 6:
    actor_results = human_eval_results_mime['actor_results']
    for actor in actor_results:
        print(f"{actor}")
        actor_results[actor]['mcq']['accuracy'] = actor_results[actor]['mcq']['correct']/actor_results[actor]['mcq']['total']
        actor_results[actor]['free_form']['accuracy'] = actor_results[actor]['free_form']['correct']/actor_results[actor]['free_form']['total']
        
        for metric in ["mcq", "free_form"]:
            print(f"{metric}: {actor_results[actor][metric]['correct']}/{actor_results[actor][metric]['total']} {actor_results[actor][metric]['accuracy']*100:.1f}%")
            
    # statistical significance test between mcq accuracies and free form accuracies between actors
    for actor in actor_results:
        for actor2 in actor_results:
            if actor == actor2:
                continue
            # calculate the p-value for the difference in accuracy between mcq and free form for each actor 
            # use t-test based on accuracy and total counts of correct and total 
            mcq1 = actor_results[actor]['mcq']
            ff1 = actor_results[actor]['free_form']
            mcq2 = actor_results[actor2]['mcq']
            ff2 = actor_results[actor2]['free_form']
            
            # Compute mean and standard error for MCQ
            mcq1_mean = mcq1['accuracy']
            mcq1_std = np.sqrt(mcq1['accuracy'] * (1 - mcq1['accuracy']))  # standard deviation
            mcq2_mean = mcq2['accuracy']
            mcq2_std = np.sqrt(mcq2['accuracy'] * (1 - mcq2['accuracy']))  # standard deviation
            
            # Perform t-test for MCQ
            t_stat_mcq, p_val_mcq = ttest_ind_from_stats(mcq1_mean, mcq1_std, mcq1['total'], mcq2_mean, mcq2_std, mcq2['total'])
            
            # Perform t-test for free-form
            ff1_mean = ff1['accuracy']
            ff1_std = np.sqrt(ff1['accuracy'] * (1 - ff1['accuracy']))  # standard deviation
            ff2_mean = ff2['accuracy']
            ff2_std = np.sqrt(ff2['accuracy'] * (1 - ff2['accuracy']))  # standard deviation
            
            t_stat_ff, p_val_ff = ttest_ind_from_stats(ff1_mean, ff1_std, ff1['total'], ff2_mean, ff2_std, ff2['total'])
            
            print(f"T-test between {actor} and {actor2}:")
            print(f"MCQ: t-stat = {t_stat_mcq:.4f}, p-value = {p_val_mcq:.10f}")
            print(f"Free-form: t-stat = {t_stat_ff:.4f}, p-value = {p_val_ff:.10f}")
            print(f"")