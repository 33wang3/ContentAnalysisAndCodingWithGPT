"""
   Run chatGPT to complete task assigned by the prompt
   for given dataset
   using OpenAI APIs
"""

#!/usr/bin/env python
# coding: utf-8


import argparse
from functools import partial
import time
from gpts import gpts
from collections import defaultdict

def get_fewshot_prompt(promptpath):
    if len(promptpath) == 0:
        return ''
    with open(f"./prompts/{promptpath}.txt", "r") as fin:
        prompt = fin.read() 
    return prompt


def strip_response(r):
    r = r.strip()
    return r


def run(out_df, task_col, idxs, gpts, output_prefix, promptpath=''):
    
    os.environ["OPENAI_API_KEY"] = "Some_API_key"
    fewshot_prompt = get_fewshot_prompt(promptpath)
    out_df = out_df.iloc[idxs]
    questions = out_df[task_col].tolist()
    prompts = [fewshot_prompt + question + "\n" for question in questions]
    infos = {}
    i_range = list(range(0, len(prompts), 50))
    for i in i_range:
        trial = 0
        while trial < 3:
            try:
                raw_responses = gpts(prompts[i:i+50])
                break
            except Exception as e:
                print(e)
                time.sleep(60)
            trial += 1
        for j in range(len(raw_responses)):
            info = {}
            input = prompts[i+j][len(fewshot_prompt):]
            answer = raw_responses[j]
            info.update({'prompt': fewshot_prompt, 'input': input, 'answer': answer})
            infos[idxs[i+j]] = info        
        # create a dictionary that returns 'NotFound' when a key is not found
        review_to_info = defaultdict(lambda: {'answer':'[]'})
        for info in infos.values():
            review_to_info[info['input'].strip()] = info
        if (i > 0 and i%200 == 0) or i == i_range[-1]:
            out_df['label'] = out_df[task_col].apply(lambda x: review_to_info[x.strip()]['answer'])
            outfilename = f"{output_prefix}_{promptpath}_{idxs[0]}-{idxs[i+len(raw_responses)-1]}.xlsx"
            out_df.to_excel(outfilename, index=False)
    return infos


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.01)

    args.add_argument('--task', type=str, required=True)
    args.add_argument('--task_split', type=str, default='train')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--evaluate', action='store_true')
    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')
    
    args.add_argument('--modelpath', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    args.add_argument('--peftpath', type=str, default='forestai/lora_hotpot_v2')
    args.add_argument('--promptpath', type=str, default='')

    args = args.parse_args()
    return args

def extract_brackets(s):
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1:
        return s[start:end+1]
    else:
        return s

# python generation_io.py --task hotpotqa --task_end_index 100 --backend llama --evaluate --task_split dev 
import os
import pandas as pd
if __name__ == '__main__':
    TOP_NUM = 10000
    # your input data set should be here!!
    in_df = pd.read_csv("infile.csv").head(TOP_NUM)

    # model_name = "gpt-4"
    model_name = "gpt-3.5-turbo-0125"
    model = partial(gpts, model=model_name, temperature=0.01)
    # prompt_file = "zero-shot_single_review-summary_v0"
    # prompt_file = "few-shot_single_review-summary_v1"
    prompt_file = "prompt_barriers"
    infos = run(in_df.copy(), 'Tweet', list(range(len(in_df['Tweet']))),
                model, output_prefix=f"Your_results-{TOP_NUM}",
                promptpath=prompt_file)


    ## prompt file: put prompt into it
    ## head : how many output
    ## in_df: import the file 
    ## infos=in_df: change to the colume name you want to analysis 
    ## output_prefix= name the output profile
