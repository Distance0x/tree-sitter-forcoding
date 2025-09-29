from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import math
from baselines.utils.run_baseline import get_roc_metrics, get_precision_recall_metrics, get_accurancy, run_baseline_threshold_experiment
import functools
import torch
from baselines.supervised import eval_supervised
from baselines.entropy import get_entropy
from baselines.rank import get_ranks, get_rank
from baselines.loss import get_ll, get_lls
import random
import re
import numpy as np
from identifier_tagging import get_identifier
import tokenize
from io import StringIO
import yapf
import scipy.stats
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from baselines.all_baselines import run_all_baselines
from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model
from baselines.utils.preprocessing import preprocess_and_save
import json
import argparse
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from baselines.sample_generate.generate import generate_data

# Tree-sitter based replacement utilities (local module)
from treesitter_replacement import (
    normalize_solution_preserve_leading,
    is_valid_ts,
    VariableRenameStyle,
)




def setup_args():
    """Setup and parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="CodeSearchNet")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--pct_words_masked', type=float, default=0.5)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_perturbation_list', type=str, default="50")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name',
                        type=str, default="Salesforce/codet5p-770m")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="test_ipynb")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str, default=None)
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--mask_temperature', type=float, default=1)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str,
                        default="~/.cache/huggingface/hub")
    parser.add_argument('--prompt_len', type=int, default=30)
    parser.add_argument('--generation_len', type=int, default=200)
    parser.add_argument('--min_words', type=int, default=55)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--baselines', type=str, default="LRR,DetectGPT,NPR")
    parser.add_argument('--perturb_type', type=str, default="random-insert-comment")
    parser.add_argument('--pct_identifiers_masked', type=float, default=0.75)
    parser.add_argument('--min_len', type=int, default=20)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--max_comment_num', type=int, default=10)
    parser.add_argument('--max_def_num', type=int, default=5)
    parser.add_argument('--cut_def', action='store_true')
    parser.add_argument('--max_todo_num', type=int, default=3)
    parser.add_argument('--gen_comment_mode', type=str, default='inline',
                        choices=['inline', 'before', 'after'])
    parser.add_argument('--gene_comment_temperature', type=float, default=0.7)
    parser.add_argument('--rename_params', action='store_true', help='Also rename function parameters when using Tree-sitter')
    parser.add_argument('--ts_underscore_prob', type=float, default=0.5, help='Probability for underscore insertion when using ts-replacement-underscore')
    parser.add_argument('--ts_underscore_repeat', type=int, default=1, help='Repeat count for underscore insertion when using ts-replacement-underscore')
    parser.add_argument('--ts_var_prob', type=float, default=1.0, help='Per-variable probability to apply rename in underscore mode')
    # Tree-sitter parentheses / deadcode augmentation arguments
    parser.add_argument('--ts_parentheses_layers', type=int, default=1, help='Number of nested parentheses layers to wrap certain expressions (ts-parentheses)')
    parser.add_argument('--ts_deadcode_max_per_func', type=int, default=1, help='Maximum deadcode blocks inserted per function (ts-deadcode)')
    parser.add_argument('--ts_deadcode_prob', type=float, default=0.3, help='Probability a function receives a deadcode insertion (ts-deadcode)')
    parser.add_argument('--ts_deadcode_seed', type=int, default=0, help='RNG seed for deadcode insertion reproducibility')

    args_dict = {
        # 'dataset': "TheVault",
        'dataset': "CodeSearchNet",
        # 'dataset_key': "CodeLlama-7b-hf-10000-tp0.2",
        # 'dataset_key': "Seed-Coder-8B-Instruct-100000-tp0.2",
        'pct_words_masked': 0.5,
        'pct_identifiers_masked': 0.75,
        'span_length': 2,
        'n_samples': 500,
        'n_perturbation_list': "50",
        'n_perturbation_rounds': 1,
        # 'base_model_name': "codellama/CodeLlama-7b-hf", # Make sure to use the same model as the one used for generating the samples
        # 'base_model_name': "ByteDance-Seed/Seed-Coder-8B-Instruct",
        'mask_filling_model_name': "Salesforce/codet5p-770m",
        'batch_size': 32,
        'chunk_size': 10,
        'n_similarity_samples': 20,
        'int8': False,
        'half': False,
        'base_half': False,
        'do_top_k': False,
        'top_k': 40,
        'do_top_p': False,
        'top_p': 0.96,
        'output_name': "test_ipynb",
        'openai_model': None,
        'openai_key': None,
        'DEVICE': 'cuda',
        'buffer_size': 1,
        'mask_top_p': 1.0,
        'mask_temperature': 1,
        'pre_perturb_pct': 0.0,
        'pre_perturb_span_length': 5,
        'random_fills': False,
        'random_fills_tokens': False,
        'cache_dir': "~/.cache/huggingface/hub",
        'prompt_len': 30,
        'generation_len': 200,
        'min_words': 55,
        'temperature': 1,
        'baselines': "LRR,DetectGPT,NPR",
        # if you want the performance of original DetectLLM-NPR and DetectGPT, use "random"
        'perturb_type': "random-comment-out-lines",
        # 'min_len': 20,
        'max_len': 128,
        'max_comment_num': 10,
        'max_def_num': 5,
        'cut_def': False,
        'max_todo_num': 3,
        'gen_comment_mode': 'inline',
        'gene_comment_temperature': 0.7
    }

    parsed = parser.parse_args()
    argv = sys.argv[1:]
    provided = set()
    for token in argv:
        if token.startswith('--'):
            key = token[2:].split('=')[0]
            provided.add(key.replace('-', '_'))

    merged = {}
    for action in parser._actions:
        dest = action.dest
        if dest == 'help':
            continue
        if dest in provided:
            merged[dest] = getattr(parsed, dest)
        elif dest in args_dict:
            merged[dest] = args_dict[dest]
        else:
            merged[dest] = getattr(parsed, dest)

    return argparse.Namespace(**merged)

# 控制数据流的，添加功能只要加一个T-sitter的continue就是
def generate_data(dataset, key, max_num=200, min_len=0, max_len=128, max_comment_num=10, max_def_num=5, cut_def=False, max_todo_num=3, ts_filter: bool = False):

    path = f'../code-generation/output/{dataset}/{key}/outputs.txt'

    logger.info(f'Loading data from {path}')
    import json
    all_originals = []
    all_samples = []  # machine generated
    all_prompts = []

    max_def_num_count = 0
    min_len_count = 0
    max_comment_num_count = 0
    function_comment_num_count = 0
    max_todo_num_count = 0
    ts_filter_count = 0

    with open(path, 'r') as f:
        for line in tqdm(f, ncols=70):
            line = line.strip()
            if line == '':
                continue
            line = json.loads(line)

            # cut out the 'def' part after the first generation
            if cut_def:
                line['output'] = line['output'].split('def')[0]
                line['solution'] = line['solution'].split('def')[0]

            # I don't like there to have too many 'def' in the code
            # ~100/100000 examples have more than 3 'def'
            if line['solution'].count('def') > max_def_num or line['output'].count('def') > max_def_num:
                max_def_num_count += 1
                continue

            # avoid examples that are too short (less than min_len words)
            # around 2000/100000 examples have around 55 words
            if len(line['solution'].split()) < min_len or len(line['output'].split()) < min_len:
                min_len_count += 1
                continue

            # if the are too many comments, skip
            def count_comment(text):
                return text.count('#')

            if count_comment(line['solution']) > max_comment_num or count_comment(line['output']) > max_comment_num:
                max_comment_num_count += 1
                continue

            # if there are too many TODOs, skip
            def count_todo_comment(text):
                return text.count('# TODO') + text.count('# todo')

            if count_todo_comment(line['solution']) > max_todo_num or count_todo_comment(line['output']) > max_todo_num:
                max_todo_num_count += 1
                continue

            # the number of text.count("'''") and text.count('"""') should be <1
            if line['solution'].count("'''") > 0 or line['solution'].count('"""') > 0 or line['output'].count("'''") > 0 or line['output'].count('"""') > 0:
                function_comment_num_count += 1
                continue

            # Optional: Tree-sitter validation only when TS filter is enabled
            if ts_filter:
                try:
                    p = line.get('prompt', '')
                    out_combined = normalize_solution_preserve_leading(f"{p}\n{line['output']}")
                    sol_combined = normalize_solution_preserve_leading(f"{p}\n{line['solution']}")
                    if not (is_valid_ts(out_combined) and is_valid_ts(sol_combined)):
                        continue
                except Exception:
                    # If any unexpected error in TS path, skip this sample
                    ts_filter_count += 1
                    continue

            # cut to 128 tokens
            all_originals.append(
                ' '.join(line['solution'].split(' ')[:max_len]))
            all_samples.append(' '.join(line['output'].split(' ')[:max_len]))
            all_prompts.append(line.get('prompt', ''))

    logger.info(
        f'{max_def_num_count} examples have more than {max_def_num} "def"')
    logger.info(f'{min_len_count} examples have less than {min_len} words')
    logger.info(
        f'{max_comment_num_count} examples have more than {max_comment_num} comments')
    logger.info(
        f'{max_todo_num_count} examples have more than {max_todo_num} TODOs')
    logger.info(
        f'{function_comment_num_count} examples have more than 1 function comment')
    logger.info(f'{ts_filter_count} examples filtered by Tree-sitter')
    logger.info(
        f'Loaded {len(all_originals)} examples after filtering, and will return {min(max_num, len(all_originals))} examples')

    # statistical analysis
    # import random
    # random.seed(42)
    # random.shuffle(all_originals)
    # random.shuffle(all_samples)

    data = {
        "prompt": all_prompts[:max_num],
        "original": all_originals[:max_num],
        "sampled": all_samples[:max_num]
    }

    return data


pattern = re.compile(r"<extra_id_\d+>")
pattern_with_space = re.compile(r" <extra_id_\d+> ")


def remove_mask_space(text):
    # find all the mask positions " <extra_id_\d+> ", and remove the space before and after the mask
    matches = pattern_with_space.findall(text)
    for match in matches:
        text = text.replace(match, match.strip())
    return text


def tokenize_and_mask_identifiers(text, args, span_length, pct, ceil_pct=False, buffer_size=1):

    varnames, pos = get_identifier(text, 'python')

    mask_string = ' <<<mask>>> '
    sampled = random.sample(varnames, int(len(varnames)*1))
    logger.info(f"Sampled {len(sampled)} identifiers to mask: {sampled}")

    # Split the text into lines
    lines = text.split('\n')

    # replacements will change the line length, so we need to start from the end
    pos.sort(key=lambda pos: (-pos[0][0], -pos[0][1]))

    # Process each position
    for start, end in pos:
        # Extract line number and pos in line
        line_number, start_pos = start
        _, end_pos = end

        # mask the identifier if it is in the sampled list
        if lines[line_number][start_pos:end_pos] in sampled:
            # Replace the identified section in the line
            lines[line_number] = lines[line_number][:start_pos] + \
                mask_string + lines[line_number][end_pos:]

    # Join the lines back together
    masked_text = '\n'.join(lines)
    # logger.info(f'masked_text: \n{masked_text}')

    tokens = masked_text.split(' ')
    # logger.info(f'tokens: \n{tokens}')

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        # logger.info(f'idx: {idx}, token: {token}')
        if token == mask_string.strip():
            # logger.info(f'filling in {token} with <extra_id_{num_filled}>')
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1

    # before removing the space before and after the mask
    text = ' '.join(tokens)

    text = remove_mask_space(text)
    # logger.info(f'text: \n{text}')
    return text


def tokenize_and_mask(text, args, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    # count the number of masks in each text with the pattern "<extra_id_\d+>"
    pattern = re.compile(r"<extra_id_\d+>")
    n_expected = [len(pattern.findall(x)) for x in texts]
    return n_expected


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, model_config, args):
    n_expected = count_masks(texts)
    stop_id = model_config['mask_tokenizer'].encode(
        f"<extra_id_{max(n_expected)}>")[0]
    tokens = model_config['mask_tokenizer'](
        texts, return_tensors="pt", padding=True).to(args.DEVICE)
    # tokens = model_config['mask_tokenizer'](texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.DEVICE)
    outputs = model_config['mask_model'].generate(**tokens, max_length=512, do_sample=True, top_p=args.mask_top_p,
                                                  num_return_sequences=1, eos_token_id=stop_id, temperature=args.mask_temperature)
    return model_config['mask_tokenizer'].batch_decode(outputs, skip_special_tokens=False)


def apply_extracted_fills(masked_texts, extracted_fills):
    n_expected = count_masks(masked_texts)
    texts = []
    # logger.info(f"n_expected: {n_expected}")

    for idx, (text, fills, n) in enumerate(zip(masked_texts, extracted_fills, n_expected)):
        if len(fills) < n:
            texts.append('')
        else:
            for fill_idx in range(n):
                text = text.replace(f"<extra_id_{fill_idx}>", fills[fill_idx])
            texts.append(text)

    # logger.info(f"texts: {texts}")
    return texts


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

# 加random_comments（比例） 参考之前的方法 （一行/两行/...)  / 插入dead code
def perturb_texts_(texts, args,  model_config, ceil_pct=False):
    span_length = args.span_length 
    pct = args.pct_words_masked
    lambda_poisson = args.span_length
    # 这部分明天再改改 封装成一个函数好一点
    if _is_ts_mode(args.perturb_type):
        # Generalized Tree-sitter based structural perturbations
        from treesitter_replacement import (
            normalize_solution_preserve_leading as _norm,
            _apply_renaming_ts as _apply_ts,
            _apply_parentheses_ts as _apply_paren,
            _apply_deadcode_ts as _apply_dead,
            remove_prompt_from_code as _rm_prompt,
            RenameMode,
        )

        pt = args.perturb_type.lower()
        tokens = pt.split('-')[1:]  # drop leading 'ts'
        # Capability flags
        do_replacement = 'replacement' in tokens
        underscore_mode = 'underscore' in tokens
        do_parentheses = 'parentheses' in tokens
        do_deadcode = 'deadcode' in tokens

        # Style for replacement (default capitalize)
        if do_replacement:
            try:
                style = _map_ts_style(pt)
            except Exception:
                style = VariableRenameStyle.CAPITALIZE
        else:
            style = None

        prompts_seq = model_config.get('ts_prompts_seq', [])
        idx_ptr = model_config.get('ts_prompt_idx', 0)
        outs = []

        for t in texts:
            p = prompts_seq[idx_ptr] if idx_ptr < len(prompts_seq) else ""
            idx_ptr += 1
            combined = _norm(f"{p}\n{t}")
            transformed = combined
            try:
                # 1. Variable renaming / underscore mode
                if do_replacement:
                    if underscore_mode:
                        transformed, _ = _apply_ts(
                            transformed,
                            style,
                            rename_params=getattr(args, 'rename_params', True),
                            mode=RenameMode.UNDERSCORE,
                            underscore_prob=getattr(args, 'ts_underscore_prob', 0.5),
                            underscore_repeat=getattr(args, 'ts_underscore_repeat', 1),
                            rng=None,
                            var_prob=getattr(args, 'ts_var_prob', 1.0),
                        )
                    else:
                        transformed, _ = _apply_ts(
                            transformed,
                            style,
                            rename_params=getattr(args, 'rename_params', True),
                        )
                # 2. Parentheses wrapping
                if do_parentheses:
                    try:
                        transformed = _apply_paren(
                            transformed,
                            layers=getattr(args, 'ts_parentheses_layers', 1),
                        )
                    except Exception:
                        pass
                # 3. Dead code insertion
                if do_deadcode:
                    try:
                        transformed = _apply_dead(
                            transformed,
                            max_per_func=getattr(args, 'ts_deadcode_max_per_func', 1),
                            prob=getattr(args, 'ts_deadcode_prob', 0.3),
                            seed=getattr(args, 'ts_deadcode_seed', 0),
                        )
                    except Exception:
                        pass
                final = _rm_prompt(transformed, _norm(p))
            except Exception:
                final = t
            outs.append(final)
        model_config['ts_prompt_idx'] = idx_ptr
        return outs
    elif args.perturb_type == 'random':
        masked_texts = [tokenize_and_mask(
            x, args, span_length, pct, ceil_pct) for x in texts]
    elif args.perturb_type == 'identifier-masking':
        masked_texts = [tokenize_and_mask_identifiers(
            x, args, span_length, pct, ceil_pct) for x in texts]
    elif args.perturb_type == 'random-line-shuffle':
        perturbed_texts = [random_line_shuffle(x, pct) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-newline':
        perturbed_texts = [random_insert_newline(
            x, pct, lambda_poisson) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-space':
        perturbed_texts = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-space-newline':
        perturbed_texts = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        perturbed_texts = [random_insert_newline(
            x, pct, lambda_poisson) for x in perturbed_texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-space+newline':
        perturbed_texts_part1 = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        perturbed_texts_part2 = [random_insert_newline(
            x, pct, lambda_poisson) for x in texts]
        total_num = len(perturbed_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        perturbed_texts_part1 = perturbed_texts_part1[:n1]
        perturbed_texts_part2 = perturbed_texts_part2[:n2]
        return perturbed_texts_part1 + perturbed_texts_part2
    elif args.perturb_type == 'random-insert-comment':
        perturbed_texts = [random_insert_comment(
            x, pct, lambda_poisson) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-comment-out-lines':
        perturbed_texts = [random_comment_out_lines(
            x, pct) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-empty-comment-lines':
        perturbed_texts = [random_insert_empty_comment_lines(
            x, pct) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-empty-triple-quote':
        perturbed_texts = [random_insert_empty_triple_quote_lines(
            x, pct) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-empty-triple-quote-without-newline':
        perturbed_texts = [random_insert_empty_triple_quote_lines_without_newline(
            x, pct) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'space+comment':
        perturbed_texts_part1 = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        perturbed_texts_part2 = [random_insert_comment(
            x, pct, lambda_poisson) for x in texts]
        total_num = len(perturbed_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        perturbed_texts_part1 = perturbed_texts_part1[:n1]
        perturbed_texts_part2 = perturbed_texts_part2[:n2]
        return perturbed_texts_part1 + perturbed_texts_part2
    elif args.perturb_type == 'newline+comment':
        perturbed_texts_part1 = [random_insert_newline(
            x, pct, lambda_poisson) for x in texts]
        perturbed_texts_part2 = [random_insert_comment(
            x, pct, lambda_poisson) for x in texts]
        total_num = len(perturbed_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        perturbed_texts_part1 = perturbed_texts_part1[:n1]
        perturbed_texts_part2 = perturbed_texts_part2[:n2]
        return perturbed_texts_part1 + perturbed_texts_part2
    elif args.perturb_type == 'space+newline+comment':
        perturbed_texts_part1 = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        perturbed_texts_part2 = [random_insert_newline(
            x, pct, lambda_poisson) for x in texts]
        perturbed_texts_part3 = [random_insert_comment(
            x, pct, lambda_poisson) for x in texts]
        total_num = len(perturbed_texts_part1)
        n1 = total_num // 3
        n2 = (total_num - n1) // 2
        n3 = total_num - n1 - n2
        perturbed_texts_part1 = perturbed_texts_part1[:n1]
        perturbed_texts_part2 = perturbed_texts_part2[:n2]
        perturbed_texts_part3 = perturbed_texts_part3[:n3]
        return perturbed_texts_part1 + perturbed_texts_part2 + perturbed_texts_part3
    elif args.perturb_type == 'gen-comment':
        mode = getattr(args, 'gen_comment_mode', 'inline')
        perturbed_texts = [random_generate_line_comment(
            x, pct, mode=mode, model_config=model_config, args=args) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-line-format':
        perturbed_texts = [_perturb_one_with_format_strategy(
            x, pct, mode='format') for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-line-unformat':
        perturbed_texts = [_perturb_one_with_format_strategy(
            x, pct, mode='unformat') for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-line-unformat+format':
        perturbed_texts_part1 = [_perturb_one_with_format_strategy(
            x, pct, mode='unformat') for x in texts]
        perturbed_texts_part2 = [_perturb_one_with_format_strategy(
            x, pct, mode='format') for x in texts]
        total_num = len(perturbed_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        perturbed_texts_part1 = perturbed_texts_part1[:n1]
        perturbed_texts_part2 = perturbed_texts_part2[:n2]
        return perturbed_texts_part1 + perturbed_texts_part2
    elif args.perturb_type == 'random-insert-noise-chars':
        perturbed_texts = [random_insert_noise_chars(
            x, pct, lambda_poisson) for x in texts]
        return perturbed_texts
    else:
        raise ValueError(f'Unknown perturb_type: {args.perturb_type}')

    raw_fills = replace_masks(masked_texts, model_config, args)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(
            f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(
            x, args, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts, model_config, args)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(
            masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

        # If it fails for more than 50 texts, then we use the original texts as perturbed texts and inform the user with warning
        if attempts > 50:
            logger.warning(
                f'WARNING: {len(idxs)} texts have no fills. Using the original texts as perturbed texts.')
            for idx in idxs:
                perturbed_texts[idx] = texts[idx]

    logger.info(f'texts: {texts[0]}')
    logger.info(f'perturbed_texts: {perturbed_texts[0]}')

    return perturbed_texts


def perturb_texts(texts, args,  model_config,  ceil_pct=False):

    def perturb_texts_once(texts, args,  model_config,  ceil_pct=False):

        chunk_size = args.chunk_size
        if '11b' in args.mask_filling_model_name:
            chunk_size //= 2

        outputs = []
        for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
            outputs.extend(perturb_texts_(
                texts[i:i + chunk_size], args, model_config, ceil_pct=ceil_pct))

        return outputs

    for i in range(args.n_perturbation_rounds):
        texts = perturb_texts_once(
            texts, args, model_config, ceil_pct=ceil_pct)

    return texts
# 改成注释的方法


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def random_line_shuffle(text, pct=0.3):
    '''
    randomly exchange the order of two adjacent lines for pct of the lines, except for the first and last line
    '''
    lines = text.split('\n')
    n_lines = len(lines)
    n_shuffled = int(n_lines * pct)
    shuffled_idxs = np.random.choice(n_lines, n_shuffled, replace=False)
    for idx in shuffled_idxs:
        if idx == n_lines - 1 or idx == 0:
            continue
        lines[idx], lines[idx+1] = lines[idx+1], lines[idx]
    return '\n'.join(lines)


def random_insert_newline(text, pct=0.3, mean=1):
    '''
    randomly insert a newline for pct of the lines
    '''
    lines = text.split('\n')
    n_lines = len(lines)
    n_inserted = int(n_lines * pct)
    inserted_idxs = np.random.choice(n_lines, n_inserted, replace=False)
    for idx in inserted_idxs:
        n_newlines = 1
        # n_newlines = scipy.stats.poisson.rvs(mean) + 1
        lines[idx] = lines[idx] + '\n'*n_newlines
    return '\n'.join(lines)


def random_insert_space(text, pct=0.3, mean=1):
    '''
    randomly insert a space for pct of the lines
    '''
    tokens = text.split(' ')
    n_tokens = len(tokens)
    n_inserted = int(n_tokens * pct)
    inserted_idxs = np.random.choice(n_tokens, n_inserted, replace=False)
    for idx in inserted_idxs:
        n_spaces = scipy.stats.poisson.rvs(mean) + 1
        # n_spaces = 1
        tokens[idx] = tokens[idx] + ' '*n_spaces
    return ' '.join(tokens)


# 按行随机插入注释
# 对每一行，在该行末尾追加 k 个空注释标记（" #"）
def random_insert_comment(text, pct=0.3, mean=1):
    lines = text.split('\n')
    n_lines = len(lines)

    if n_lines == 0:
        logger.warning(f"No lines to insert comment")
        return text
    n_inserted = int(n_lines * pct)
    inserted_idxs = np.random.choice(n_lines, n_inserted, replace=False)
    for idx in inserted_idxs:
        k = scipy.stats.poisson.rvs(mean) + 1
        suffix = ''.join([' #' for _ in range(k)])
        lines[idx] = lines[idx] + suffix
    return '\n'.join(lines)

def random_insert_noise_chars(text, pct=0.3, mean=1):
    """
    在随机若干行行尾插入由候选符号组成的短串。
    候选符号：:, ), ], [, (, ,, ", ', {, }, .
    pct: 选取的行比例
    mean: 控制符号串长度 (Poisson(mean)+1)
    """
    lines = text.split('\n')
    n_lines = len(lines)
    if n_lines == 0:
        return text
    n_pick = int(n_lines * pct)
    if n_pick <= 0:
        n_pick = 1
    idxs = np.random.choice(n_lines, n_pick, replace=False)
    candidates = [':', ')', ']', '[', '(', ',', '"', "'", '{', '}', '.']
    for i in idxs:
        k = scipy.stats.poisson.rvs(mean) + 1
        noise = ''.join(random.choice(candidates) for _ in range(k))
        # 在行尾追加，前面加一个空格以减少破坏语法概率
        lines[i] = lines[i] + ' ' + noise
    return '\n'.join(lines)

def random_comment_out_lines(text, pct=0.3):
    lines = text.split('\n')
    n_lines = len(lines)
    if n_lines == 0:
        return text
    n_to_comment = int(n_lines * pct)
    if n_to_comment <= 0:
        n_to_comment = 1
    idxs = np.random.choice(n_lines, n_to_comment, replace=False)
    for idx in idxs:
        line = lines[idx]
        j = 0
        while j < len(line) and line[j] in (' ', '\t'):
            j += 1
        if j < len(line) and line[j:j+1] == '#':
            continue
        lines[idx] = line[:j] + '# ' + line[j:]
    return '\n'.join(lines)


def random_insert_empty_comment_lines(text, pct=0.3):
    lines = text.split('\n')
    n_lines = len(lines)
    if n_lines == 0:
        return text
    n_inserted = int(n_lines * pct)
    if n_inserted <= 0:
        n_inserted = 1
    # 允许在行间或末尾插入，位置范围 [0, n_lines]
    idxs = np.random.choice(n_lines + 1, n_inserted, replace=False)
    # 逆序插入以避免位置偏移
    for idx in sorted(idxs, reverse=True):
        indent = ''
        ref_line = None
        if idx < n_lines:
            ref_line = lines[idx]
        elif n_lines > 0:
            ref_line = lines[-1]
        if ref_line is not None:
            j = 0
            while j < len(ref_line) and ref_line[j] in (' ', '\t'):
                j += 1
            indent = ref_line[:j]
        lines.insert(idx, indent + '#')
    return '\n'.join(lines)


def random_insert_empty_triple_quote_lines(text, pct=0.3):
    """
    随机插入一个空的三引号字符串块（空行文档字符串样式）。
    示例（转义显示）： '\'\'\'\\n\'\'\'' 或 '"\"\"\"\\n\"\"\"'.

    “行间插入”指在两行之间插入新的独立行。本实现仅在“安全位置”插入：
    - 紧跟在以冒号结尾的语句之后（如 if/for/while/def/class 等），并使用子缩进，保证语法有效；
    - 空行或注释行的邻接处；
    - 文件末尾（EOF）。
    这些策略尽量避免破坏原有代码结构与缩进需求。
    """
    lines = text.split('\n')
    n_lines = len(lines)
    if n_lines == 0:
        return text
    # 计算“安全插入点”（位置 i 表示在 lines[i-1] 与 lines[i] 之间；i==0 表示文件头，i==n_lines 表示文件末尾）
    safe_positions = []  # 列表元素为 (pos, indent)
    for i in range(n_lines + 1):
        prev_line = lines[i - 1] if i - 1 >= 0 and n_lines > 0 else ''
        next_line = lines[i] if i < n_lines else ''
        prev_stripped = prev_line.strip()
        next_stripped = next_line.strip()

        # 情况1：前一行是 block header（以 ':' 结尾），使用子缩进
        if prev_stripped.endswith(':'):
            indent = _child_indent(_leading_indent(prev_line))
            safe_positions.append((i, indent))
            continue

        # 情况2：在空行或注释行的邻接处（保持相邻行的缩进）
        if next_stripped == '' or next_line.lstrip().startswith('#'):
            indent = _leading_indent(next_line) if next_line != '' else _leading_indent(prev_line)
            safe_positions.append((i, indent))
            continue
        if prev_stripped == '' or prev_line.lstrip().startswith('#'):
            indent = _leading_indent(next_line) if next_line != '' else _leading_indent(prev_line)
            safe_positions.append((i, indent))
            continue

        # 情况3：文件末尾（EOF），使用上一行的缩进
        if i == n_lines:
            indent = _leading_indent(prev_line)
            safe_positions.append((i, indent))

    if not safe_positions:
        return text

    n_inserted = int(n_lines * pct)
    if n_inserted <= 0:
        n_inserted = 1
    n_inserted = min(n_inserted, len(safe_positions))

    pick_idxs = np.random.choice(len(safe_positions), n_inserted, replace=False)
    # 逆序按位置插入以避免偏移
    inserts = sorted((safe_positions[j] for j in pick_idxs), key=lambda x: x[0], reverse=True)
    for pos, indent in inserts:
        # 随机选择三引号类型
        if random.random() < 0.5:
            openq = "'" * 3
            closeq = "'" * 3
        else:
            openq = '"' * 3
            closeq = '"' * 3
        # 插入两行：先插入关闭行，再插入打开行，保证最终顺序为 open -> close
        lines.insert(pos, indent + closeq)
        lines.insert(pos, indent + openq)
    return '\n'.join(lines) 


def random_insert_empty_triple_quote_lines_without_newline(text, pct=0.3):
    """
    随机插入一个单行三引号空字符串（中间仅一个空格）。
    例如: "''' '''" 或 '""" """'。
    """
    lines = text.split('\n')
    n_lines = len(lines)
    if n_lines == 0:
        return text
    # 计算“安全插入点”
    safe_positions = []
    for i in range(n_lines + 1):
        prev_line = lines[i - 1] if i - 1 >= 0 and n_lines > 0 else ''
        next_line = lines[i] if i < n_lines else ''
        prev_stripped = prev_line.strip()
        next_stripped = next_line.strip()
        if prev_stripped.endswith(':'):
            indent = _child_indent(_leading_indent(prev_line))
            safe_positions.append((i, indent))
            continue
        if next_stripped == '' or next_line.lstrip().startswith('#'):
            indent = _leading_indent(next_line) if next_line != '' else _leading_indent(prev_line)
            safe_positions.append((i, indent))
            continue
        if prev_stripped == '' or prev_line.lstrip().startswith('#'):
            indent = _leading_indent(next_line) if next_line != '' else _leading_indent(prev_line)
            safe_positions.append((i, indent))
            continue
        if i == n_lines:
            indent = _leading_indent(prev_line)
            safe_positions.append((i, indent))

    if not safe_positions:
        return text

    n_inserted = int(n_lines * pct)
    if n_inserted <= 0:
        n_inserted = 1
    n_inserted = min(n_inserted, len(safe_positions))

    pick_idxs = np.random.choice(len(safe_positions), n_inserted, replace=False)
    inserts = sorted((safe_positions[j] for j in pick_idxs), key=lambda x: x[0], reverse=True)
    for pos, indent in inserts:
        # 单行三引号空串（仅一个空格）
        q = "'" if random.random() < 0.5 else '"'
        token = (q * 3) + ' ' + (q * 3)
        lines.insert(pos, indent + token)
    return '\n'.join(lines)

def _leading_indent(s):
    j = 0
    while j < len(s) and s[j] in (' ', '\t'):
        j += 1
    return s[:j]


def _child_indent(indent):
    return indent + ('\t' if '\t' in indent else '    ')


def _shorten_comment(text: str) -> str:
    s = text.strip()
    s = s.replace('\r', '')
    if '\n' in s:
        s = s.split('\n', 1)[0]
    # remove leading comment markers if any
    if s.startswith('#'):
        s = s.lstrip('#').strip()
    return s


def random_generate_line_comment(text, pct=0.3, mode='inline', model_config=None, args=None):
    lines = text.split('\n')
    n_lines = len(lines)
    if n_lines == 0:
        return text

    # lazy-load base model if needed
    if (model_config is not None) and (('base_model' not in model_config) or ('base_tokenizer' not in model_config)):
        from baselines.utils.loadmodel import load_base_model_and_tokenizer
        model_config = load_base_model_and_tokenizer(args, model_config)

    base_model = model_config.get('base_model', None) if model_config else None
    base_tokenizer = model_config.get('base_tokenizer', None) if model_config else None
    if base_model is None or base_tokenizer is None:
        return text

    n_pick = int(n_lines * pct)
    if n_pick <= 0:
        n_pick = 1
    idxs = np.random.choice(n_lines, n_pick, replace=False)

    prompts = []
    meta = []
    for idx in idxs:
        code_line = lines[idx].strip('\n')
        # keep very short/blank lines still, but prompt will be generic
        prompt = (
            "You are a helpful coding assistant. Write a concise Python inline comment that explains the following code line. "
            "Return only the comment text without leading '#'.\n"
            f"Line: {code_line}\n"
            "Comment:"
        )
        prompts.append(prompt)
        meta.append(idx)

    if len(prompts) == 0:
        return text

    inputs = base_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    # remove token_type_ids if present to avoid models that don't accept it
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(args.DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        gen_out = base_model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=True,
            top_p=getattr(args, 'top_p', 0.96),
            temperature=getattr(args, 'gene_comment_temperature', 0.7),
            pad_token_id=base_tokenizer.eos_token_id,
            eos_token_id=base_tokenizer.eos_token_id
        )
    decoded = base_tokenizer.batch_decode(gen_out, skip_special_tokens=True)

    # extract generated tail after the prompt roughly
    gen_comments = []
    for p, out in zip(prompts, decoded):
        # take substring after last occurrence of prompt anchor 'Comment:'
        pos = out.rfind('Comment:')
        tail = out[pos + len('Comment:'):] if pos != -1 else out
        gen_comments.append(_shorten_comment(tail))

    # apply insertions
    for idx, comment in zip(meta, gen_comments):
        indent = _leading_indent(lines[idx])
        if mode == 'inline':
            lines[idx] = lines[idx] + ' # ' + comment
        elif mode == 'before':
            lines.insert(idx, f"{indent}# {comment}")
        else:  # 'after'
            lines.insert(idx + 1, f"{indent}# {comment}")

    return '\n'.join(lines)



    lines = text.split('\n')
    n_lines = len(lines)
    if n_lines == 0:
        return text
    eff_pct = min(max(pct, 0.0), 0.1)
    n_inserted = int(n_lines * eff_pct)
    if n_inserted <= 0:
        n_inserted = 1
    assign_pat = re.compile(r"^\s*([A-Za-z_]\w*)\s*=")
    idxs = np.random.choice(n_lines + 1, n_inserted, replace=False)
    for idx in sorted(idxs, reverse=True):
        ref_line = lines[idx-1] if idx-1 >= 0 else (lines[0] if n_lines > 0 else '')
        indent = _leading_indent(ref_line)
        var = None
        for j in range(max(0, idx-10), min(n_lines, idx+1)):
            m = assign_pat.match(lines[j])
            if m:
                var = m.group(1)
        if var is None:
            var = f"_dcv_{random.randint(0, 10**9)}"
            init = [f"{indent}{var} = 0"]
        else:
            init = []
        delta = random.randint(10, 100)
        if random.random() < 0.5:
            ops = [
                f"{indent}{var} = {var} + {delta}",
                f"{indent}{var} = {var} - {delta}",
            ]
        else:
            ops = [
                f"{indent}{var} = {var} - {delta}",
                f"{indent}{var} = {var} + {delta}",
            ]
        block = init + ops
        for b in reversed(block):
            lines.insert(idx, b)
    return '\n'.join(lines)

# tree-sitter相关的函数
def _is_ts_mode(perturb_type: str) -> bool:
    """Return True when using any Tree-sitter based perturbation (ts-*)."""
    try:
        return isinstance(perturb_type, str) and perturb_type.startswith("ts-")
    except Exception:
        return False


def _map_ts_style(perturb_type: str) -> "VariableRenameStyle":
    """Map perturb_type suffix or tokens to VariableRenameStyle."""
    try:
        # Ensure VariableRenameStyle is available even if import fails above
        from treesitter_replacement import VariableRenameStyle as _VRS
    except Exception:
        class _VRS:
            CAPITALIZE = 'capitalize'
            CAMEL = 'camel'
            SNAKE = 'snake'
    VRS = globals().get('VariableRenameStyle', locals().get('_VRS'))
    pt = (perturb_type or '').lower()
    if 'camel' in pt:
        return VRS.CAMEL
    if 'snake' in pt:
        return VRS.SNAKE
    return VRS.CAPITALIZE


# python格式化器
class _PythonFormatterSingle:
    def __init__(self):
        self.control_structure_patterns = {
            r"for\s+#\s*TODO:\s*Your\s+code\s+here:": "for _tempmask_ in [None]:",
            r"if\s+#\s*TODO:\s*Your\s+code\s+here:": "if [TEMPMASK]:",
            r"while\s+#\s*TODO:\s*Your\s+code\s+here:": "while [TEMPMASK]:",
        }
        self.reverse_control_structure_patterns = {
            "for _tempmask_ in [None]:": "for # TODO: Your code here:",
            "if [TEMPMASK]:": "if # TODO: Your code here:",
            "while [TEMPMASK]:": "while # TODO: Your code here:",
        }
        self.yapf_style = {
            "based_on_style": "pep8",
            "spaces_before_comment": 2,
            "split_before_logical_operator": True,
            "column_limit": 100,
            "indent_width": 4,
        }

    def mask_control_structures(self, code: str) -> str:
        for pattern, replacement in self.control_structure_patterns.items():
            code = re.sub(pattern, replacement, code)
        return code

    def unmask_control_structures(self, code: str) -> str:
        for replacement, original in self.reverse_control_structure_patterns.items():
            code = code.replace(replacement, original)
        return code

    def format_code(self, code: str):
        try:
            code = self.mask_control_structures(code)
            formatted_code, _changed = yapf.yapf_api.FormatCode(
                code,
                style_config=self.yapf_style
            )
            return self.unmask_control_structures(formatted_code)
        except Exception:
            return None

class _SpaceReducer:
    def __init__(self, source: str):
        self.source = source
        self.tokens = []
        if source:
            self._tokenize_source()

    def _tokenize_source(self):
        token_gen = tokenize.generate_tokens(StringIO(self.source).readline)
        self.tokens = list(token_gen)

    def _handle_empty_control_structure(self, token_idx):
        if token_idx + 1 < len(self.tokens):
            next_token = self.tokens[token_idx + 1]
            if next_token.type in (tokenize.NEWLINE, tokenize.NL):
                return True
        return False

    def reduce_spaces(self) -> str:
        result = []
        prev_token = None
        line_start = True
        previous_was_newline = False

        for i, token in enumerate(self.tokens):
            tok_type = token.type
            tok_string = token.string
            start = token.start

            if line_start and tok_type not in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                if prev_token and prev_token.end[0] != start[0]:
                    result.append(' ' * start[1])
                line_start = False

            if tok_type in (tokenize.NEWLINE, tokenize.NL):
                if not previous_was_newline:
                    result.append('\n')
                    previous_was_newline = True
                line_start = True
            elif tok_type == tokenize.COMMENT:
                if prev_token and prev_token.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                    result.append(' ')
                result.append(tok_string)
                previous_was_newline = False
            elif tok_type == tokenize.STRING:
                result.append(tok_string)
                previous_was_newline = False
            elif tok_type == tokenize.OP:
                result.append(tok_string)
                previous_was_newline = False
            elif tok_type in (tokenize.NAME, tokenize.NUMBER):
                if tok_type == tokenize.NAME and tok_string in ('if', 'while', 'for') and self._handle_empty_control_structure(i):
                    result.append(tok_string)
                else:
                    if prev_token and prev_token.type in (tokenize.NAME, tokenize.NUMBER):
                        result.append(' ')
                    result.append(tok_string)
                previous_was_newline = False
            elif tok_type in (tokenize.INDENT, tokenize.DEDENT):
                continue
            else:
                result.append(tok_string)
                previous_was_newline = False

            prev_token = token

        processed = ''.join(result)
        lines = processed.splitlines()

        cleaned_lines = []
        blank_line = False
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                if not blank_line:
                    cleaned_lines.append('')
                    blank_line = True
            else:
                cleaned_lines.append(stripped)
                blank_line = False

        return '\n'.join(cleaned_lines) + '\n'


# python压缩器
class _PythonUnformatterSingle:
    def __init__(self):
        self.reverse_control_structure_patterns = {
            'for _tempmask_ in [None]:': 'for # TODO: Your code here:',
            'if [TEMPMASK]:': 'if # TODO: Your code here:',
            'while [TEMPMASK]:': 'while # TODO: Your code here:',
        }
        self.space_reducer = _SpaceReducer("")

    def unmask_control_structures(self, code: str) -> str:
        for replacement, original in self.reverse_control_structure_patterns.items():
            code = code.replace(replacement, original)
        return code

    def unformat_code(self, code: str):
        try:
            self.space_reducer.source = code
            self.space_reducer._tokenize_source()
            compressed_code = self.space_reducer.reduce_spaces()
            if compressed_code:
                compressed_code = self.unmask_control_structures(compressed_code)
                return compressed_code
            return None
        except Exception:
            return None





def _replace_lines_by_index(orig_lines, target_lines, idxs):
    if len(target_lines) == len(orig_lines):
        out = orig_lines[:]
        for i in idxs:
            out[i] = target_lines[i]
        return out
    index_map = {}
    for j, ln in enumerate(target_lines):
        key = ln.rstrip('\n')
        index_map.setdefault(key, []).append(j)
    out = orig_lines[:]
    for i in idxs:
        key = orig_lines[i].rstrip('\n')
        lst = index_map.get(key)
        if lst:
            j = lst.pop(0)
            out[i] = target_lines[j]
    return out


def _perturb_one_with_format_strategy(text, pct=0.3, mode='format'):
    if not isinstance(text, str) or text == '':
        return text
    if mode == 'format':
        formatter = _PythonFormatterSingle()
        target = formatter.format_code(text)
    else:
        unformatter = _PythonUnformatterSingle()
        target = unformatter.unformat_code(text)
    if target is None:
        return text
    orig_lines = text.splitlines(True)
    target_lines = target.splitlines(True)
    n = len(orig_lines)
    if n == 0:
        return text
    n_pick = int(n * pct)
    if n_pick <= 0:
        n_pick = 1
    idxs = np.random.choice(n, n_pick, replace=False)
    out_lines = _replace_lines_by_index(orig_lines, target_lines, idxs)
    return ''.join(out_lines)

def vislualize_distribution(predictions, title, ax):

    # remove non-finite pairs (NaN/Inf) together to keep alignment
    reals = []
    samples = []
    for r, s in zip(predictions['real'], predictions['samples']):
        if np.isfinite(r) and np.isfinite(s):
            reals.append(r)
            samples.append(s)

    if len(reals) == 0 or len(samples) == 0:
        # nothing to plot; show empty panel with title
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        return

    ax.hist(reals, bins=30, density=True, alpha=0.5,
            color='orange', edgecolor='orange', label='Real')
    ax.hist(samples, bins=30, density=True,
            alpha=0.5, color='green', edgecolor='green', label='Samples')

    # overlay normal fits if variance is valid
    try:
        mu_r, std_r = norm.fit(reals)
        data_min = min(min(reals), min(samples))
        data_max = max(max(reals), max(samples))
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
            data_min, data_max = data_min - 1.0, data_max + 1.0
        if std_r and np.isfinite(std_r) and std_r > 0:
            x = np.linspace(data_min, data_max, 100)
            p = norm.pdf(x, mu_r, std_r)
            ax.plot(x, p, linewidth=3, color='orange')
    except Exception:
        pass

    try:
        mu_s, std_s = norm.fit(samples)
        data_min = min(min(reals), min(samples))
        data_max = max(max(reals), max(samples))
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
            data_min, data_max = data_min - 1.0, data_max + 1.0
        if std_s and np.isfinite(std_s) and std_s > 0:
            x = np.linspace(data_min, data_max, 100)
            p = norm.pdf(x, mu_s, std_s)
            ax.plot(x, p, linewidth=3, color='green')
    except Exception:
        pass

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()


def main():
    """Main function to run the code detection pipeline."""
    args = setup_args()

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir, base_model_name, SAVE_FOLDER = preprocess_and_save(args)
    model_config = {}
    model_config['cache_dir'] = cache_dir

    # mask filling t5 model
    model_config = load_mask_filling_model(
        args, mask_filling_model_name, model_config)

    logger.info(f'args: {args}')

    # When using TS-based replacement perturbation, enable TS filtering in loader
    ts_mode = _is_ts_mode(getattr(args, 'perturb_type', ''))
    data = generate_data(
        args.dataset,
        args.dataset_key,
        max_num=args.n_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        max_comment_num=args.max_comment_num,
        max_def_num=args.max_def_num,
        cut_def=args.cut_def,
        max_todo_num=args.max_todo_num,
        ts_filter=ts_mode,
    )

    logger.info(f'Original: {data["original"][0]}')
    logger.info(f'Sampled: {data["sampled"][0]}')

    ceil_pct = False
    texts = ['''
    def remove_mask_space(text, args, **kwargs):
        # find all the mask positions " <extra_id_\d+> ", and remove the space before and after the mask
        pattern = re.compile(r" <extra_id_\d+> ")
        matches = pattern.findall(text)
        for match in matches:
            text = text.replace(match, match.strip())
        return text
    ''']
    span_length = args.span_length
    pct = args.pct_words_masked
    lambda_poisson = args.span_length

    # Prepare texts
    prompts = data.get('prompt', [])
    original_text = data["original"]
    sampled_text = data["sampled"]

    if args.perturb_type == 'random':
        masked_texts = [tokenize_and_mask(
            x, args, span_length, pct, ceil_pct) for x in texts]
    elif _is_ts_mode(args.perturb_type):
        # TS 模式下，这里只是演示占位，不做掩码预览
        masked_texts = list(texts)
    elif args.perturb_type == 'identifier-masking':
        masked_texts = [tokenize_and_mask_identifiers(
            x, args, span_length, pct, ceil_pct) for x in texts]
    elif args.perturb_type == 'random-line-shuffle':
        masked_texts = [random_line_shuffle(x, pct) for x in texts]
    elif args.perturb_type == 'random-insert-newline':
        masked_texts = [random_insert_newline(
            x, pct, lambda_poisson) for x in texts]
    elif args.perturb_type == 'random-insert-space':
        masked_texts = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
    elif args.perturb_type == 'random-insert-space-newline':
        masked_texts = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        masked_texts = [random_insert_newline(
            x, pct, lambda_poisson) for x in masked_texts]
    elif args.perturb_type == 'random-insert-space+newline':
        perturbed_texts_part1 = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        perturbed_texts_part2 = [random_insert_newline(
            x, pct, lambda_poisson) for x in texts]
        total_num = len(perturbed_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        perturbed_texts_part1 = perturbed_texts_part1[:n1]
        perturbed_texts_part2 = perturbed_texts_part2[:n2]
        masked_texts = perturbed_texts_part1 + perturbed_texts_part2
    elif args.perturb_type == 'random-insert-comment':
        masked_texts = [random_insert_comment(
            x, pct, lambda_poisson) for x in texts]
    elif args.perturb_type == 'random-comment-out-lines':
        masked_texts = [random_comment_out_lines(
            x, pct) for x in texts]
    elif args.perturb_type == 'random-insert-empty-comment-lines':
        masked_texts = [random_insert_empty_comment_lines(
            x, pct) for x in texts]
    elif args.perturb_type == 'space+comment':
        masked_texts_part1 = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        masked_texts_part2 = [random_insert_comment(
            x, pct, lambda_poisson) for x in texts]
        total_num = len(masked_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        masked_texts_part1 = masked_texts_part1[:n1]
        masked_texts_part2 = masked_texts_part2[:n2]
        masked_texts = masked_texts_part1 + masked_texts_part2
    elif args.perturb_type == 'newline+comment':
        masked_texts_part1 = [random_insert_newline(
            x, pct, lambda_poisson) for x in texts]
        masked_texts_part2 = [random_insert_comment(
            x, pct, lambda_poisson) for x in texts]
        total_num = len(masked_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        masked_texts_part1 = masked_texts_part1[:n1]
        masked_texts_part2 = masked_texts_part2[:n2]
        masked_texts = masked_texts_part1 + masked_texts_part2
    elif args.perturb_type == 'space+newline+comment':
        masked_texts_part1 = [random_insert_space(
            x, pct, lambda_poisson) for x in texts]
        masked_texts_part2 = [random_insert_newline(
            x, pct, lambda_poisson) for x in texts]
        masked_texts_part3 = [random_insert_comment(
            x, pct, lambda_poisson) for x in texts]
        total_num = len(masked_texts_part1)
        n1 = total_num // 3
        n2 = (total_num - n1) // 2
        n3 = total_num - n1 - n2
        masked_texts_part1 = masked_texts_part1[:n1]
        masked_texts_part2 = masked_texts_part2[:n2]
        masked_texts_part3 = masked_texts_part3[:n3]
        masked_texts = masked_texts_part1 + masked_texts_part2 + masked_texts_part3
    elif args.perturb_type == 'gen-comment':
        mode = getattr(args, 'gen_comment_mode', 'inline')
        masked_texts = [random_generate_line_comment(
            x, pct, mode=mode, model_config=model_config, args=args) for x in texts]
        # gen-comment does not use T5 masking; return directly
        return masked_texts
    elif args.perturb_type == 'random-line-format':
        masked_texts = [_perturb_one_with_format_strategy(
            x, pct, mode='format') for x in texts]
    elif args.perturb_type == 'random-line-unformat':
        masked_texts = [_perturb_one_with_format_strategy(
            x, pct, mode='unformat') for x in texts]
    elif args.perturb_type == 'random-line-unformat+format':
        masked_texts_part1 = [_perturb_one_with_format_strategy(
            x, pct, mode='unformat') for x in texts]
        masked_texts_part2 = [_perturb_one_with_format_strategy(
            x, pct, mode='format') for x in texts]
        total_num = len(masked_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        masked_texts_part1 = masked_texts_part1[:n1]
        masked_texts_part2 = masked_texts_part2[:n2]
        masked_texts = masked_texts_part1 + masked_texts_part2
    elif args.perturb_type == 'random-insert-empty-triple-quote':
        masked_texts = [random_insert_empty_triple_quote_lines(
            x, pct) for x in texts]
    elif args.perturb_type == 'random-insert-empty-triple-quote-without-newline':
        masked_texts = [random_insert_empty_triple_quote_lines_without_newline(
            x, pct) for x in texts]
    elif args.perturb_type == 'random-insert-noise-chars':
        masked_texts = [random_insert_noise_chars(
            x, pct, lambda_poisson) for x in texts]
    else:
        raise ValueError(f'Unknown perturb_type: {args.perturb_type}')

    if not ts_mode:
        raw_fills = replace_masks(masked_texts, model_config, args)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        logger.info(f'original texts: {texts[0]}')
        logger.info(f'masked_texts: {masked_texts[0]}')
        logger.info(f'perturbed_texts: {perturbed_texts[0]}')
    else:
        logger.info('TS mode: skip mask-based preview demo logs')

    # from baselines.detectGPT import perturb_texts

    original_text = data["original"]
    sampled_text = data["sampled"]

    # Always use the standard perturbation pipeline function
    perturb_fn = functools.partial(perturb_texts, args=args, model_config=model_config)

    # Prepare TS prompts sequence to align with expanded texts when TS mode is active
    if ts_mode:
        N = max(n_perturbation_list)
        model_config['ts_prompts_seq'] = [p for p in prompts for _ in range(N)]
        model_config['ts_prompt_idx'] = 0
    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(max(n_perturbation_list))])
    if ts_mode:
        model_config['ts_prompt_idx'] = 0
    p_original_text = perturb_fn([x for x in original_text for _ in range(max(n_perturbation_list))])

    results = []
    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * max(n_perturbation_list): (idx + 1) * max(n_perturbation_list)],
            "perturbed_original": p_original_text[idx * max(n_perturbation_list): (idx + 1) * max(n_perturbation_list)]
        })

    selected_index = 1
    selected_perturb = 3

    print(original_text[selected_index])
    # p_original_text[:5]
    print(p_original_text[int(args.n_perturbation_list)
          * selected_index+selected_perturb])
    # print the difference between the original and perturbed text
    print("\nDifference between original and perturbed text:")
    print([x for x in p_original_text[int(args.n_perturbation_list)*selected_index +
          selected_perturb].split(' ') if x not in original_text[selected_index].split(' ')])

    # show the length of the original and perturbed text
    print(f"original text length: {len(original_text)}")
    print(f"perturbed text length: {len(p_original_text)}")

    model_config['mask_model'] = model_config['mask_model'].cpu()
    torch.cuda.empty_cache()

    # start to load the base scoring model
    model_config = load_base_model_and_tokenizer(args, model_config)

    for res in tqdm(results, desc="Computing unperturbed log likelihoods"):
        res["original_ll"] = get_ll(res["original"], args, model_config)
        res["sampled_ll"] = get_ll(res["sampled"], args, model_config)

    for res in tqdm(results, desc="Computing unperturbed log rank"):
        res["original_logrank"] = get_rank(
            res["original"], args, model_config, log=True)
        res["sampled_logrank"] = get_rank(
            res["sampled"], args, model_config, log=True)

    for res in tqdm(results, desc="Computing perturbed log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"], args, model_config)
        p_original_ll = get_lls(res["perturbed_original"], args, model_config)

        for n_perturbation in n_perturbation_list:
            res[f"perturbed_sampled_ll_{n_perturbation}"] = np.mean(
                [i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)])
            res[f"perturbed_original_ll_{n_perturbation}"] = np.mean(
                [i for i in p_original_ll[:n_perturbation] if not math.isnan(i)])
            res[f"perturbed_sampled_ll_std_{n_perturbation}"] = np.std([i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)]) if len([
                i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)]) > 1 else 1
            res[f"perturbed_original_ll_std_{n_perturbation}"] = np.std([i for i in p_original_ll[:n_perturbation] if not math.isnan(i)]) if len([
                i for i in p_original_ll[:n_perturbation] if not math.isnan(i)]) > 1 else 1

    for res in tqdm(results, desc="Computing perturbed log rank"):
        p_sampled_rank = get_ranks(
            res["perturbed_sampled"], args, model_config, log=True)
        p_original_rank = get_ranks(
            res["perturbed_original"], args, model_config, log=True)
        for n_perturbation in n_perturbation_list:
            res[f"perturbed_sampled_logrank_{n_perturbation}"] = np.mean(
                [i for i in p_sampled_rank[:n_perturbation] if not math.isnan(i)])
            res[f"perturbed_original_logrank_{n_perturbation}"] = np.mean(
                [i for i in p_original_rank[:n_perturbation] if not math.isnan(i)])

    torch.cuda.empty_cache()

    # corresponds to the number of samples, and the result of each sample is stored in a dictionary
    print(len(results))
    # corresponds to the computed metrics of for each sample
    print(results[0].keys())

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    predictions = {'real': [], 'samples': []}
    for res in results:
        predictions['real'].append(-res['original_logrank'])
        predictions['samples'].append(-res['sampled_logrank'])
    _, _, roc_auc = get_roc_metrics(
        predictions['real'], predictions['samples'])

    print(f"ROC AUC of logrank: {roc_auc}")
    vislualize_distribution(predictions, f'Logrank AUC = {roc_auc}', axs[0, 0])

    predictions = {'real': [], 'samples': []}
    for res in results:
        predictions['real'].append(-res['original_ll']/res['original_logrank'])
        predictions['samples'].append(-res['sampled_ll'] /
                                      res['sampled_logrank'])
    _, _, roc_auc = get_roc_metrics(
        predictions['real'], predictions['samples'])
    print(f'ROC AUC of LRR: {roc_auc}')
    vislualize_distribution(predictions, f'LRR AUC = {roc_auc}', axs[0, 1])

    predictions = {'real': [], 'samples': []}
    for res in results:
        real_comp = (
            res['original_ll'] - res[f'perturbed_original_ll_{n_perturbation}']) / res[f'perturbed_original_ll_std_{n_perturbation}']
        sample_comp = (
            res['sampled_ll'] - res[f'perturbed_sampled_ll_{n_perturbation}']) / res[f'perturbed_sampled_ll_std_{n_perturbation}']

        # avoid nan
        if math.isnan(real_comp) or math.isnan(sample_comp):
            logger.warning(f"NaN detected, skipping")
            continue

        predictions['real'].append(real_comp)
        predictions['samples'].append(sample_comp)
    _, _, roc_auc = get_roc_metrics(
        predictions['real'], predictions['samples'])

    print(f"ROC AUC of DetectGPT with DetectCodeGPT's perturbation")
    vislualize_distribution(
        predictions, f"DetectGPT with DetectCodeGPT's perturbation AUC = {roc_auc}", axs[1, 0])

    predictions = {'real': [], 'samples': []}
    for res in results:
        predictions['real'].append(
            res[f'perturbed_original_logrank_{n_perturbation}']/res["original_logrank"])
        predictions['samples'].append(
            res[f'perturbed_sampled_logrank_{n_perturbation}']/res["sampled_logrank"])
    _, _, roc_auc = get_roc_metrics(
        predictions['real'], predictions['samples'])
    print(f'ROC AUC of DetectCodeGPT: {roc_auc}')
    vislualize_distribution(
        predictions, f'DetectCodeGPT AUC = {roc_auc}', axs[1, 1])

    plt.tight_layout()
    # 取 base_model_name 作为文件名，去除路径中的斜杠和特殊字符
    import re
    base_name = args.base_model_name if args.base_model_name else 'results'
    base_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
    perturb = getattr(args, 'perturb_type', 'perturb') or 'perturb'
    perturb = re.sub(r'[^a-zA-Z0-9_-]', '_', perturb)
    minlen = str(getattr(args, 'min_len', 'NA'))
    pdf_name = f"{base_name}-{perturb}-minlen{minlen}.pdf"
    plt.savefig(pdf_name)
    print(f"结果已保存为: {pdf_name}")


if __name__ == "__main__":
    main()
