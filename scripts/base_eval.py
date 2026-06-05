"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288

    # Evaluate a Foundry Local model (OpenAI-compatible endpoint)
    python -m scripts.base_eval --eval core --foundry-model gpt-oss-2b --foundry-base-url http://127.0.0.1:5273
"""
import os
import csv
import time
import json
import yaml
import urllib.error
import urllib.request
import urllib.parse
from datetime import datetime
import shutil
import random
import zipfile
import tempfile
import argparse
import subprocess
import re
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task, render_prompts_mc, render_prompts_schema, render_prompts_lm
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM

    # Reduce VRAM pressure on CUDA by loading in reduced precision.
    load_kwargs = {"low_cpu_mem_usage": True}
    if device.type == "cuda":
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        load_kwargs["torch_dtype"] = torch.bfloat16 if bf16_ok else torch.float16

    model = AutoModelForCausalLM.from_pretrained(hf_path, **load_kwargs)
    model.to(device)
    model.eval()

    # Respect model context length so eval truncation can prevent OOM.
    max_seq_len = None
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                max_seq_len = value
                break

    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes


# -----------------------------------------------------------------------------
# Foundry Local loading utilities

def _normalize_foundry_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("Foundry base URL is empty")
    return base_url


def discover_foundry_base_url():
    """Best-effort discovery of Foundry Local service URL."""
    try:
        result = subprocess.run(
            ["foundry", "service", "status"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        match = re.search(r"(https?://[^\s]+)/openai/status", text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def _normalize_path_for_match(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text)


def find_foundry_tokenizer_dir(model_id: str, tokenizer_dir: str | None = None):
    """Find tokenizer.json for a Foundry Local model."""
    if tokenizer_dir:
        tk_path = os.path.join(tokenizer_dir, "tokenizer.json")
        if not os.path.exists(tk_path):
            raise FileNotFoundError(f"tokenizer.json not found in --foundry-tokenizer-dir: {tokenizer_dir}")
        return tokenizer_dir

    roots = []
    env_dir = os.environ.get("FOUNDRY_LOCAL_MODEL_DIR", "").strip()
    if env_dir:
        roots.append(env_dir)
    roots.append(os.path.join(os.path.expanduser("~"), ".foundry", "cache", "models"))

    model_tokens = [tok for tok in _normalize_path_for_match(model_id).split() if len(tok) > 1]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "tokenizer.json" not in filenames:
                continue
            path_tokens = set(_normalize_path_for_match(dirpath).split())
            if all(tok in path_tokens for tok in model_tokens):
                return dirpath
    return None


class FoundryLocalClient:
    """OpenAI-compatible client for Foundry Local."""
    def __init__(self, model_id: str, base_url: str, api_key: str | None, timeout: int = 120):
        self.model_id = model_id
        self.base_url = _normalize_foundry_base_url(base_url)
        self.api_key = api_key or "local"
        self.timeout = timeout

    def _completion_urls(self):
        if self.base_url.endswith("/v1") or self.base_url.endswith("/openai/v1"):
            return [f"{self.base_url}/completions"]
        return [
            f"{self.base_url}/v1/completions",
            f"{self.base_url}/openai/v1/completions",
        ]

    def load_model(self, ttl_seconds: int = 600):
        model = urllib.parse.quote(self.model_id, safe="")
        urls = [f"{self.base_url}/openai/load/{model}?ttl={int(ttl_seconds)}"]
        if self.base_url.endswith("/openai"):
            urls = [f"{self.base_url}/load/{model}?ttl={int(ttl_seconds)}"]
        last_error = None
        for url in urls:
            try:
                request = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as _:
                    return
            except Exception as e:
                last_error = e
        if last_error is not None:
            raise RuntimeError(f"Failed to load Foundry model '{self.model_id}': {last_error}") from last_error

    def _post_json(self, url: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def completions(self, *, prompt: str, max_tokens: int, temperature: float, echo: bool = False, logprobs: int | None = None):
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if echo:
            payload["echo"] = True
        if logprobs is not None:
            payload["logprobs"] = logprobs

        last_error = None
        for url in self._completion_urls():
            try:
                return self._post_json(url, payload)
            except Exception as e:
                last_error = e
        raise RuntimeError(f"Foundry Local request failed on all completion endpoints: {last_error}") from last_error

    def score_continuation(self, prefix: str, continuation: str):
        """Average token logprob of continuation, conditioned on prefix."""
        full_prompt = prefix + continuation
        response = self.completions(
            prompt=full_prompt,
            max_tokens=0,
            temperature=0.0,
            echo=True,
            logprobs=1,
        )
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Foundry response has no choices")
        choice = choices[0]
        token_logprobs = (choice.get("logprobs") or {}).get("token_logprobs")
        text_offset = (choice.get("logprobs") or {}).get("text_offset")
        if token_logprobs is None or text_offset is None:
            raise RuntimeError("Foundry completion did not return token logprobs/text_offset; logprob scoring is unavailable")
        start_char = len(prefix)
        continuation_lps = [
            lp for lp, offset in zip(token_logprobs, text_offset)
            if offset >= start_char and lp is not None
        ]
        if not continuation_lps:
            raise RuntimeError("No continuation token logprobs returned by Foundry for scoring")
        return sum(continuation_lps) / len(continuation_lps)

    def generate(self, prompt: str, max_tokens: int):
        response = self.completions(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Foundry response has no choices")
        return choices[0].get("text", "")


def evaluate_task_foundry(client: FoundryLocalClient, tokenizer, data, device, task_meta):
    """
    CORE task evaluation backed by Foundry Local API.
    Uses logprob scoring for MC/schema and deterministic completion for LM.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    for idx in range(rank, len(data), world_size):
        item = data[idx]
        task_type = task_meta["task_type"]
        num_fewshot = task_meta["num_fewshot"]
        continuation_delimiter = task_meta["continuation_delimiter"]

        fewshot_examples = []
        if num_fewshot > 0:
            rng = random.Random(1234 + idx)
            available_indices = [i for i in range(len(data)) if i != idx]
            fewshot_indices = rng.sample(available_indices, num_fewshot)
            fewshot_examples = [data[i] for i in fewshot_indices]

        if task_type == "multiple_choice":
            prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
            scores = []
            for choice_text, prompt in zip(item["choices"], prompts):
                prefix = prompt[:-len(choice_text)] if len(choice_text) > 0 else prompt
                scores.append(client.score_continuation(prefix, choice_text))
            pred_idx = max(range(len(scores)), key=lambda i: scores[i])
            is_correct = pred_idx == item["gold"]
        elif task_type == "schema":
            prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
            continuation = item["continuation"]
            scores = []
            for prompt in prompts:
                prefix = prompt[:-len(continuation)] if len(continuation) > 0 else prompt
                scores.append(client.score_continuation(prefix, continuation))
            pred_idx = max(range(len(scores)), key=lambda i: scores[i])
            is_correct = pred_idx == item["gold"]
        elif task_type == "language_modeling":
            prompt_without, prompt_with = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
            continuation = prompt_with[len(prompt_without):]
            continuation_tokens = tokenizer(continuation)
            max_tokens = max(1, len(continuation_tokens))
            generated = client.generate(prompt_without, max_tokens=max_tokens)
            is_correct = generated.startswith(continuation)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        correct[idx] = float(is_correct)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://huggingface.co/datasets/marlosb/auxiliary_data/resolve/main/eval_bundle.zip"
EXCLUDED_CORE_TASKS = {
    "jeopardy",
    "bigbench_qa_wikidata",
    "commonsense_qa",
    "squad",
    "coqa",
    "boolq",
    "bigbench_language_identification",
}


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1, foundry_client: FoundryLocalClient | None = None):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = [t for t in config['icl_tasks'] if t['label'] not in EXCLUDED_CORE_TASKS]

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(
            f"{datetime.now().isoformat(timespec='seconds')} - task start - "
            f"{label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})"
        )

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        if foundry_client is not None:
            accuracy = evaluate_task_foundry(foundry_client, tokenizer, data, device, task_meta)
        else:
            accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(
            f"{datetime.now().isoformat(timespec='seconds')} - task end - {label} | "
            f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s"
        )

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--foundry-model', type=str, default=None, help='Foundry Local model id (OpenAI-compatible, e.g. gpt-oss-2b)')
    parser.add_argument('--foundry-base-url', type=str, default='', help='Foundry Local base URL (auto-detected if omitted)')
    parser.add_argument('--foundry-api-key', type=str, default=os.environ.get("FOUNDRY_LOCAL_API_KEY", "local"), help='Foundry Local API key (default: env or "local")')
    parser.add_argument('--foundry-tokenizer-dir', type=str, default=None, help='Path containing tokenizer.json for foundry model (auto-discovered if omitted)')
    parser.add_argument('--foundry-timeout', type=int, default=120, help='HTTP timeout in seconds for Foundry requests')
    parser.add_argument('--foundry-load-ttl', type=int, default=600, help='Auto-load Foundry model with this TTL before eval (set <=0 to disable)')
    parser.add_argument('--append-report', action='store_true', help='Append this run to existing base-model-evaluation report section')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    is_foundry_model = args.foundry_model is not None
    if sum(int(x) for x in [is_hf_model, is_foundry_model, args.model_tag is not None]) > 1:
        parser.error("Choose exactly one model source: --hf-path OR --foundry-model OR --model-tag")

    foundry_client = None
    if is_foundry_model:
        if any(mode in eval_modes for mode in ("bpb", "sample")):
            parser.error("Foundry Local backend currently supports --eval core only")
        foundry_base_url = args.foundry_base_url.strip()
        if not foundry_base_url:
            foundry_base_url = os.environ.get("FOUNDRY_LOCAL_ENDPOINT", "").strip()
        if not foundry_base_url:
            foundry_base_url = os.environ.get("FOUNDRY_LOCAL_BASE_URL", "").strip()
        if not foundry_base_url:
            foundry_base_url = discover_foundry_base_url() or "http://127.0.0.1:5273"
        print0(f"Using Foundry base URL: {foundry_base_url}")

        tokenizer_dir = find_foundry_tokenizer_dir(args.foundry_model, args.foundry_tokenizer_dir)
        if tokenizer_dir is None:
            parser.error("Could not find tokenizer.json for Foundry model. Set --foundry-tokenizer-dir explicitly.")
        print0(f"Using Foundry tokenizer from: {tokenizer_dir}")
        tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
        foundry_client = FoundryLocalClient(
            model_id=args.foundry_model,
            base_url=foundry_base_url,
            api_key=args.foundry_api_key,
            timeout=args.foundry_timeout,
        )
        if args.foundry_load_ttl > 0:
            print0(f"Loading Foundry model: {args.foundry_model} (ttl={args.foundry_load_ttl})")
            foundry_client.load_model(ttl_seconds=args.foundry_load_ttl)
        model = None
        sequence_len = 1024
        token_bytes = None
        model_name = f"foundry-local/{args.foundry_model}"
        model_slug = f"foundry-local-{args.foundry_model}".replace("/", "-").replace(":", "-")
    elif is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {meta['step']})"
        model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes and not is_hf_model and not is_foundry_model:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and (is_hf_model or is_foundry_model):
        print0("\nSkipping sampling for external models (HF/Foundry not supported)")

    # --- BPB evaluation ---
    if 'bpb' in eval_modes and not is_foundry_model:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task, foundry_client=foundry_client)

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- Log to report ---
    from nanochat.report import get_report
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    if args.append_report:
        base_eval_report = os.path.join(get_base_dir(), "report", "base-model-evaluation.md")
        if os.path.exists(base_eval_report):
            with open(base_eval_report, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Keep previous section body, skip heading + timestamp line.
            previous_body = "".join(lines[3:]) if len(lines) >= 3 else ""
            if previous_body.strip():
                report_data = [previous_body] + report_data

    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
