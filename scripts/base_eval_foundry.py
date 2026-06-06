"""
Foundry Local CORE evaluation for base models.

This script is invoked from scripts.base_eval when --foundry-model is set.
It preserves the same report and CSV format as base_eval.py.
"""
import argparse
import csv
import difflib
import json
import os
import random
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime

import torch
import torch.distributed as dist
import yaml

from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    download_file_with_lock,
    get_base_dir,
    print0,
)
from nanochat.core_eval import render_prompts_lm, render_prompts_mc, render_prompts_schema
from nanochat.tokenizer import HuggingFaceTokenizer

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
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def write_core_csv(output_csv_path, core_results):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label in core_results["results"]:
            acc = core_results["results"][label]
            centered = core_results["centered_results"][label]
            f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
        f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")


def find_foundry_tokenizer_dir(model_id: str, explicit_dir: str | None):
    if explicit_dir:
        tk_path = os.path.join(explicit_dir, "tokenizer.json")
        if os.path.exists(tk_path):
            return explicit_dir
        raise FileNotFoundError(f"tokenizer.json not found in --foundry-tokenizer-dir: {explicit_dir}")

    model_name = model_id.split(":", 1)[0]
    version = model_id.split(":", 1)[1] if ":" in model_id else None
    roots = []
    env_dir = os.environ.get("FOUNDRY_LOCAL_MODEL_DIR", "").strip()
    if env_dir:
        roots.append(env_dir)
    roots.extend(
        [
            os.path.join(os.path.expanduser("~"), ".foundry", "cache", "models"),
            os.path.join(os.path.expanduser("~"), ".aitk", "cache", "models"),
        ]
    )

    def has_tokenizer(path):
        return os.path.exists(os.path.join(path, "tokenizer.json"))

    for root in roots:
        if not os.path.isdir(root):
            continue

        # Common layout: <root>\<publisher>\<model_name>\v<version>\tokenizer.json
        for publisher in os.listdir(root):
            pub_dir = os.path.join(root, publisher)
            if not os.path.isdir(pub_dir):
                continue
            model_dir = os.path.join(pub_dir, model_name)
            if not os.path.isdir(model_dir):
                continue
            if version:
                vdir = os.path.join(model_dir, f"v{version}")
                if os.path.isdir(vdir) and has_tokenizer(vdir):
                    return vdir
            for child in os.listdir(model_dir):
                cdir = os.path.join(model_dir, child)
                if os.path.isdir(cdir) and has_tokenizer(cdir):
                    return cdir
            if has_tokenizer(model_dir):
                return model_dir

    return None


class FoundryLocalClient:
    def __init__(self, model_id: str, base_url: str, api_key: str, timeout: int):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "local"
        self.timeout = timeout
        self.inference_bases = self._discover_inference_bases()

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _post_json(self, url: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, url: str):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"}, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)

    def _discover_inference_bases(self):
        """
        Foundry service can expose inference on endpoints listed by /openai/status.
        Prefer discovered endpoints first, then fall back to provided base_url.
        """
        bases = []
        for status_url in (f"{self.base_url}/openai/status", f"{self.base_url}/status"):
            try:
                payload = self._get_json(status_url)
                for ep in payload.get("endpoints", []):
                    ep = (ep or "").rstrip("/")
                    if ep and ep not in bases:
                        bases.append(ep)
            except Exception:
                continue
        if self.base_url not in bases:
            bases.append(self.base_url)
        print0(f"Foundry inference endpoints: {bases}")
        return bases

    def _completion_endpoints(self):
        model_candidates = [self.model_id]
        if ":" in self.model_id:
            model_candidates.append(self.model_id.split(":", 1)[0])
        # Older/newer Foundry builds can expose different Azure API versions.
        api_versions = [
            "2025-01-01-preview",
            "2024-10-21",
            "2024-06-01",
            "2024-05-01-preview",
            "2024-02-15-preview",
            "2024-02-01",
            "2023-12-01-preview",
            "2023-05-15",
        ]
        endpoints = []
        for base in self.inference_bases:
            endpoints.extend(
                [
                    (f"{base}/v1/completions", "openai"),
                    (f"{base}/openai/v1/completions", "openai"),
                    (f"{base}/openai/completions", "openai"),
                ]
            )
            for model in model_candidates:
                model_encoded = urllib.parse.quote(model, safe="")
                for api_version in api_versions:
                    endpoints.append(
                        (f"{base}/openai/deployments/{model_encoded}/completions?api-version={api_version}", "azure")
                    )
                    endpoints.append(
                        (f"{base}/deployments/{model_encoded}/completions?api-version={api_version}", "azure")
                    )
        return endpoints

    def _chat_endpoints(self):
        model_candidates = [self.model_id]
        if ":" in self.model_id:
            model_candidates.append(self.model_id.split(":", 1)[0])
        api_versions = [
            "2025-01-01-preview",
            "2024-10-21",
            "2024-06-01",
            "2024-05-01-preview",
            "2024-02-15-preview",
            "2024-02-01",
            "2023-12-01-preview",
            "2023-05-15",
        ]
        endpoints = []
        for base in self.inference_bases:
            endpoints.extend(
                [
                    (f"{base}/v1/chat/completions", "openai"),
                    (f"{base}/openai/v1/chat/completions", "openai"),
                    (f"{base}/openai/chat/completions", "openai"),
                ]
            )
            for model in model_candidates:
                model_encoded = urllib.parse.quote(model, safe="")
                for api_version in api_versions:
                    endpoints.append(
                        (f"{base}/openai/deployments/{model_encoded}/chat/completions?api-version={api_version}", "azure")
                    )
                    endpoints.append(
                        (f"{base}/deployments/{model_encoded}/chat/completions?api-version={api_version}", "azure")
                    )
        return endpoints

    def load_model(self, ttl_seconds: int):
        if ttl_seconds <= 0:
            return
        model_candidates = [self.model_id]
        # Some Foundry installs reject versioned ids in /openai/load; try unversioned alias too.
        if ":" in self.model_id:
            model_candidates.append(self.model_id.split(":", 1)[0])

        endpoint_templates = [
            "{base}/openai/load/{model}?ttl={ttl}",
            "{base}/load/{model}?ttl={ttl}",
        ]

        errors = []
        for candidate in model_candidates:
            model = urllib.parse.quote(candidate, safe="")
            for template in endpoint_templates:
                url = template.format(base=self.base_url, model=model, ttl=int(ttl_seconds))
                req = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    method="GET",
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as _:
                        if candidate != self.model_id:
                            print0(f"Foundry load accepted alias '{candidate}' for requested '{self.model_id}'")
                        return
                except urllib.error.HTTPError as e:
                    errors.append(f"{url} -> HTTP {e.code}")
                except Exception as e:
                    errors.append(f"{url} -> {e}")

        # Don't hard-fail here. Some Foundry versions auto-load on first completion request.
        print0(
            "Warning: explicit Foundry load call failed; continuing with inference requests. "
            f"Tried: {errors}"
        )

    def completions(self, prompt: str, max_tokens: int, temperature: float, echo: bool = False, logprobs: int | None = None):
        base_payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if echo:
            base_payload["echo"] = True
        if logprobs is not None:
            base_payload["logprobs"] = logprobs

        tried = []
        last_error = None
        for url, kind in self._completion_endpoints():
            payload = dict(base_payload)
            if kind == "azure":
                payload.pop("model", None)
            try:
                return self._post_json(url, payload)
            except Exception as e:
                tried.append(url)
                last_error = e
        raise RuntimeError(f"Foundry completion failed. Last error: {last_error}. Tried endpoints: {tried}") from last_error

    def chat_completion(self, prompt: str, max_tokens: int, temperature: float):
        base_payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        tried = []
        last_error = None
        for url, kind in self._chat_endpoints():
            payload = dict(base_payload)
            if kind == "azure":
                payload.pop("model", None)
            try:
                response = self._post_json(url, payload)
                choices = response.get("choices") or []
                if not choices:
                    raise RuntimeError("Chat response has no choices")
                msg = choices[0].get("message") or {}
                text = msg.get("content") or ""
                if not text:
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                return text
            except Exception as e:
                tried.append(url)
                last_error = e
        raise RuntimeError(f"Foundry chat completion failed. Last error: {last_error}. Tried endpoints: {tried}") from last_error

    def score_continuation(self, prefix: str, continuation: str):
        response = self.completions(prefix + continuation, max_tokens=0, temperature=0.0, echo=True, logprobs=1)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Foundry response has no choices")
        logprobs = (choices[0].get("logprobs") or {})
        token_logprobs = logprobs.get("token_logprobs")
        text_offset = logprobs.get("text_offset")
        if token_logprobs is None or text_offset is None:
            raise RuntimeError("Foundry endpoint did not return token logprobs/text_offset")
        start_char = len(prefix)
        continuation_lps = [lp for lp, off in zip(token_logprobs, text_offset) if off >= start_char and lp is not None]
        if not continuation_lps:
            raise RuntimeError("No continuation token logprobs returned for scoring")
        return sum(continuation_lps) / len(continuation_lps)

    def generate(self, prompt: str, max_tokens: int):
        try:
            response = self.completions(prompt, max_tokens=max_tokens, temperature=0.0)
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("Foundry response has no choices")
            return choices[0].get("text", "")
        except Exception:
            return self.chat_completion(prompt, max_tokens=max_tokens, temperature=0.0)


def _normalize_text(text: str):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _best_choice_from_text(answer_text: str, choices: list[str]):
    answer_norm = _normalize_text(answer_text)
    if not answer_norm:
        return 0
    for i, c in enumerate(choices):
        c_norm = _normalize_text(c)
        if answer_norm.startswith(c_norm) or c_norm.startswith(answer_norm):
            return i
    scores = [difflib.SequenceMatcher(None, answer_norm, _normalize_text(c)).ratio() for c in choices]
    return max(range(len(scores)), key=lambda i: scores[i])


def _mc_prompt_for_chat(item, fewshot_examples):
    lines = [
        "Choose the single best option.",
        "Answer using only the option text exactly as written.",
        "",
    ]
    for ex in fewshot_examples:
        lines.append(f"Question: {ex['query']}")
        for i, c in enumerate(ex["choices"], 1):
            lines.append(f"{i}. {c}")
        lines.append(f"Answer: {ex['choices'][ex['gold']]}")
        lines.append("")
    lines.append(f"Question: {item['query']}")
    for i, c in enumerate(item["choices"], 1):
        lines.append(f"{i}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def _schema_prompt_for_chat(item, fewshot_examples):
    lines = [
        "Choose which context best matches the continuation.",
        "Answer with only the option number.",
        "",
    ]
    for ex in fewshot_examples:
        lines.append(f"Continuation: {ex['continuation']}")
        for i, c in enumerate(ex["context_options"], 1):
            lines.append(f"{i}. {c}")
        lines.append(f"Answer: {ex['gold'] + 1}")
        lines.append("")
    lines.append(f"Continuation: {item['continuation']}")
    for i, c in enumerate(item["context_options"], 1):
        lines.append(f"{i}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def _estimate_max_tokens_from_text(text: str):
    """
    Estimate a safe max_tokens budget from raw text when no local tokenizer is available.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 1
    word_estimate = max(1, len(stripped.split()))
    char_estimate = max(1, len(stripped) // 3)
    return max(1, min(512, max(word_estimate, char_estimate)))


def evaluate_task_foundry(client: FoundryLocalClient, tokenizer, data, device, task_meta):
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
            k = min(num_fewshot, len(available_indices))
            fewshot_indices = rng.sample(available_indices, k)
            fewshot_examples = [data[i] for i in fewshot_indices]

        if task_type == "multiple_choice":
            prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
            try:
                scores = []
                for choice_text, prompt in zip(item["choices"], prompts):
                    prefix = prompt[:-len(choice_text)] if choice_text else prompt
                    scores.append(client.score_continuation(prefix, choice_text))
                pred_idx = max(range(len(scores)), key=lambda i: scores[i])
            except Exception:
                answer = client.chat_completion(_mc_prompt_for_chat(item, fewshot_examples), max_tokens=32, temperature=0.0)
                pred_idx = _best_choice_from_text(answer, item["choices"])
            is_correct = pred_idx == item["gold"]
        elif task_type == "schema":
            prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
            continuation = item["continuation"]
            try:
                scores = []
                for prompt in prompts:
                    prefix = prompt[:-len(continuation)] if continuation else prompt
                    scores.append(client.score_continuation(prefix, continuation))
                pred_idx = max(range(len(scores)), key=lambda i: scores[i])
            except Exception:
                answer = client.chat_completion(_schema_prompt_for_chat(item, fewshot_examples), max_tokens=8, temperature=0.0)
                match = re.search(r"\d+", answer or "")
                pred_idx = max(0, min(len(item["context_options"]) - 1, (int(match.group(0)) - 1))) if match else 0
            is_correct = pred_idx == item["gold"]
        elif task_type == "language_modeling":
            prompt_without, prompt_with = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
            continuation = prompt_with[len(prompt_without):]
            if tokenizer is not None:
                max_tokens = max(1, len(tokenizer(continuation)))
            else:
                max_tokens = _estimate_max_tokens_from_text(continuation)
            generated = client.generate(prompt_without, max_tokens=max_tokens)
            is_correct = generated.startswith(continuation)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        correct[idx] = float(is_correct)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


def evaluate_core_foundry(client: FoundryLocalClient, tokenizer, device, max_per_task=-1):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = [t for t in config["icl_tasks"] if t["label"] not in EXCLUDED_CORE_TASKS]

    random_baselines = {}
    with open(eval_meta_data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(
            f"{datetime.now().isoformat(timespec='seconds')} - task start - "
            f"{label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})"
        )

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task_foundry(client, tokenizer, data, device, task_meta)
        results[label] = accuracy
        rb = random_baselines[label]
        centered = (accuracy - 0.01 * rb) / (1.0 - 0.01 * rb)
        centered_results[label] = centered

        elapsed = time.time() - start_time
        print0(
            f"{datetime.now().isoformat(timespec='seconds')} - task end - {label} | "
            f"accuracy: {accuracy:.4f} | centered: {centered:.4f} | time: {elapsed:.2f}s"
        )

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}


def run_foundry_eval(args: argparse.Namespace):
    eval_modes = set(mode.strip() for mode in args.eval.split(","))
    if eval_modes != {"core"}:
        raise ValueError("Foundry evaluator supports only --eval core")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    _ = (ddp, ddp_rank, ddp_local_rank, ddp_world_size)  # keep lint quiet

    base_url = args.foundry_base_url.strip()
    print0(f"Using Foundry base URL: {base_url}")
    client = FoundryLocalClient(args.foundry_model, base_url, args.foundry_api_key, args.foundry_timeout)
    if args.foundry_load_ttl > 0:
        print0(f"Loading Foundry model: {args.foundry_model} (ttl={args.foundry_load_ttl})")
        client.load_model(args.foundry_load_ttl)

    tokenizer = None
    tokenizer_dir = find_foundry_tokenizer_dir(args.foundry_model, args.foundry_tokenizer_dir)
    if tokenizer_dir is not None:
        print0(f"Using Foundry tokenizer from: {tokenizer_dir}")
        tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    else:
        print0(
            "Foundry tokenizer.json not found; continuing with raw-text prompts only. "
            "Token-based generation length will be estimated."
        )

    model_name = f"foundry-local/{args.foundry_model}"
    model_slug = args.foundry_model.replace("/", "-").replace(":", "-")
    print0(f"Evaluating model: {model_name}")
    print0("Eval modes: core")
    print0("\n" + "=" * 80)
    print0("CORE Evaluation")
    print0("=" * 80)
    core_results = evaluate_core_foundry(client, tokenizer, device, max_per_task=args.max_per_task)

    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        write_core_csv(output_csv_path, core_results)
        print0(f"\nResults written to: {output_csv_path}")
        print0(f"CORE metric: {core_results['core_metric']:.4f}")

    from nanochat.report import get_report

    report_data = [{"model": model_name, "CORE metric": core_results["core_metric"]}, core_results["centered_results"]]
    if args.append_report:
        base_eval_report = os.path.join(get_base_dir(), "report", "base-model-evaluation.md")
        if os.path.exists(base_eval_report):
            with open(base_eval_report, "r", encoding="utf-8") as f:
                lines = f.readlines()
            previous_body = "".join(lines[3:]) if len(lines) >= 3 else ""
            if previous_body.strip():
                report_data = [previous_body] + report_data
    get_report().log(section="Base model evaluation", data=report_data)
    compute_cleanup()
