import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from InternVL.internvl_chat.gui_data_processor.vla_detokenizer import ActionTokenizer

# ─── filename parsing utilities ─────────────────────────────────────────────── #
def detect_format(stem: str) -> str:
    """
    If filename contains 'dbg' → double-bracket grounding,
    'sbg' → single-bracket grounding, else legacy.
    """
    stem_low = stem.lower()
    if "dbg" in stem_low:
        return "grounding"
    if "sbg" in stem_low:
        return "sb_grounding"
    return "legacy"


def has_normalized(stem: str) -> bool:
    return "normalized" in stem.lower()


_RE_NPVG = re.compile(r"npv(\d+)g", re.IGNORECASE)
_RE_NPV  = re.compile(r"npv(\d+)(?!g)", re.IGNORECASE)
_RE_PV   = re.compile(r"pv(\d+)", re.IGNORECASE)


def prompt_base(stem: str) -> str:
    """
    Derive the canonical 'prompt base' (e.g. 'vla_singleimg_system_prompt_v5')
    from any filename containing a 'pv5' or 'npv5g' / 'npv5' pattern.
    """
    if m := _RE_NPVG.search(stem):
        return f"vla_singleimg_system_prompt_v{m.group(1)}"
    if m := _RE_NPV.search(stem):
        return f"vla_singleimg_system_prompt_v{m.group(1)}"
    if m := _RE_PV.search(stem):
        return f"vla_singleimg_system_prompt_v{m.group(1)}"
    # fallback—should rarely happen if naming is consistent
    idx = stem.find("vla_")
    return stem[idx:] if idx >= 0 else stem


# ─── evaluator ──────────────────────────────────────────────────────────────── #
class VLAEvaluator:
    COORD_THR, SCROLL_THR = 50, 5
    mouse_actions    = {"single_click", "double_click", "moveTo", "dragTo"}
    scroll_action    = "vscroll"
    keyboard_actions = {"hotkey", "typewrite"}

    def __init__(self, tokenizer: ActionTokenizer):
        self.tk = tokenizer

    @staticmethod
    def _scale(err: float, thr: int) -> float:
        return max(0.0, 1.0 - err / thr)

    def evaluate_action(self, gt: Dict[str, Any],
                              pred: Dict[str, Any]) -> Dict[str, float]:
        """
        Returns only the metrics relevant to the action type:
          • action_acc      – always
          • button_acc      – mouse only
          • coordinate_mae  – mouse only
          • coordinate_prox – mouse only
          • scroll_mae      – vscroll only
          • scroll_prox     – vscroll only
          • value_acc       – keyboard only
        """
        m: Dict[str, float] = {}
        # action accuracy
        m["action_acc"] = float(gt.get("action") == pred.get("action"))
        act = gt.get("action")

        if act in self.mouse_actions:
            m["button_acc"] = float(gt.get("button") == pred.get("button"))
            xg, yg = gt.get("x"), gt.get("y")
            xp, yp = pred.get("x"), pred.get("y")
            if None in (xg, yg, xp, yp):
                m["coordinate_mae"]  = float(self.COORD_THR + 1)
                m["coordinate_prox"] = 0.0
            else:
                dx, dy = abs(xg - xp), abs(yg - yp)
                m["coordinate_mae"]  = (dx + dy) / 2.0
                m["coordinate_prox"] = float(dx < 5 and dy < 5)

        elif act == self.scroll_action:
            sg, sp = gt.get("n_scrolls"), pred.get("n_scrolls")
            if sg is None or sp is None:
                m["scroll_mae"]  = float(self.SCROLL_THR + 1)
                m["scroll_prox"] = 0.0
            else:
                diff = abs(sg - sp)
                m["scroll_mae"]  = float(diff)
                m["scroll_prox"] = float(diff < 5)

        elif act in self.keyboard_actions:
            v_gt, v_pr = gt.get("value") or [], pred.get("value") or []
            m["value_acc"] = float(v_gt == v_pr)

        # press / wait / complete → only action_acc
        return m

    def _clean_pred(self, s: str) -> str:
        # strip occasional LM markers, fix off-by-one tokens, etc.
        if "<|im_start|>" not in s:
            return s
        s = s.replace("<s>", "").replace("<|im_start|> ", "[UNUSED_TOKEN_0]")
        return re.sub(r"\[UNUSED_TOKEN_(\d+)\]", 
                      lambda m: f"[UNUSED_TOKEN_{int(m.group(1))+1}]", s)

    def load_groundtruth(self, path: str) -> Dict[int, Dict[str, Any]]:
        gt_map: Dict[int, Dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading GT {os.path.basename(path)}"):
                obj = json.loads(line)
                sid = obj["id"]
                enc = next(c["value"] for c in obj["conversations"]
                           if c["from"] == "gpt")
                try:
                    gt_map[sid] = self.tk.decode(enc)
                except Exception:
                    pass
        return gt_map

    def evaluate_file(self,
                      gt_map: Dict[int, Dict[str, Any]],
                      inference_file: str,
                      out_dir: str) -> None:
        metric_sum   = defaultdict(float)
        metric_count = defaultdict(int)
        evals: List[Dict[str, Any]] = []
        parsed_ok = 0

        with open(inference_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Eval {os.path.basename(inference_file)}"):
                obj = json.loads(line)
                sid = obj["id"]
                if sid not in gt_map:
                    continue

                raw = self._clean_pred(obj.get("prediction", ""))
                try:
                    pred    = self.tk.decode(raw)
                    metrics = self.evaluate_action(gt_map[sid], pred)
                    parsed_ok += 1
                except Exception:
                    # if decoding fails, we at least count action_acc=0
                    metrics = {"action_acc": 0.0}

                for k, v in metrics.items():
                    metric_sum[k]   += v
                    metric_count[k] += 1

                evals.append({"id": sid, "metrics": metrics})

        # compute aggregate: for each metric, avg = sum/count, and also report count
        aggregate = {
            k: {"score": metric_sum[k] / metric_count[k], "count": metric_count[k]}
            for k in metric_sum
        }

        os.makedirs(out_dir, exist_ok=True)
        save_path = Path(out_dir) / f"evaluation_{Path(inference_file).name}"
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump({
                "inference_file":           Path(inference_file).name,
                "num_samples":              len(evals),
                "num_successfully_decoded": parsed_ok,
                "aggregate":                aggregate,
                "evaluations":              evals,
            }, fp, indent=2)
        print(f"Saved → {save_path}")


# ─── build a GT index keyed by (base, format, normalized) ───────────────────── #
def build_gt_index(gt_dir: Path) -> Dict[tuple, Path]:
    idx: Dict[tuple, Path] = {}
    for fp in gt_dir.glob("test_*.jsonl"):
        print(fp)
        stem = fp.stem
        if "no_prev" in stem.lower():
            continue
        fmt  = detect_format(stem)
        norm = has_normalized(stem)
        base = prompt_base(stem)
        key  = (base, fmt, norm)
        if key in idx:
            print(f"[WARN] duplicate GT for {key}: {fp} vs {idx[key]}")
        idx[key] = fp

    return idx


# ─── main entrypoint ───────────────────────────────────────────────────────── #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir",        default="groundtruth_vla")
    p.add_argument("--inference_dir", default="250611_Exp_Inference")
    p.add_argument("--out_dir",       default="evaluation_results")
    args = p.parse_args()

    gt_index = build_gt_index(Path(args.gt_dir))
    if not gt_index:
        print("No GT files found."); return

    inf_files = [f for f in Path(args.inference_dir).glob("*.jsonl")
                 if "no_prev" not in f.stem.lower()]
    if not inf_files:
        print("No inference files."); return

    for inf in tqdm(inf_files, desc="Inference sets"):
        stem = inf.stem
        print(stem)
        fmt  = detect_format(stem)
        norm = has_normalized(stem)
        base = prompt_base(stem)

        gt_path = gt_index.get((base, fmt, norm))

        if not gt_path:
            print(f"[ERROR] GT not found for {stem} "
                  f"(format={fmt}, normalized={norm})")
            continue

        tk  = ActionTokenizer(format=fmt)
        ev  = VLAEvaluator(tk)
        gt  = ev.load_groundtruth(str(gt_path))
        if not gt:
            print(f"[WARN] GT empty for {gt_path.name}")
            continue

        ev.evaluate_file(gt, str(inf), args.out_dir)
        time.sleep(random.uniform(0.2, 0.5))


if __name__ == "__main__":
    main()
