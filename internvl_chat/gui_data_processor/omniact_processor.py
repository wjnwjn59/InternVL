import os
import json
import argparse
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

# Import tokenizer (works if file is vla_utils/vla_detokenizer.py or local vla_detokenizer.py)
try:
    from vla_utils.vla_detokenizer import ActionTokenizer
except Exception:
    from vla_detokenizer import ActionTokenizer


def format_human(task_text, image_token: str = '<image>') -> str:
    return f"{image_token}\nQ: what action should the agent take to {task_text.strip()}? A:"


def get_image_size(image_path: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def _toggle_screen_name(file_name: str) -> List[str]:
    """
    Toggle between screen{n}.png and screen_{n}.png.
    Returns candidate filenames (excluding the original if unchanged).
    """
    cands: List[str] = []
    m = re.match(r'^(.*?)(screen)_(\d+)(\.png)$', file_name, re.IGNORECASE)
    if m:
        cands.append(m.group(1) + m.group(2) + m.group(3) + m.group(4))  # screen_2.png -> screen2.png
    else:
        m2 = re.match(r'^(.*?)(screen)(\d+)(\.png)$', file_name, re.IGNORECASE)
        if m2:
            cands.append(m2.group(1) + m2.group(2) + "_" + m2.group(3) + m2.group(4))  # screen2.png -> screen_2.png
    return cands


# -------------------- pyautogui -> action dicts --------------------
# Accept ints OR floats for coordinates, positional or named.
_NUM = r'-?(?:\d+(?:\.\d+)?|\.\d+)'
_COORD_KV = (
    rf'(?:x\s*=\s*(?P<x>{_NUM})\s*,\s*y\s*=\s*(?P<y>{_NUM})'
    rf'|(?P<x2>{_NUM})\s*,\s*(?P<y2>{_NUM}))'
)
_BUTTON_KV = r"(?:button\s*=\s*['\"](?P<button>left|right)['\"])"
_TEXT_Q = r"['\"](?P<text>.*?)['\"]"
# Allow ANY non-quote chars for key names so "+", "=", etc. work
_KEYS_Q = r"['\"](?P<key>[^'\"]+)['\"]"

# Tolerate trailing commas and extra kwargs we don't care about: (?:\s*,[^)]*)?
_click_re      = re.compile(rf"pyautogui\.(?:click)\(\s*{_COORD_KV}(?:\s*,\s*{_BUTTON_KV})?(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)
_dblclick_re   = re.compile(rf"pyautogui\.(?:doubleclick)\(\s*{_COORD_KV}(?:\s*,\s*{_BUTTON_KV})?(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)

# rightClick: coords or no-args
_rightclick_re       = re.compile(rf"pyautogui\.(?:rightclick)\(\s*{_COORD_KV}(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)
_rightclick_noargs_re= re.compile(r"pyautogui\.(?:rightclick)\(\s*(?:,[^)]*)?\s*\)", re.IGNORECASE)

# moveTo: numeric coords OR a single anchor string (ignore the string)
_move_re         = re.compile(rf"pyautogui\.(?:moveto)\(\s*{_COORD_KV}(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)
_move_anchor_re  = re.compile(r"pyautogui\.(?:moveto)\(\s*['\"][^'\"]+['\"](?:\s*,[^)]*)?\s*\)", re.IGNORECASE)

_drag_re         = re.compile(rf"pyautogui\.(?:dragto)\(\s*{_COORD_KV}(?:\s*,\s*{_BUTTON_KV})?(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)

# Vertical scroll (accept + / -)
_scroll_re       = re.compile(r"pyautogui\.(?:scroll)\(\s*(?P<scroll>[+-]?\d+)(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)
# Horizontal scroll: number can be quoted or not, with optional sign
_hscroll_re      = re.compile(r"pyautogui\.(?:hscroll)\(\s*(?P<hscroll>['\"]?[+-]?\d+['\"]?)(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)

# Write / typewrite (keep full string in dict; tokenizer omits when encoding)
_write_re        = re.compile(rf"pyautogui\.(?:write|typewrite)\(\s*{_TEXT_Q}(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)

# Press / Hotkey (allow symbols)
_press_re        = re.compile(rf"pyautogui\.(?:press)\(\s*{_KEYS_Q}(?:\s*,[^)]*)?\s*\)", re.IGNORECASE)
_hotkeys_blob_re = re.compile(r"pyautogui\.(?:hotkey)\(\s*(?P<keys>['\"][^'\"]+['\"](?:\s*,\s*['\"][^'\"]+['\"])*)\s*(?:,[^)]*)?\)", re.IGNORECASE)


def _float_to_int(s: Optional[str]) -> Optional[int]:
    """round-half-up (away from zero) for pixel coords."""
    if s is None:
        return None
    try:
        x = float(s)
        if x >= 0:
            return int(math.floor(x + 0.5))
        else:
            return int(math.ceil(x - 0.5))
    except Exception:
        return None


def _extract_coords(m: re.Match) -> Tuple[Optional[int], Optional[int]]:
    if m.groupdict().get("x") is not None:
        return _float_to_int(m.group("x")), _float_to_int(m.group("y"))
    if m.groupdict().get("x2") is not None:
        return _float_to_int(m.group("x2")), _float_to_int(m.group("y2"))
    return None, None


def _norm_button(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = s.strip().lower()
    return s2 if s2 in {"left", "right"} else None


def _clean_literal(s: str) -> str:
    t = s.strip()
    t = t.replace(r"\\'", "'").replace(r'\\"', '"')
    t = t.replace(r"\'", "'").replace(r'\"', '"')
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        t = t[1:-1]
    t = t.strip('\'"').strip()
    return t


def _split_quoted_keys(keys_blob: str) -> List[str]:
    # '"ctrl", "+", "w"' → ["ctrl","+","w"]
    parts = [p.strip() for p in keys_blob.split(",")]
    cleaned = []
    for p in parts:
        c = _clean_literal(p)
        if c:
            cleaned.append(c)
    return cleaned


def pyautogui_script_to_actions(
    script_text: str,
    before_after_hint: Optional[Tuple[Optional[str], Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a text block with pyautogui calls into a list of normalized action dicts.
    - WRITE keeps full string in dict (value=["..."]); tokenizer omits it when encoding.
    - moveTo("anchor"): ignore anchor → move_to with x=None, y=None.
    """
    before_frame, after_frame = (before_after_hint or (None, None))
    actions: List[Dict[str, Any]] = []
    unparsed: List[str] = []

    for raw in script_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = _dblclick_re.search(line)
        if m:
            x, y = _extract_coords(m)
            btn = _norm_button(m.groupdict().get("button"))
            actions.append(dict(action="double_click", button=btn or "left",
                                x=x, y=y, n_scrolls=None, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # rightClick with coords
        m = _rightclick_re.search(line)
        if m:
            x, y = _extract_coords(m)
            actions.append(dict(action="single_click", button="right",
                                x=x, y=y, n_scrolls=None, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # rightClick() no-args
        m = _rightclick_noargs_re.search(line)
        if m:
            actions.append(dict(action="single_click", button="right",
                                x=None, y=None, n_scrolls=None, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # moveTo("anchor") → ignore anchor (no coords)
        m = _move_anchor_re.search(line)
        if m:
            actions.append(dict(action="move_to", button=None,
                                x=None, y=None, n_scrolls=None, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # moveTo(x, y)
        m = _move_re.search(line)
        if m:
            x, y = _extract_coords(m)
            actions.append(dict(action="move_to", button=None,
                                x=x, y=y, n_scrolls=None, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # dragTo(x, y, [button=...])
        m = _drag_re.search(line)
        if m:
            x, y = _extract_coords(m)
            btn = _norm_button(m.groupdict().get("button")) or "left"
            actions.append(dict(action="drag_to", button=btn,
                                x=x, y=y, n_scrolls=None, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # click(x, y, [button=...])
        m = _click_re.search(line)
        if m:
            x, y = _extract_coords(m)
            btn = _norm_button(m.groupdict().get("button")) or "left"
            actions.append(dict(action="single_click", button=btn,
                                x=x, y=y, n_scrolls=None, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # scroll / hscroll
        m = _scroll_re.search(line)
        if m:
            s = int(m.group("scroll"))
            actions.append(dict(action="vscroll", button=None,
                                x=None, y=None, n_scrolls=s, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        m = _hscroll_re.search(line)
        if m:
            val = _clean_literal(m.group("hscroll"))
            s = int(val)
            actions.append(dict(action="hscroll", button=None,
                                x=None, y=None, n_scrolls=s, value=[],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # write / typewrite
        m = _write_re.search(line)
        if m:
            txt = m.group("text")
            actions.append(dict(action="write", button=None,
                                x=None, y=None, n_scrolls=None, value=[txt] if txt is not None else [],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # press
        m = _press_re.search(line)
        if m:
            key = m.group("key")
            actions.append(dict(action="press", button=None,
                                x=None, y=None, n_scrolls=None, value=[key],
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # hotkey
        m = _hotkeys_blob_re.search(line)
        if m:
            keys = _split_quoted_keys(m.group("keys"))
            actions.append(dict(action="hotkey", button=None,
                                x=None, y=None, n_scrolls=None, value=keys,
                                before_frame=before_frame, after_frame=after_frame))
            continue

        # Not matched
        if line:
            unparsed.append(line)

    # If nothing parsed, fail fast (print full script + unparsed lines)
    if not actions:
        raise ValueError(
            "No actions parsed from output script.\n"
            "----- BEGIN SCRIPT -----\n"
            f"{script_text}\n"
            "-----  END  SCRIPT -----\n"
            f"Unparsed lines: {unparsed}"
        )

    return actions


# -------------------- render actions via tokenizer --------------------
def render_actions(
    actions: List[Dict[str, Any]],
    action_format: str = "nl",          # nl | legacy | concise
    grounding_format: str = "double",   # double | single
) -> str:
    fmt = (action_format or "nl").strip().lower()
    if fmt == "nl":
        out_lines = []
        for a in actions:
            parts = [a.get("action") or "None"]
            if a.get("button"):
                parts.append(f"button={a['button']}")
            if a.get("x") is not None and a.get("y") is not None:
                parts.append(f"coords=({a['x']}, {a['y']})")
            if a.get("n_scrolls") is not None:
                parts.append(f"scrolls={a['n_scrolls']}")
            if a.get("value"):
                parts.append(f"value={a['value']}")
            out_lines.append(" | ".join(parts))
        return "\n".join(out_lines).strip()

    tok = ActionTokenizer(action_format=fmt, grounding_format=grounding_format)
    rendered: List[str] = [tok.encode(a) for a in actions]
    return "\n".join(rendered).strip()


# -------------------- main dataset processing --------------------
def process_omniact(
    split_json_path: str,
    output_jsonl_path: str,
    metafile_path: str,
    root_dir: str = "./",
    annotation_name: str = "omniact_chat.jsonl",
    dataset_name: str = "omniact_chat",
    data_augment: bool = False,
    max_dynamic_patch: int = 12,
    repeat_time: int = 1,
    action_format: str = "nl",
    grounding_format: str = "double",
):
    # Load split json
    with open(split_json_path, 'r') as f:
        data = json.load(f)

    chat_data = []

    for k, v in data.items():
        # Read task file
        task_path = os.path.join(root_dir, v["task"]) if not os.path.isabs(v["task"]) else v["task"]
        try:
            with open(task_path, 'r') as tf:
                lines = [l.rstrip('\n') for l in tf.readlines()]
            if not lines:
                raise ValueError(f"Annotation file empty: {task_path}")

            # Input
            if lines[0].startswith("Task:"):
                task_text = lines[0][len("Task:"):].strip()
            else:
                task_text = lines[0].strip()

            # Output script
            output_lines = []
            if len(lines) > 1 and lines[1].strip().lower().startswith('output'):
                after_colon = lines[1].split(':', 1)[1].strip() if ':' in lines[1] else ''
                if after_colon:
                    output_lines = [after_colon]
                    if len(lines) > 2:
                        output_lines += [l for l in lines[2:] if l.strip() != '']
                else:
                    output_lines = [l for l in lines[2:] if l.strip() != '']
            else:
                output_lines = [l for l in lines[1:] if l.strip() != '']
            output_script = "\n".join(output_lines).strip() if output_lines else None
            if not output_script:
                raise ValueError(f"No GPT output found in annotation: {task_path}")

            # Convert to actions (includes fail-fast if none parsed)
            before_after = (v.get("before_frame"), v.get("after_frame"))
            actions = pyautogui_script_to_actions(output_script, before_after_hint=before_after)

            # Render actions
            assistant_text = render_actions(actions, action_format=action_format, grounding_format=grounding_format)

        except Exception as e:
            raise RuntimeError(f"Failed to process annotation {task_path} (id={k}): {e}")

        # Image info (robust path resolution + screenN <-> screen_N toggle)
        image_rel_path = v["image"].lstrip("/\\")
        image_abs_path = os.path.join(root_dir, image_rel_path)

        width, height = get_image_size(image_abs_path)
        if width is None or height is None:
            dir_name, file_name = os.path.split(image_abs_path)
            tried_paths = [image_abs_path]

            for variant in _toggle_screen_name(file_name):
                alt_path = os.path.join(dir_name, variant)
                tried_paths.append(alt_path)
                w, h = get_image_size(alt_path)
                if w is not None and h is not None:
                    width, height = w, h
                    image_abs_path = alt_path
                    image_rel_path = os.path.relpath(alt_path, root_dir)
                    break

            if width is None or height is None:
                raise FileNotFoundError(f"Could not read image: tried paths: {tried_paths}")

        # Compose conversation
        conversations = [
            {"from": "human", "value": format_human(task_text)},
            {"from": "gpt", "value": assistant_text}
        ]

        # Metadata (keep parsed actions for debugging)
        metadata = {key: v[key] for key in v if key not in ["task", "image"]}
        metadata["actions"] = actions

        chat_data.append({
            "id": int(k),
            "image": image_rel_path,
            "width": width,
            "height": height,
            "conversations": conversations,
            "metadata": metadata
        })

    # Write chat-format .jsonl
    with open(output_jsonl_path, 'w') as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write metafile .json
    if metafile_path and metafile_path != "/dev/null":
        annotation_abs_path = os.path.abspath(output_jsonl_path)
        metafile = {
            dataset_name: {
                "root": root_dir,
                "annotation": annotation_abs_path,
                "data_augment": data_augment,
                "max_dynamic_patch": max_dynamic_patch,
                "repeat_time": repeat_time,
                "length": len(chat_data)
            }
        }
        with open(metafile_path, 'w') as f:
            json.dump(metafile, f, indent=2)


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/home/vli/thangdd_workspace/datasets/OmniAct/")
    parser.add_argument("--prefix", type=str, default="internvlfm_")
    parser.add_argument("--dataset", type=str, default="omniact",
                        help="Dataset tag to include in filenames, e.g., 'omniact'.")
    parser.add_argument("--action-format", type=str, default="nl",
                        choices=["nl", "legacy", "concise"],
                        help="How to render actions: 'nl' (human-readable), 'legacy' or 'concise' (tokenized).")
    parser.add_argument("--grounding-format", type=str, default="double",
                        choices=["double", "single"],
                        help="Tokenizer grounding_format ([[x, y]] vs [x, y]) when action-format != 'nl'.")

    args = parser.parse_args()

    base_dir = args.base_dir
    prefix = args.prefix
    dataset_tag = (args.dataset or "omniact").strip()
    act_fmt = (args.action_format or "nl").strip().lower()

    splits = [("train", True), ("val", False), ("test", False)]
    for split_name, do_metafile in splits:
        split_json = os.path.join(base_dir, f"{split_name}.json")

        out_stem = f"{prefix}{dataset_tag}_{split_name}_{act_fmt}_chat"
        out_jsonl = os.path.join(base_dir, f"{out_stem}.jsonl")
        metafile = os.path.join(base_dir, f"{prefix}{dataset_tag}_{split_name}_{act_fmt}_metafile.json") if do_metafile else None
        annotation_name = f"{out_stem}.jsonl"
        dataset_name = f"{prefix}{dataset_tag}_{split_name}_{act_fmt}_chat"

        process_omniact(
            split_json_path=split_json,
            output_jsonl_path=out_jsonl,
            metafile_path=metafile if do_metafile else "/dev/null",
            root_dir=base_dir,
            annotation_name=annotation_name,
            dataset_name=dataset_name,
            action_format=args.action_format,
            grounding_format=args.grounding_format,
        )
