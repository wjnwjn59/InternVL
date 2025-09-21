from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import re


class ActionTokenizer:
    """
    Configuration:
      - action_format ∈ {"legacy", "concise"}
          * legacy  : <|SEPARATOR|> fields, coords as <|COORD_START|>[[x, y]]<|COORD_END|> or single-bracket.
          * concise : compact tokens (no <|SEPARATOR|> / <|None|>), e.g.:
                      <|DOUBLE_CLICK|><|BUTTON_LEFT|><|COORD_START|>[[46, 440]]<|COORD_END|>
                      <|HOTKEY|><|KEY_CTRL|><|KEY_C|>
                      <|WRITE|>
                      <|VSCROLL|>5
                      <|HSCROLL|>-120
      - grounding_format ∈ {"double", "single"}

    Notes:
      - WRITE: omit its text payload from the encoded string.
      - Value items map to <|KEY_*|> when possible (letters a–z, 'ctrl', 'enter', '+', '-', '=', etc.).
      - For VSCROLL/HSCROLL, the signed magnitude immediately follows the token in concise mode.
    """

    def __init__(self, *, action_format: str = "legacy", grounding_format: str = "double") -> None:
        af = action_format.lower()
        gf = grounding_format.lower()
        if af not in {"legacy", "concise"}:
            raise ValueError("action_format must be 'legacy' or 'concise'")
        if gf not in {"double", "single"}:
            raise ValueError("grounding_format must be 'double' or 'single'")
        self.action_format = af
        self.grounding_format = gf

        # Literal tokens
        self.SEP = "<|SEPARATOR|>"
        self.T_NONE = "<|None|>"
        self.T_COORD_START = "<|COORD_START|>"
        self.T_COORD_END = "<|COORD_END|>"

        # Actions
        self.T_SINGLE_CLICK = "<|SINGLE_CLICK|>"
        self.T_DOUBLE_CLICK = "<|DOUBLE_CLICK|>"
        self.T_MOVE_TO = "<|MOVE_TO|>"
        self.T_DRAG_TO = "<|DRAG_TO|>"
        self.T_VSCROLL = "<|VSCROLL|>"
        self.T_HSCROLL = "<|HSCROLL|>"
        self.T_WRITE = "<|WRITE|>"
        self.T_PRESS = "<|PRESS|>"
        self.T_HOTKEY = "<|HOTKEY|>"

        # Buttons
        self.T_BUTTON_LEFT = "<|BUTTON_LEFT|>"
        self.T_BUTTON_RIGHT = "<|BUTTON_RIGHT|>"

        # Keys (A–Z)
        self.key_to_token: Dict[str, str] = {}
        self.token_to_key: Dict[str, str] = {}
        for i in range(26):
            ch = chr(ord("a") + i)
            tok = f"<|KEY_{ch.upper()}|>"
            self.key_to_token[ch] = tok
            self.token_to_key[tok] = ch

        # Special keys / modifiers
        specials = {
            "enter": "<|KEY_ENTER|>",
            "esc": "<|KEY_ESC|>",
            "tab": "<|KEY_TAB|>",
            "backspace": "<|KEY_BACKSPACE|>",
            "space": "<|KEY_SPACE|>",
            "capslock": "<|KEY_CAPSLOCK|>",
            "left": "<|KEY_LEFT|>",
            "up": "<|KEY_UP|>",
            "right": "<|KEY_RIGHT|>",
            "down": "<|KEY_DOWN|>",
            "alt": "<|KEY_ALT|>",
            "ctrl": "<|KEY_CTRL|>",
            "shift": "<|KEY_SHIFT|>",
            "cmd": "<|KEY_CMD|>",
        }
        for k, tok in specials.items():
            self.key_to_token[k] = tok
            self.token_to_key[tok] = k

        # Symbol keys (extend as needed)
        symbols = {
            "+": "<|KEY_+|>",
            "-": "<|KEY_-|>",
        }
        for k, tok in symbols.items():
            self.key_to_token[k] = tok
            self.token_to_key[tok] = k

        # Vocabulary: semantic → token
        self.vla2tok: Dict[str, str] = {
            "single_click": self.T_SINGLE_CLICK,
            "double_click": self.T_DOUBLE_CLICK,
            "moveto": self.T_MOVE_TO, "move_to": self.T_MOVE_TO,
            "dragto": self.T_DRAG_TO, "drag_to": self.T_DRAG_TO,
            "vscroll": self.T_VSCROLL,
            "hscroll": self.T_HSCROLL,
            "write": self.T_WRITE,
            "press": self.T_PRESS,
            "hotkey": self.T_HOTKEY,
            "left": self.T_BUTTON_LEFT,
            "right": self.T_BUTTON_RIGHT,
        }
        self.tok2vla: Dict[str, str] = {v: k for k, v in self.vla2tok.items()}

        self._token_re = re.compile(r"<\|[A-Z_]+?\|>")

    # ───────────────────────── helpers ───────────────────────── #
    @staticmethod
    def _canon(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        return s.strip().lower().replace(" ", "").replace("-", "_")

    def _is_none_token(self, s: Optional[str]) -> bool:
        return s == self.T_NONE

    def _none_or_str(self, v: Any) -> str:
        return self.T_NONE if (v is None) else str(v)

    def _map_semantic(self, w: Optional[str]) -> str:
        if w is None:
            return self.T_NONE
        c = self._canon(w)
        return self.vla2tok.get(c, w)

    def _unmap_semantic(self, w: str) -> Optional[str]:
        if self._is_none_token(w):
            return None
        return self.tok2vla.get(w, w)

    @staticmethod
    def _clean_key_literal(s: str) -> str:
        t = s.strip()
        t = t.replace(r"\\'", "'").replace(r'\\"', '"')
        t = t.replace(r"\'", "'").replace(r'\"', '"')
        if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
            t = t[1:-1]
        t = t.strip('\'"').strip()
        return t

    def _map_value_item(self, v: Any) -> str:
        if v is None:
            return self.T_NONE
        if isinstance(v, str):
            s = self._clean_key_literal(v)
            if s.startswith("<|") and s.endswith("|>"):
                return s
            c = s.lower()
            if len(c) == 1 and "a" <= c <= "z":
                return self.key_to_token.get(c, s)
            return self.key_to_token.get(c, s)  # ctrl, enter, '+', '-', '=', etc.
        return str(v)

    def _unmap_value_item(self, v: str) -> Optional[str]:
        if self._is_none_token(v):
            return None
        if v in self.token_to_key:
            return self.token_to_key[v]
        return v

    def _to_int_or_none(self, s: str) -> Optional[int]:
        if self._is_none_token(s):
            return None
        s2 = s.strip()
        if s2.lower() == "none":
            return None
        try:
            return int(s2)
        except Exception as e:
            raise ValueError(f"Expected integer or <|None|>, got {s!r}") from e

    # Grounding helpers
    def _coord_text(self, x: int, y: int) -> str:
        if self.grounding_format == "double":
            inner = f"[[{x}, {y}]]"
        else:
            inner = f"[{x}, {y}]"
        return f"{self.T_COORD_START}{inner}{self.T_COORD_END}"

    def _parse_coord_block(self, coord_field: str) -> Tuple[Optional[int], Optional[int]]:
        if coord_field == self.T_NONE:
            return None, None
        if not (coord_field.startswith(self.T_COORD_START) and coord_field.endswith(self.T_COORD_END)):
            raise ValueError("Malformed coordinate: missing <|COORD_START|>/ <|COORD_END|>")
        inner = coord_field[len(self.T_COORD_START) : -len(self.T_COORD_END)].strip()
        if inner.startswith("[[") and inner.endswith("]]"):
            inner = inner[2:-2]
        elif inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1]
        else:
            raise ValueError("Malformed coordinate: expected [[x, y]] or [x, y]")
        xs, ys = [t.strip() for t in inner.split(",")]
        return int(xs), int(ys)

    # ──────────────────────── legacy (SEPARATOR) ──────────────────────── #
    def _enc_legacy(self, act: Dict[str, Any]) -> str:
        coord = self.T_NONE
        if act.get("x") is not None and act.get("y") is not None:
            coord = self._coord_text(int(act["x"]), int(act["y"]))
        sc = self._none_or_str(act.get("n_scrolls", None))
        parts = [
            self._map_semantic(act["action"]),
            self._map_semantic(act.get("button")),
            coord,
            sc,
        ]
        out = self.SEP.join(parts) + self.SEP
        # omit values for WRITE
        if self._canon(act.get("action")) != "write":
            for v in (act.get("value") or []):
                out += self._map_value_item(v) + self.SEP
        return out

    def _dec_legacy(self, s: str) -> Dict[str, Any]:
        p = s.split(self.SEP)
        if p and p[-1] == "":
            p = p[:-1]
        if len(p) < 4:
            raise ValueError("legacy malformed: needs ≥4 fields")
        act, btn, coord, sc, *vals = p
        x = y = None
        if coord != self.T_NONE:
            x, y = self._parse_coord_block(coord)
        return {
            "action": self._unmap_semantic(act),
            "button": self._unmap_semantic(btn),
            "x": x,
            "y": y,
            "n_scrolls": self._to_int_or_none(sc),
            "value": [self._unmap_value_item(v) for v in vals if v not in {"", self.T_NONE}],
        }

    # ───────────────────────── concise (compact) ───────────────────────── #
    def _enc_concise(self, act: Dict[str, Any]) -> str:
        s = []
        s.append(self._map_semantic(act["action"]))
        btn_tok = self._map_semantic(act.get("button"))
        if btn_tok != self.T_NONE:
            s.append(btn_tok)
        if act.get("x") is not None and act.get("y") is not None:
            s.append(self._coord_text(int(act["x"]), int(act["y"])))
        if self._canon(act.get("action")) in {"vscroll", "hscroll"} and act.get("n_scrolls") is not None:
            s.append(str(int(act["n_scrolls"])))
        if self._canon(act.get("action")) != "write":
            for v in (act.get("value") or []):
                s.append(self._map_value_item(v))
        return "".join(s)

    def _dec_concise(self, s: str) -> Dict[str, Any]:
        pos = 0
        n = len(s)
        action: Optional[str] = None
        button: Optional[str] = None
        x = y = None
        n_scrolls: Optional[int] = None
        values: List[str] = []

        def next_token_at(i: int) -> Optional[re.Match]:
            return self._token_re.match(s, i)

        while pos < n:
            m = next_token_at(pos)
            if not m:
                # read signed magnitude after VSCROLL/HSCROLL
                if n_scrolls is None and action in {"vscroll", "hscroll"}:
                    num_m = re.match(r"[+-]?\d+", s[pos:])
                    if num_m:
                        n_scrolls = int(num_m.group(0))
                        pos += len(num_m.group(0))
                        continue
                if s[pos].isspace():
                    pos += 1
                    continue
                raise ValueError(f"Unexpected content at position {pos}: {s[pos:pos+16]!r}")
            tok = m.group(0)
            pos = m.end()

            if tok == self.T_COORD_START:
                end_idx = s.find(self.T_COORD_END, pos)
                if end_idx == -1:
                    raise ValueError("Missing <|COORD_END|> in concise coordinate block")
                coord_inner = s[pos:end_idx]
                coord_field = f"{self.T_COORD_START}{coord_inner}{self.T_COORD_END}"
                x, y = self._parse_coord_block(coord_field)
                pos = end_idx + len(self.T_COORD_END)
                continue

            if tok == self.SEP or tok == self.T_NONE:
                continue

            sem = self.tok2vla.get(tok)
            if sem in {"single_click", "double_click", "moveto", "dragto", "vscroll", "hscroll", "write", "press", "hotkey"}:
                action = sem
                continue

            if sem in {"left", "right"}:
                button = sem
                continue

            if tok in self.token_to_key:
                values.append(self.token_to_key[tok])
                continue

            values.append(tok)

        return {
            "action": action,
            "button": button,
            "x": x,
            "y": y,
            "n_scrolls": n_scrolls,
            "value": values,
        }

    def encode(self, action: dict) -> str:
        if self.action_format == "legacy":
            return self._enc_legacy(action)
        return self._enc_concise(action)

    def decode(self, sentence: str) -> dict:
        if self.action_format == "legacy":
            return self._dec_legacy(sentence)
        return self._dec_concise(sentence)


def compute_action_evaluation(gt_action: dict, pred_action: dict) -> dict:
    evaluation = {}
    evaluation["action_score"] = 1 if gt_action.get("action") == pred_action.get("action") else 0
    evaluation["button_score"] = 1 if gt_action.get("button") == pred_action.get("button") else 0

    x_gt, y_gt = gt_action.get("x"), gt_action.get("y")
    x_pred, y_pred = pred_action.get("x"), pred_action.get("y")
    if x_gt is None or y_gt is None or x_pred is None or y_pred is None:
        mae = 0
        coordinate_proximity = 0
    else:
        x_error = abs(x_gt - x_pred)
        y_error = abs(y_gt - y_pred)
        mae = (x_error + y_error) / 2.0
        if x_error == 0 and y_error == 0:
            coordinate_proximity = 1
        elif x_error <= 10 and y_error <= 10:
            coordinate_proximity = 0.5
        else:
            coordinate_proximity = 0
    evaluation["coordinate_mae"] = mae
    evaluation["coordinate_proximity"] = coordinate_proximity

    gt_scroll = gt_action.get("n_scrolls")
    pred_scroll = pred_action.get("n_scrolls")
    if gt_scroll is None and pred_scroll is None:
        scroll_score = 1
    elif gt_scroll is not None and pred_scroll is not None:
        if gt_scroll == pred_scroll:
            scroll_score = 1
        elif abs(gt_scroll - pred_scroll) == 1:
            scroll_score = 0.5
        else:
            scroll_score = 0
    else:
        scroll_score = 0
    evaluation["scroll_score"] = scroll_score

    gt_value = gt_action.get("value") or []
    pred_value = pred_action.get("value") or []
    evaluation["value_score"] = 1 if gt_value == pred_value else 0

    scores = [
        evaluation["action_score"],
        evaluation["button_score"],
        evaluation["coordinate_proximity"],
        evaluation["scroll_score"],
        evaluation["value_score"],
    ]
    evaluation["average_score"] = sum(scores) / len(scores)
    return evaluation

# ─────────────────────────── demo / quick tests ─────────────────────────── #
if __name__ == "__main__":
    # Example from your prompt
    action = dict(
        action="double_click",
        button="left",
        x=46,
        y=440,
        n_scrolls=None,
        value=[],
        before_frame="images/action_1_before.png",
        after_frame="images/action_1_after.png",
    )

    # Legacy + double-bracket grounding
    tok_legacy = ActionTokenizer(action_format="legacy", grounding_format="double")
    enc_legacy = tok_legacy.encode(action)
    print(enc_legacy)
    print(tok_legacy.decode(enc_legacy))

    # Concise + double-bracket grounding
    tok_concise = ActionTokenizer(action_format="concise", grounding_format="double")
    enc_concise = tok_concise.encode(action)
    print(enc_concise)
    print(tok_concise.decode(enc_concise))

    # Concise HOTKEY example
    hotkey = dict(action="hotkey", button=None, x=None, y=None, n_scrolls=None, value=["ctrl", "c"])
    print(tok_concise.encode(hotkey))
    print(tok_concise.decode(tok_concise.encode(hotkey)))

    # Concise VSCROLL with magnitude
    scroll = dict(action="vscroll", button=None, x=None, y=None, n_scrolls=5, value=[])
    print(tok_concise.encode(scroll))
    print(tok_concise.decode(tok_concise.encode(scroll)))

    # Concise WRITE should NOT include text payload
    write_ex = dict(action="write", button=None, x=None, y=None, n_scrolls=None, value=["hello world"])
    print(tok_concise.encode(write_ex))  # → <|WRITE|>
    print(tok_concise.decode(tok_concise.encode(write_ex)))
