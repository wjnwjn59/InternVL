from transformers import AutoTokenizer, AutoModel

class SpecialTokenUpdater:
    ACTION_TOKENS = [
        "<|SINGLE_CLICK|>",
        "<|DOUBLE_CLICK|>",
        "<|BUTTON_LEFT|>",
        "<|BUTTON_RIGHT|>",
        "<|COORD_START|>",
        "<|COORD_END|>",
        "<|MOVE_TO|>",
        "<|DRAG_TO|>",
        "<|VSCROLL|>",
        "<|HSCROLL|>",
        "<|WRITE|>",
        "<|PRESS|>",
        "<|HOTKEY|>",
        "<|KEY_A|>",
        "<|KEY_B|>",
        "<|KEY_C|>",
        "<|KEY_D|>",
        "<|KEY_E|>",
        "<|KEY_F|>",
        "<|KEY_G|>",
        "<|KEY_H|>",
        "<|KEY_I|>",
        "<|KEY_J|>",
        "<|KEY_K|>",
        "<|KEY_L|>",
        "<|KEY_M|>",
        "<|KEY_N|>",
        "<|KEY_O|>",
        "<|KEY_P|>",
        "<|KEY_Q|>",
        "<|KEY_R|>",
        "<|KEY_S|>",
        "<|KEY_T|>",
        "<|KEY_U|>",
        "<|KEY_V|>",
        "<|KEY_W|>",
        "<|KEY_X|>",
        "<|KEY_Y|>",
        "<|KEY_Z|>",
        "<|KEY_ENTER|>",
        "<|KEY_ESC|>",
        "<|KEY_TAB|>",
        "<|KEY_BACKSPACE|>",
        "<|KEY_SPACE|>",
        "<|KEY_CAPSLOCK|>",
        "<|KEY_LEFT|>",
        "<|KEY_UP|>",
        "<|KEY_RIGHT|>",
        "<|KEY_DOWN|>",
        "<|KEY_ALT|>",
        "<|KEY_CTRL|>",
        "<|KEY_SHIFT|>",
        "<|KEY_CMD|>",
        "<|KEY_+|>",
        "<|KEY_-|>",
        "<|SEPARATOR|>",
        "<|None|>"
    ]

    def __init__(self, tokenizer, verbose=True):
        self.tokenizer = tokenizer
        self.verbose = verbose

    def add_special_tokens(self, special_tokens):
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        if self.verbose:
            print(f"Added {num_added_toks} new special tokens.")
            for token in special_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                print(f"Token '{token}' ID: {token_id}")
        return num_added_toks

    def add_action_tokens(self):
        return self.add_special_tokens(self.ACTION_TOKENS)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/home/vli/thangdd_workspace/pretrained/OpenGVLab_InternVL2_5-2B", trust_remote_code=True)
    updater = SpecialTokenUpdater(tokenizer)
    updater.add_action_tokens()