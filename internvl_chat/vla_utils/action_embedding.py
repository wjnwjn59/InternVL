from transformers import AutoTokenizer, AutoModel

class SpecialTokenUpdater:
    def __init__(self, model_path, verbose=True):
        self.model_path = model_path
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.embeddings = self.model.get_input_embeddings()

    def add_special_tokens(self, special_tokens):
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.embeddings = self.model.get_input_embeddings()  # Update embeddings reference
        if self.verbose:
            print(f"\nModel: {self.model_path}")
            print(f"Added {num_added_toks} new special tokens.")
            for token in special_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                weight = self.embeddings.weight[token_id].detach().cpu().numpy()
                print(f"Embedding for '{token}' (ID {token_id}): {weight}")
        return num_added_toks

if __name__ == "__main__":
    model_id = "/home/vli/thangdd_workspace/pretrained/OpenGVLab_InternVL2_5-2B"
    new_action_tokens = [
        "<|single_click|>",
        "<|double_click|>",
        "<|moveTo|>",
        "<|dragTo|>",
        "<|vscroll|>",
        "<|typewrite|>",
        "<|press|>",
        "<|hotKey|>",
    ]
    updater = SpecialTokenUpdater(model_id)
    updater.add_special_tokens(new_action_tokens)