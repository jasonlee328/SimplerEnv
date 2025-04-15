
# # import numpy as np
# # from transformers import Qwen2Tokenizer

# # class ActionTokenizer:
# #     def __init__(
# #         self, tokenizer: Qwen2Tokenizer, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0
# #     ) -> None:
# #         """
# #         Discretizes continuous robot actions into N bins per dimension and maps them to tokens at the end of vocab.
# #         """
# #         self.tokenizer = tokenizer
# #         self.n_bins = bins
# #         self.min_action = min_action
# #         self.max_action = max_action

# #         # Create Uniform Bins + Compute Bin Centers
# #         self.bins = np.linspace(min_action, max_action, self.n_bins)
# #         self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

# #         # Action tokens start near the end of the vocabulary
# #         self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

# #     def __call__(self, action: np.ndarray):
# #         """
# #         Clip & bin actions to discrete tokens.
# #         """
# #         action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
# #         discretized_action = np.digitize(action, self.bins)
        
# #         # Convert to token IDs: we place these tokens at the end of the vocab.
# #         action_token_ids = (self.tokenizer.vocab_size - discretized_action).tolist()

# #         # Decode to strings. Each action value corresponds to one token ID.
# #         # Depending on your tokenizer, this might return special or unknown tokens since these indices might not map
# #         # to known tokens. In a real scenario, you'd ensure these IDs map to custom tokens or placeholders.
# #         return self.tokenizer.convert_ids_to_tokens(action_token_ids)

# #     def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
# #         """
# #         Returns continuous actions for discrete action token IDs.
# #         """
# #         action_token_ids = np.array([
# #         self.tokenizer.vocab_size if token is None else token for token in action_token_ids
# #         ])
# #         discretized_actions = self.tokenizer.vocab_size - action_token_ids
# #         discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
# #         return self.bin_centers[discretized_actions]

# #     @property
# #     def vocab_size(self) -> int:
# #         return self.n_bins


# # if __name__ == "__main__":
# #     # Given action as a string
# #     action_str = "[0.0, 0.03482142835855484, -0.1322687864303589, 0.261273056268692, -0.08133558183908463, 0.1104976087808609, 1.0]"

# #     # Parse the string into a numpy array
# #     action_str = action_str.strip("[]")
# #     action_values = np.array([float(val.strip()) for val in action_str.split(",")])

# #     # Load the Qwen2TokenizerFast tokenizer
# #     base_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")

# #     # Initialize ActionTokenizer
# #     action_tokenizer = ActionTokenizer(tokenizer=base_tokenizer, bins=256, min_action=-1.0, max_action=1.0)

# #     # Convert continuous actions to tokens
# #     tokens = action_tokenizer(action_values)
# #     print("Discretized tokens:", tokens)

# #     # Convert tokens back to actions
# #     token_ids = base_tokenizer.convert_tokens_to_ids(tokens)
# #     recovered_actions = action_tokenizer.decode_token_ids_to_actions(np.array(token_ids))
# #     print("Recovered actions (approx.):", recovered_actions)

import numpy as np
from transformers import Qwen2Tokenizer
import numpy as np

class ActionTokenizer:
    def __init__(self, tokenizer: Qwen2Tokenizer, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps them to tokens at the end of vocab.
        """
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        # Create Uniform Bins and compute Bin Centers.
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Action tokens are assumed to be placed near the end of the vocabulary.
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Converts token IDs back to continuous actions using the bin centers.
        """
        action_token_ids = np.array([
            self.tokenizer.vocab_size if token is None else token for token in action_token_ids
        ])
        # Reverse the token mapping: discretized index = vocab_size - token_id.
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        # Adjust indices to match the bin_centers array (which has one fewer element than bins).
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

if __name__ == "__main__":
    # Provided list of tokens.
    # tokens = ['\u00ea\u013b\u012d', '\u00ef\u0143\u00b2', '\u00ef\u00a7\u00a9', '\u00ea\u00b2\u013b', '\u00f0\u013f\u0137\u00b1', '\u00f0\u013f\u013a\u013c', '\u00f0\u0141\u0130\u0133']
    # tokens = ['êĻĭ', 'ïŃ²', 'ï§©', 'ê²Ļ', 'ðĿķ±', 'ðĿĺļ', 'ðŁİĳ']
    # tokens = ['\u00f0\u013f\u013b\u013e', '\u00dd\u00a5', '\u00dd\u00a5', '\u00e0\u00a5\u00b1', '\u00f0\u0141\u0131\u0129', '\u00c8\u00b2', '\u00e2\u00bd\u0139']
    # tokens = ['\u00f0\u013f\u013b\u013e', '\u00dd\u00a5', '\u00dd\u00a5', '\u00e0\u00a5\u00b1', '\u00f0\u0141\u0131\u0129', '\u00c8\u00b2', '\u00e2\u00bd\u0139']
    tokens =["\u00e2\u00a4\u00a6","\u00e1\u00b8\u00bb","\u00dd\u00a5","\u00d4\u0133","\u00f0\u013f\u0135\u0137","\u00ef\u00a8\u0124","\u00f0\u0141\u0130\u0133"]
    # tokens = ['\u00ec\u00b3\u0127', '\u00e1\u0142\u0126', '\u00ef\u00a8\u0124', '\u00e1\u00a8\u0123', '\u00ea\u00b3\u0124', '\u00f0\u013f\u013a\u013c', '\u00f0\u0141\u0130\u0133']
    # tokens = ['\u00ed\u0137\u00ae', '\u00ed\u0135\u00ae', '\u00ef\u00a8\u0124', '\u00e2\u013c\u00a3', '\u00f0\u0138\u00a5\u00a8', '\u00f0\u013f\u013a\u013c', '\u00f0\u0141\u0130\u0133']
    # tokens = ['<ACTION_0>','<ACTION_0>','<ACTION_97>','<ACTION_0>','<ACTION_255>','<ACTION_0>','<ACTION_0>']
    
    # Load the pretrained Qwen2Tokenizer.
    base_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
    
    # Initialize our ActionTokenizer.
    action_tokenizer = ActionTokenizer(tokenizer=base_tokenizer, bins=256, min_action=-1.0, max_action=1.0)
    
    # Convert the tokens to token IDs.
    token_ids = base_tokenizer.convert_tokens_to_ids(tokens)
    print("Token IDs:", token_ids)
    
    # Decode token IDs back to continuous actions.
    recovered_actions = action_tokenizer.decode_token_ids_to_actions(np.array(token_ids))
    print("Recovered continuous actions:", recovered_actions)


# import numpy as np

# class ActionTokenizer:
#     def __init__(self, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0) -> None:
#         """
#         Maps tokens of the form <ACTION_i> (where i is between 0 and bins-1)
#         to continuous actions. <ACTION_0> maps to min_action (e.g. -1.0),
#         <ACTION_255> maps to max_action (e.g. 1.0), and tokens in between are linearly spaced.
#         """
#         self.n_bins = bins
#         self.min_action = min_action
#         self.max_action = max_action
#         # Create a linearly spaced array of continuous values for each bin index.
#         self.bin_values = np.linspace(min_action, max_action, bins)

#     def decode_tokens_to_actions(self, action_tokens: list) -> np.ndarray:
#         """
#         Converts a list of tokens of the form <ACTION_i> into their corresponding continuous actions.
#         """
#         bin_indices = []
#         for token in action_tokens:
#             if token.startswith("<ACTION_") and token.endswith(">"):
#                 try:
#                     idx = int(token[len("<ACTION_"):-1])
#                 except ValueError:
#                     idx = 0  # Default to 0 if parsing fails
#             else:
#                 idx = 0  # Default index if token format is unexpected
#             bin_indices.append(idx)
#         # Ensure indices are within the valid range.
#         bin_indices = np.clip(np.array(bin_indices), 0, self.n_bins - 1)
#         return self.bin_values[bin_indices]

# if __name__ == "__main__":
#     # Example tokens.
#     tokens = ['<ACTION_0>', '<ACTION_0>', '<ACTION_97>', '<ACTION_0>', '<ACTION_255>', '<ACTION_0>', '<ACTION_0>']
    
#     # Initialize our ActionTokenizer.
#     action_tokenizer = ActionTokenizer(bins=256, min_action=-1.0, max_action=1.0)
    
#     # Decode tokens to continuous actions.
#     recovered_actions = action_tokenizer.decode_tokens_to_actions(tokens)
#     print("Recovered continuous actions:", recovered_actions)
