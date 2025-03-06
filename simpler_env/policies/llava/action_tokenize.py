
import numpy as np
from transformers import Qwen2Tokenizer

class ActionTokenizer:
    def __init__(
        self, tokenizer: Qwen2Tokenizer, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps them to tokens at the end of vocab.
        """
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Action tokens start near the end of the vocabulary
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray):
        """
        Clip & bin actions to discrete tokens.
        """
        action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
        discretized_action = np.digitize(action, self.bins)
        
        # Convert to token IDs: we place these tokens at the end of the vocab.
        action_token_ids = (self.tokenizer.vocab_size - discretized_action).tolist()

        # Decode to strings. Each action value corresponds to one token ID.
        # Depending on your tokenizer, this might return special or unknown tokens since these indices might not map
        # to known tokens. In a real scenario, you'd ensure these IDs map to custom tokens or placeholders.
        return self.tokenizer.convert_ids_to_tokens(action_token_ids)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.
        """
        action_token_ids = np.array([
        self.tokenizer.vocab_size if token is None else token for token in action_token_ids
        ])
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins


if __name__ == "__main__":
    # Given action as a string
    action_str = "[-0.12160591036081314, 0.19079051911830902, -0.1322687864303589, 0.261273056268692, -0.08133558183908463, 0.1104976087808609, 1.0]"

    # Parse the string into a numpy array
    action_str = action_str.strip("[]")
    action_values = np.array([float(val.strip()) for val in action_str.split(",")])

    # Load the Qwen2TokenizerFast tokenizer
    base_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")

    # Initialize ActionTokenizer
    action_tokenizer = ActionTokenizer(tokenizer=base_tokenizer, bins=256, min_action=-1.0, max_action=1.0)

    # Convert continuous actions to tokens
    tokens = action_tokenizer(action_values)
    print("Discretized tokens:", tokens)

    # Convert tokens back to actions
    token_ids = base_tokenizer.convert_tokens_to_ids(tokens)
    recovered_actions = action_tokenizer.decode_token_ids_to_actions(np.array(token_ids))
    print("Recovered actions (approx.):", recovered_actions)

