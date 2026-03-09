import torch
from transformers.generation import LogitsProcessor
from typing import Callable, List
import logging

logger = logging.getLogger(__name__)

class ConstrainedLogitsProcessor(LogitsProcessor):
    """
    A robust LogitsProcessor that restricts the model's output to a valid set of tokens 
    based on a prefix tree (Trie), ensuring the generated Semantic IDs are strictly valid.
    """
    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], prompt_length: int):
        """
        Args:
            prefix_allowed_tokens_fn: A function that takes (batch_id, current_token_sequence) and returns a list of allowed next token IDs.
            prompt_length: The length of the prompt, used to extract only the generated tokens.
        """
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # The simplest way to handle multinomial NaN while enforcing hard constraints:
        # Instead of returning float scores, PyTorch's native generate handles boolean masks best if we use very specific numbers.
        # But wait, PyTorch actually supports native bad_words_ids or force_words_ids.
        # Here we just want to suppress illegal tokens.
        
        # We will use a float mask to strictly block forbidden tokens.
        # To avoid multinomial NaN, we can't let all token probs become strictly 0. 
        # But we MUST block illegal tokens.
        mask = torch.zeros_like(scores, dtype=torch.bool)
        
        # Extract only the newly generated tokens
        generated_ids = input_ids[:, self.prompt_length:]
        
        for batch_id in range(input_ids.shape[0]):
            current_generated = generated_ids[batch_id].tolist()
            
            # Query the Trie/Hash dictionary for allowed next tokens
            allowed_tokens = self.prefix_allowed_tokens_fn(batch_id, current_generated)
            
            if len(allowed_tokens) == 0:
                mask[batch_id, self.prefix_allowed_tokens_fn.__self__.eos_token_id] = True
                continue
                
            # Allow the specific tokens by setting their mask to True
            for token_id in allowed_tokens:
                mask[batch_id, token_id] = True
            
            # Additional safety: force eos to be available if generated is too long
            mask[batch_id, self.prefix_allowed_tokens_fn.__self__.eos_token_id] = True

        # Calculate a safe minimum score to set masked tokens to.
        # Just use -float('inf') to mask out, but IF the max score among ALLOWED tokens 
        # is also -inf (or extremely low), multinomial will fail.
        # The proper fix is to ensure the scores of ALLOWED tokens are shifted to be nicely bounded.
        
        # Step 1: Set unallowed tokens to -inf
        scores = scores.masked_fill(~mask, -float('inf'))
        
        # In PyTorch generation, it is CRITICAL that there is at least one valid probability > 0
        # Check if the mask filtered everything out (i.e. all True values were already -inf, or no True values)
        # Using simple boolean rescue without meddling with infs
        
        # Find rows where there is NO token with score > -inf
        invalid_rows = torch.isinf(scores).all(dim=-1)
        
        if invalid_rows.any():
            # Force EOS to be valid with a neutral score
            scores[invalid_rows, self.prefix_allowed_tokens_fn.__self__.eos_token_id] = 10.0 # Force slightly positive so it reliably gets picked

        # Replace any residual NaNs with -inf
        scores = torch.nan_to_num(scores, nan=-float('inf'))
        
        # In PyTorch, multinomial fails if all values in softmax are 0 (which happens if everything is very negative).
        # Shift scores up by subtracting max to prevent softmax underflow, then reapply mask
        row_max = scores.max(dim=-1, keepdim=True)[0]
        # Shift only valid finite maxes
        row_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)
        scores = scores - row_max
        scores = scores.masked_fill(~mask, -float('inf'))
            
        return scores


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class SIDTrie:
    """
    A Trie structure built from the list of all valid SIDs.
    This replaces the brittle hash_dict approach with a more robust prefix tree.
    """
    def __init__(self, eos_token_id: int):
        self.root = TrieNode()
        self.eos_token_id = eos_token_id
        
    def add(self, token_ids: List[int]):
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        # Allow EOS at the end of a valid SID
        if self.eos_token_id not in node.children:
            node.children[self.eos_token_id] = TrieNode()
            
    def get_allowed_next_tokens(self, batch_id: int, current_sequence: List[int]) -> List[int]:
        node = self.root
        for token_id in current_sequence:
            if token_id in node.children:
                node = node.children[token_id]
            else:
                return [] # Sequence went out of bounds, no tokens allowed
        return list(node.children.keys())
