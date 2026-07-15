import collections
import os

class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab if vocab is not None else {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges if merges is not None else {}
        self.vocab_size = len(self.vocab)
        if "\\n" in self.inverse_vocab:
            self.newline_id = self.inverse_vocab["\\n"]
        else:
            self.newline_id = None
        self.eos_id = self.newline_id

    @classmethod
    def from_file(cls, path: str, vocab_size: int) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if "\\n" not in text:
            text += "\\n"
            
        # 1. Base vocabulary (unique characters)
        chars = sorted(set(text))
        vocab = {i: ch for i, ch in enumerate(chars)}
        inverse_vocab = {ch: i for i, ch in enumerate(chars)}
        
        # 2. Convert text to list of ids
        words = text.split("\\n")
        word_ids = [[inverse_vocab[c] for c in word] for word in words if word]
        
        merges = {}
        current_vocab_size = len(vocab)
        
        while current_vocab_size < vocab_size:
            pairs = collections.Counter()
            for w in word_ids:
                for i in range(len(w) - 1):
                    pairs[(w[i], w[i+1])] += 1
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            
            new_id = current_vocab_size
            merges[best_pair] = new_id
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            
            new_word_ids = []
            for w in word_ids:
                i = 0
                new_w = []
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i+1]) == best_pair:
                        new_w.append(new_id)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                new_word_ids.append(new_w)
                
            word_ids = new_word_ids
            current_vocab_size += 1
            
        return cls(vocab, merges)
        
    def encode(self, s: str) -> list[int]:
        ids = [self.inverse_vocab.get(c, self.inverse_vocab.get('?', 0)) for c in s]
        
        while len(ids) >= 2:
            best_pair = None
            best_rank = float('inf')
            
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                if pair in self.merges:
                    rank = self.merges[pair] 
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                        
            if best_pair is None:
                break
                
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(self.merges[best_pair])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        return ids
        
    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)
        
    def get_state(self) -> dict:
        return {
            "vocab": self.vocab,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        
    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        vocab = {int(k) if isinstance(k, str) else k: v for k, v in state["vocab"].items()}
        merges = {}
        for k_str, v in state["merges"].items():
            k1, k2 = map(int, k_str.split(","))
            merges[(k1, k2)] = v
        return cls(vocab, merges)

if __name__ == "__main__":
    with open("dummy.txt", "w") as f:
        f.write("ali\\nahmet\\nayşe\\nfatma\\nmehmet\\n")
        
    tok = BPETokenizer.from_file("dummy.txt", 15)
    print("Vocab size:", tok.vocab_size)
    print("New vocab entries:", {k: v for k, v in tok.vocab.items() if k > 11}) # assuming <11 base chars
    
    enc = tok.encode("ahmet")
    print("Encoded 'ahmet':", enc)
    print("Decoded:", tok.decode(enc))
    
    assert tok.decode(enc) == "ahmet"
    state = tok.get_state()
    tok2 = BPETokenizer.from_state(state)
    assert tok2.encode("ahmet") == enc
    print("All tests passed.")
    os.remove("dummy.txt")
