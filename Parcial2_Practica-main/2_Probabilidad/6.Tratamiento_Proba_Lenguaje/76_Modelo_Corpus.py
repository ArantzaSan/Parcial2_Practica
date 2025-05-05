import re
from collections import Counter
import os

class CorpusModel:
    def __init__(self, dir=None, raw_text=None):
        self.tokens = []; self.vocab = set(); self.freq = Counter(); self.length = 0
        self._load_and_process(dir, raw_text)

    def _load_and_process(self, dir, raw_text):
        texts = []
        if dir:
            try:
                texts = [open(os.path.join(dir, f), 'r', encoding='utf-8').read() for f in os.listdir(dir) if f.endswith(".txt")]
            except FileNotFoundError: print(f"Dir '{dir}' not found"); return
        elif raw_text: texts.append(raw_text)
        else: print("No dir or text provided"); return

        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            self.tokens.extend(tokens); self.vocab.update(tokens); self.freq.update(tokens); self.length += len(tokens)

    def get_tokens(self): return self.tokens
    def get_vocab(self): return self.vocab
    def get_freq(self): return self.freq
    def freq_word(self, word): return self.freq.get(word, 0)
    def prob_word(self, word): return self.freq_word(word) / self.length if self.length else 0
    def stats(self): return {"size": self.length, "vocab_size": len(self.vocab), "most_common": self.freq.most_common(1)[0] if self.freq else None, "least_common": self.freq.most_common()[-5:] if len(self.freq) > 5 else self.freq.most_common()}

if __name__ == "__main__":
    if not os.path.exists("ej"): os.makedirs("ej")
    with open("ej/t1.txt", "w", encoding='utf-8') as f: f.write("Este es el primer texto.")
    with open("ej/t2.txt", "w", encoding='utf-8') as f: f.write("El segundo texto también es.")

    m = CorpusModel(dir="ej")
    print("Tokens:", m.get_tokens()[:5])
    print("Vocab:", list(m.get_vocab())[:5])
    print("Freq:", m.get_freq().most_common(5))
    print("Freq 'el':", m.freq_word("el"))
    print("Prob 'el':", m.prob_word("el"))
    print("Stats:", m.stats())

    rt = "Texto crudo ejemplo."
    mc = CorpusModel(raw_text=rt)
    print("\n--- Crudo ---")
    print("Tokens:", mc.get_tokens())
    print("Vocab:", mc.get_vocab())
    print("Freq:", mc.get_freq())
    print("Stats:", mc.stats())

    import shutil
    if os.path.exists("ej"): shutil.rmtree("ej")
