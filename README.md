# small_GPT_hp
Using torch, built from scratc some small decoder only language models. Trained just on the texts from the Harry Potter novels.

This model can generate infinite Harry Potter-like text.

There are two models:
1. `small_GPT_hp_bpe_tokenizer.py` has a byte pair encoded tokenizer. Built the tokenizer off of similar structure as GPT-4 (thanks to Karpathy's minbpe)
   * This model produced the output `harry_potter_text_nano_GPT_all_books_bpe_tokenizer.txt` which at a glance is an improvement over the character level model's output. Similar word structure and word invention like that in the books.
2. `small_GPT_hp.py` has a small character level tokenizer. Simply mapped characters to integers.
   * This model produced the output `harry_potter_text_nano_GPT_char_level_tokenizer.txt` based on books 1 & 6 and `/harry_potter_text_nano_GPT_all_books.txt` was trained on all 7 novels. Looks kind of Harry Potterish.



The nanoGPT .txt files contain output.
