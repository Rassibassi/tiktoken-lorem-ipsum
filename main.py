from pathlib import Path

import tiktoken
from scipy.optimize import root_scalar

text = Path("150-paragraphs.txt").read_text()
encoder = tiktoken.get_encoding("cl100k_base")


def token_len(text):
    return len(encoder.encode(text))


def generate_file_with_tokens(
    target_tokens, input_file="150-paragraphs.txt", output_file_prefix="lorem"
):
    text = Path(input_file).read_text()
    current_length = token_len(text)

    multiplier = target_tokens // current_length + 1

    new_text = text * multiplier

    max_reduction = len(new_text)

    def reduce_text(n):
        return token_len(new_text[: -int(n)]) - target_tokens

    result = root_scalar(reduce_text, bracket=[1, max_reduction], method="brentq")
    new_text = new_text[: -int(result.root)]

    final_length = token_len(new_text)
    assert final_length == target_tokens

    output_file = f"{output_file_prefix}_{target_tokens:08d}.txt"
    Path(output_file).write_text(new_text)
    print(f"File '{output_file}' with {target_tokens} tokens created.")


def main():
    ns = []
    for n in range(10, 22):
        ns.append(2**n)

    for n in range(0, 12):
        ns.append(2**n * 1000)

    for n in ns:
        generate_file_with_tokens(n)


main()
