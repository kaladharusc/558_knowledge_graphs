# !  # /usr/bin/python3

import sys


def parse_and_return_chunks(line):
    import re
    regex = r'(<\w+?>)(.*?)(<\/\w+?>)'
    replace_reg = r'[<>]'
    matches = re.finditer(regex, line)
    res = []

    def bio(x):
        if x[0] == 0:
            return "B-{0}".format(x[1])
        else:
            return "I-{0}".format(x[1])

    for match in matches:
        words = match.group(2).split(" ")

        chunk_name = match.group(1)
        chunk_name = re.sub(replace_reg, "", chunk_name)
        chunks = [chunk_name]*len(words)
        chunks = list(map(bio, enumerate(chunks)))
        if "." in words[-1]:
            words[-1] = words[-1][:-1]
            words.append(".")
            chunks.append(".")
        words = list(zip(words, chunks))
        res.extend(map(lambda x: " ".join(x), words))
        res.append("")
    return "\n".join(res)


def create_chunks_file_from_input(input_file):
    with open(input_file, "r") as read_file, open("word_chunks.txt", "w+") as write_file:
        for line in read_file:
            write_file.write((parse_and_return_chunks(line)))
            write_file.write("\n")


def create_feature_file(input_file):
    pass


def train_model():
    pass


def predict():
    pass


if __name__ == "__main__":
    create_chunks_file_from_input("train-ucla.txt")
