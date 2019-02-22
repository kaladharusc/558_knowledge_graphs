# !  # /usr/bin/python3

import sys
import optparse
import pycrfsuite
import crfutils


def predict():
    pass


def train(feature_file):
    trainer = pycrfsuite.Trainer(verbose=True)
    X_train, Y_train = get_feature_from_file(feature_file)
    for xseq, yseq in zip(X_train, Y_train):
        trainer.append(xseq, yseq)

    # Set the parameters of the model
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,

        # coefficient for L2 penalty
        'c2': 0.01,

        # maximum number of iterations
        'max_iterations': 200,

        # whether to include transitions that
        # are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train('crf.model')


def get_feature_from_file(file_name):
    X, Y = [], []
    with open(file_name,  "r") as f:
        for line in f:
            if len(line) == 0:
                continue
            arr = line.strip("\n").split("\t")
            X.append([arr[1:]])
            Y.append([arr[0]])
    return (X, Y)


def create_crf_suite_text(file_name):
    separator = ' '
    fields = 'w lemma pos y'
    templates = (
        (('w', -2), ),
        (('w', -1), ),
        (('w',  0), ),
        (('w',  1), ),
        (('w',  2), ),
        (('w', -1), ('w',  0)),
        (('w',  0), ('w',  1)),
        (('lemma', 0),),
        (('pos', -2), ),
        (('pos', -1), ),
        (('pos',  0), ),
        (('pos',  1), ),
        (('pos',  2), ),
        (('pos', -2), ('pos', -1)),
        (('pos', -1), ('pos',  0)),
        (('pos',  0), ('pos',  1)),
        (('pos',  1), ('pos',  2)),
        (('pos', -2), ('pos', -1), ('pos',  0)),
        (('pos', -1), ('pos',  0), ('pos',  1)),
        (('pos',  0), ('pos',  1), ('pos',  2)),
    )

    def feature_extractor(X):
        crfutils.apply_templates(X, templates)
        if X:
            X[0]['F'].append('__BOS__')     # BOS feature
            X[-1]['F'].append('__EOS__')    # EOS feature

    fo = open("new_crf_text.txt", "w+")
    # Parse the command-line arguments.
    parser = optparse.OptionParser(usage="""usage: %prog [options]
        This utility reads a data set from STDIN, and outputs attributes to STDOUT.
        Each line of a data set must consist of field values separated by SEPARATOR
        characters. The names and order of field values can be specified by -f option.
        The separator character can be specified with -s option. Instead of outputting
        attributes, this utility tags the input data when a model file is specified by
        -t option (CRFsuite Python module must be installed).""")
    parser.add_option(
        '-t', dest='model',
        help='tag the input using the model (requires "crfsuite" module)'
    )
    parser.add_option(
        '-f', dest='fields', default=fields,
        help='specify field names of input data [default: "%default"]'
    )
    parser.add_option(
        '-s', dest='separator', default=separator,
        help='specify the separator of columns of input data [default: "%default"]'
    )
    (options, args) = parser.parse_args()

    # The fields of input: ('w', 'pos', 'y) by default.
    F = options.fields.split(' ')

    # The generator function readiter() reads a sequence from a
    for X in crfutils.readiter(open(file_name, "r"), F, options.separator):
        feature_extractor(X)
        crfutils.output_features(fo, X, 'y')

    # Train
    # train(fo)
    # import crfsuite
    # tagger = crfsuite.Tagger()
    # tagger.open("trained.model")

    # # For each sequence from STDIN.
    # for X in crfutils.readiter(fo, F, options.separator):
    #     # Obtain features.
    #     feature_extractor(X)
    #     xseq = crfutils.to_crfsuite(X)
    #     yseq = tagger.tag(xseq)
    #     for t in range(len(X)):
    #         v = X[t]
    #         fo.write('\t'.join([v[f] for f in F]))
    #         fo.write('\t%s\n' % yseq[t])
    #     fo.write('\n')


import spacy
nlp = spacy.load('en_core_web_sm')


def parse_and_return_chunks(line):
    import re

    regex = r'(<\w+?>)(.*?)(<\/\w+?>)'
    replace_reg = r'[<>]'
    matches = re.finditer(regex, line)

    def pos_tag(sentence, chunk_name):
        doc = nlp(sentence)
        res = []

        for token in doc:
            # res.append([token.text, token.lemma_, token.pos_, token.tag_,
            #             token.dep_, token.shape_, token.is_alpha, token.is_stop, chunk_name])
            res.append(
                " ".join([token.text, token.lemma_, token.pos_, chunk_name]))
        return res

    def bio(x):
        if x[0] == 0:
            return "B-{0}".format(x[1])
        else:
            return "I-{0}".format(x[1])

    all_lines = []
    for match in matches:
        sentence = match.group(2)
        words = sentence.split(" ")

        chunk_name = match.group(1)
        chunk_name = re.sub(replace_reg, "", chunk_name)
        # chunks = [chunk_name]*len(words)
        # chunks = list(map(bio, enumerate(chunks)))
        # if "." in words[-1]:
        #     words[-1] = words[-1][:-1]
        #     words.append(".")
        #     chunks.append(".")
        # words = list(zip(words, chunks))
        # res.extend(map(lambda x: " ".join(x), words))
        all_lines.extend(pos_tag(sentence, chunk_name))
        all_lines.append("")
    return "\n".join(all_lines)


def create_chunks_file_from_input(input_file):
    with open(input_file, "r") as read_file, open("word_chunks.txt", "w+") as write_file:
        for line in read_file:
            write_file.write((parse_and_return_chunks(line)))
            write_file.write("\n")


if __name__ == "__main__":
    # create_chunks_file_from_input("train-ucla.txt")
    # create_crf_suite_text("word_chunks.txt")
    train("new_crf_text.txt")
