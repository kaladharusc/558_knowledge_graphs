# !  # /usr/bin/python3

import sys
import optparse
import pycrfsuite
import crfutils
from sklearn.metrics import classification_report
import numpy as np
import re
import json

common_template = "w lemma pos tag dep stop alpha shape"


def get_start_end(c, is_start=True):
    if is_start:
        return "<{0}>".format(c)
    else:
        return "</{0}>".format(c)


def get_formatted_string(one_line):
    w, c = one_line[0]
    res = [get_start_end(c), w]
    last_tag = c
    for w, c in one_line[1:]:
        if last_tag == c:
            res.append(w)
        else:
            res.append(get_start_end(last_tag, False))
            last_tag = c
            res.append(get_start_end(c))
            res.append(w)
    res.append(get_start_end(last_tag, False))
    return " ".join(res)


def create_out_with_tags(chunk_file, predictions, outfile, write_to_file=True):
    predictions = list(map(lambda x: x[0], predictions))
    res = []
    with open(chunk_file, "r") as read_file, open(outfile, "w+") as write_file:
        temp = []
        i = 0
        for line in read_file:
            line = line.strip("\n").strip(" ")
            word = line.split(" ")[0]
            if len(line) == 0:
                one_line = list(zip(temp, predictions[i:i+len(temp)]))
                res.append(one_line)
                i = i+len(temp)
                if write_to_file:
                    write_file.write(get_formatted_string(one_line))
                    write_file.write("\n")
                temp = []
            else:
                temp.append(word)
    return res


def get_chukns_and_string_values(arr):
    res = []
    for line in arr:
        line = line.strip(" ").strip("\n")

        regex = r'(<\w+?>)(.*?)(<\/\w+?>)'
        replace_reg = r'[<>]'
        matches = re.finditer(regex, line)
        one_line_res = {}
        for match in matches:
            sentence = match.group(2).strip(" ")
            # words = sentence.split(" ")

            chunk_name = match.group(1)
            chunk_name = re.sub(replace_reg, "", chunk_name)
            if chunk_name not in one_line_res:
                one_line_res[chunk_name] = []
            # all_lines.extend(pos_tag(sentence, chunk_name))
            one_line_res[chunk_name].append(sentence)
            # one_line_res.append((chunk_name, sentence))
        res.append(one_line_res)
    return res


def post_process_final_out(final_file):
    res = []
    with open(final_file, "r") as f:
        for line in f:
            res.append(line)
    for i in range(len(res)):
        line = res[i]
        line = line.replace(" .", ".")
        line = line.replace(" ,", ",")
        line = line.replace(" ;", ";")
        line = line.replace(" :", ":")
        line = line.replace(" )", ")")
        line = line.replace("( ", "(")
        res[i] = line
    with open(final_file, "w") as f:
        f.write("".join(res))
# results : TP FP FN P R


def calc_f1(test_file, chunk_file):
    actual_lines = []
    predicted_lines = []
    results = {}
    with open(test_file, "r") as chunk_read_file, open("final_ans_"+chunk_file, "r") as read_final_file:
        for line in chunk_read_file:
            line = line.strip(" ").strip("\n")
            if len(line) > 0:
                actual_lines.append(line)
        for line in read_final_file:
            line = line.strip(" ").strip("\n")
            if len(line) > 0:
                predicted_lines.append(line)
    actual_lines_formatted = get_chukns_and_string_values(actual_lines)
    prdicted_lines_formatted = get_chukns_and_string_values(predicted_lines)
    for i in range(len(actual_lines_formatted)):
        act_line = actual_lines_formatted[i]
        pred_line = prdicted_lines_formatted[i]
        seen = []
        for k, v in act_line.items():
            seen.append(k)
            if k not in results:
                results[k] = [0, 0, 0]
            a_v = set(v)
            if k in pred_line:
                p_v = set(pred_line[k])
            else:
                p_v = set()
            results[k][0] = results[k][0] + len(a_v.intersection(p_v))
            results[k][1] = results[k][1] + len(p_v-a_v)
            results[k][2] = results[k][2] + len(a_v-p_v)
        for se in seen:
            # del act_line[k]
            if se in pred_line:
                del pred_line[se]
        for k, v in pred_line.items():
            if k not in results:
                results[k] = [0, 0, 0]
            results[k][1] = results[k][1] + len(v)

    for k, v in results.items():
        p = v[0]/(v[0]+v[1])
        r = v[0]/(v[0]+v[2])
        if p+r != 0:
            f1 = (2*p*r)/(p+r)
        else:
            f1 = 0
        v.append(p)
        v.append(r)
        v.append(f1)
    import json
    print(json.dumps(results))


def predict(model_name, test_file, out_file):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_name)
    chunks_file_test = "chunks_test_"+test_file
    crf_suite_test = "crf_suite_test_"+test_file
    create_chunks_file_from_input(test_file, chunks_file_test)
    create_crf_suite_text(chunks_file_test, crf_suite_test,
                          common_template)

    x_test = get_feature_from_file(crf_suite_test)
    y_pred = [tagger.tag(xseq) for xseq in x_test]
    create_out_with_tags(chunks_file_test, y_pred, out_file)
    # final_file = "final_ans_"+chunks_file_test
    # final_file = out_file
    # post_process_final_out(out_file)
    # print(y_pred)


words_chunks_predicted = []


def predict_with_f1(model_name, test_file):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_name)
    chunks_file_test = "chunks_test_"+test_file
    crf_suite_test = "crf_suite_test_"+test_file
    create_chunks_file_from_input(test_file, chunks_file_test, True)
    create_crf_suite_text(
        chunks_file_test, crf_suite_test, common_template + " y", True)

    x_test, y_test = get_feature_from_file(crf_suite_test, True)
    y_pred = [tagger.tag(xseq) for xseq in x_test]
    labels = list(set(map(lambda x: x[0], y_test)))
    labels_map = {k: v for v, k in enumerate(labels)}

    create_out_with_tags(chunks_file_test, y_pred,
                         "final_ans_"+chunks_file_test, True)
    final_file = "final_ans_"+chunks_file_test
    post_process_final_out(final_file)
    calc_f1(test_file, chunks_file_test)


def train(feature_file):
    trainer = pycrfsuite.Trainer(verbose=True)
    X_train, Y_train = get_feature_from_file(feature_file, True)
    for xseq, yseq in zip(X_train, Y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 0.1,
        'c2': 0.01,
        'max_iterations': 1000,
        'feature.possible_transitions': True
    })
    trainer.train('crf.model')


def get_feature_from_file(file_name, y_included=False):
    X, Y = [], []
    with open(file_name,  "r") as f:
        for line in f:
            line = line.strip(" ").strip("\n")
            if len(line) == 0:
                continue
            arr = line.strip("\n").strip("\t").split("\t")
            if y_included:
                X.append([arr[1:]])
                Y.append([arr[0]])
            else:
                X.append([arr])
    tup = (X, Y) if y_included else X
    return tup


def create_crf_suite_text(file_name, crf_suite_file, fields, include_y=False):
    separator = ' '
    needed_features = ['w', 'lemma', 'tag',
                       'dep', 'stop', 'alpha', 'shape', 'pos']
    # prev_and_present = [-2,-1,0,1,2]
    # needed_features = ['w', 'pos', 'lemma', 'stop','alpha', 'shape', 'tag', 'dep']
    templates = []
    for feat in needed_features:
        # for i in prev_and_present:
        templates.extend([
            ((feat, -3), ),
            ((feat, -2), ),
            ((feat, -1), ),
            ((feat,  0), ),
            ((feat,  1), ),
            ((feat,  2), ),
            ((feat,  3), ),
            ((feat, -1), (feat,  0)),
            ((feat,  0), (feat,  1)),
            ((feat, -2), (feat, -1)),
            ((feat, 1), (feat, 2)),
            ((feat, -2), (feat, -1), (feat,  0)),
            ((feat, -1), (feat,  0), (feat,  1)),
            ((feat,  0), (feat,  1), (feat,  2)),
            ((feat,  -3), (feat,  -2), (feat,  -1), (feat,  0)),
            ((feat,  -2), (feat,  -1), (feat,  -0), (feat,  1)),
            ((feat,  -1), (feat,  0), (feat,  1), (feat,  2)),
            ((feat,  0), (feat,  1), (feat,  2), (feat,  3))

        ])
    templates = tuple(templates)

    def feature_extractor(X):
        crfutils.apply_templates(X, templates)
        if X:
            X[0]['F'].append('__BOS__')     # BOS feature
            X[-1]['F'].append('__EOS__')    # EOS feature

    fo = open(crf_suite_file, "w+")
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
    Y = 'y' if include_y else None
    for X in crfutils.readiter(open(file_name, "r"), F, options.separator):
        feature_extractor(X)
        crfutils.output_features(fo, X, Y)


import spacy
nlp = spacy.load('en_core_web_sm')


def pos_tag(sentence, chunk_name):
    doc = nlp(sentence)
    res = []

    i = 0
    for token in doc:
        # res.append([token.shape_])
        features = [token.text, token.lemma_, token.pos_,
                    token.tag_, token.dep_, str(token.is_stop), str(token.is_alpha), token.shape_]
        if chunk_name:
            # new_chunk_name = "B-"+chunk_name if i ==0 else "I-"+chunk_name
            # features.append(new_chunk_name)
            features.append(chunk_name)
        i += 1
        res.append(" ".join(features))
    return res


def bio(x):
    if x[0] == 0:
        return "B-{0}".format(x[1])
    else:
        return "I-{0}".format(x[1])


def parse_and_return_features(line):
    line = line.strip(" ").strip("\n")
    res = pos_tag(line, None)
    res.append("")
    return "\n".join(res)


actual_tokens_and_chunks = []


def parse_and_return_chunks(line):
    line = line.strip(" ").strip("\n")

    regex = r'(<\w+?>)(.*?)(<\/\w+?>)'
    replace_reg = r'[<>]'
    matches = re.finditer(regex, line)

    all_lines = []
    for match in matches:
        sentence = match.group(2).strip(" ")
        words = sentence.split(" ")

        chunk_name = match.group(1)
        chunk_name = re.sub(replace_reg, "", chunk_name)
        all_lines.extend(pos_tag(sentence, chunk_name))

        actual_tokens_and_chunks.append(
            list(zip(words, [chunk_name]*(len(words)))))
    all_lines.append("")
    return "\n".join(all_lines)


def create_chunks_file_from_input(input_file, word_chunks_file, has_chunk=False):
    with open(input_file, "r", encoding='utf-8') as read_file, open(word_chunks_file, "w+", encoding='utf-8') as write_file:
        for line in read_file:
            if has_chunk:
                write_file.write((parse_and_return_chunks(line)))
            else:
                write_file.write((parse_and_return_features(line)))
            write_file.write("\n")


if __name__ == "__main__":
    # test_file = "final-test-ucla.txt"
    # train_File = "train-ucla.txt"

    # create_chunks_file_from_input("train-ucla.txt", "word_chunks.txt", True)
    # create_crf_suite_text(
    #     "word_chunks.txt", "new_crf_text.txt", common_template+" y", True)
    # train("new_crf_text.txt")
    # # # predict_with_f1(sys.argv[1], sys.argv[2])
    # print("test_data")
    # predict_with_f1("crf.model", test_file)
    # print("train_data")
    # predict_with_f1("crf.model", train_File)
    predict(sys.argv[1], sys.argv[2], sys.argv[3])
    # predict("crf.model", "final-test-ucla.txt", "final_submission.txt")

    # calc_f1(test_file, chunks_file_test)
