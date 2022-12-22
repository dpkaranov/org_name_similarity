# import libraries
import argparse
import json
import os

import spacy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class NamSim:
    '''
    Class for checking similarity of organisations' names
    '''
    def __init__(self):
        self.model_spacy = spacy.load("en_core_web_sm")
        try:
            self.model_st = SentenceTransformer('./models/1.0_db-multilingual-cased-v2')
        except Exception:
            print('Check the model folder. Perhaps it is empty.')

    def check_similarity(self, name_1: str, name_2: str):
        '''
        Method of NamSim for checking two str for similarity
        '''
        first = self.model_st.encode([name_1], convert_to_numpy=True)
        second = self.model_st.encode([name_2], convert_to_numpy=True)
        cosine_scores = cos_sim(first, second)
        return float(cosine_scores[0][0]) > 0.88

    def parse_deep(self, text: str):
        '''
        Method of NamSim for parsing text with
        Spacy and checking similarity of all organisations' names
        with SentenceTransformer
        '''
        add_val = False
        token_dic = {}
        last_tokens = []
        sentences = text.split('.')
        for sen in sentences:
            doc = self.model_spacy(sen)
            for token in doc.ents:
                if token.label_ == 'ORG' and len(token) > 1:
                    print(token)
                    if len(last_tokens) > 0:
                        for t in last_tokens:
                            if self.check_similarity(token, t):
                                token_dic[t].append((token.start_char, token.end_char))
                                add_val = True
                                break
                    if not add_val:
                        last_tokens.append(token.text)
                        token_dic[token.text] = []
                        token_dic[token.text].append([token.start_char, token.end_char])
                add_val = False
        return token_dic

    def parse_text(self, path: str):
        '''
        Method of NamSim to start parsing text
        '''
        file_name = path.split('/')[-1].split('.')[0]
        if os.path.exists(path) and path.endswith('.txt'):
            with open(path, 'r') as file:
                text = file.read()
            file.close()
            text = text.strip()
            data = self.parse_deep(text)
            with open(os.path.join('./out', file_name + '.json'), 'w') as new_file:
                json.dump(data, new_file, indent=4)
            new_file.close()
        else:
            print("Path to file doesn't exsist")


if __name__ == '__main__':
    namsim = NamSim()
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', type=str, nargs=2, required=False)
    parser.add_argument('--path', type=str, nargs=1, required=False)
    args = parser.parse_args()
    if args.check:
        res = namsim.check_similarity(args.check[0], args.check[1])
        print(res)
    elif args.path:
        namsim.parse_text(args.path[0])
    else:
        namsim.parse_text('./texts/1.txt')
