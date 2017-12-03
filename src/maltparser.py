#!/usr/bin/env python
# -*- coding: utf-8 -*-
from jpype import *
import corpus2

class MaltParser(object):
    def __init__(self, model_file):
        ConcurrentMaltParserService = JClass("org.maltparser.concurrent.ConcurrentMaltParserService")
        url = JClass("java.io.File")(model_file).toURI().toURL()

        self.model = ConcurrentMaltParserService.initializeParserModel(url)
        self.ConllStreamWriter = JClass("g419.corpus.io.writer.ConllStreamWriter")

    def parse_sentence(self, sentence):
        conll_sent = self.ConllStreamWriter.convertSentence(sentence)
        return self.model.parse(conll_sent)



