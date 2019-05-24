#!/usr/bin/env python3
"""
    This program splits a .tsv file into however many different dialects it
    contains.  Each line in the .tsv file corresponds to one line in a file 
    the name of which ends with -DIA.txt, where DIA is the tab-separated
    dialect from the .tsv line; the DIA is not copied into the output.
"""
import sys

Filename = (
"MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-6-train.tsv"
    )

"""
"MADAR-SHARED-TASK-initial-release-28Jan2019/MADAR_TRAINING_SAMPLE.tsv"
"MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-26-dev.tsv"
"MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-26-train.tsv"
"MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-6-dev.tsv"
"MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-6-train.tsv"
"""

OutputPrefix = "outs/MADAR_TRAINING_SAMPLE"
BusyBody = False# Because, without my help, kenlm sticks <s> and </s> in.
CHARMODE = False

def command_line():
    global CHARMODE, Filename, BusyBody, OutputPrefix
    mode = 0
    for a in sys.argv:
        if mode == 0:
            mode = -1
        elif mode == 1:
            OutputPrefix = a
            mode = -1
        elif mode == 2:
            Filename = a
            mode = -1
        elif a == '-c':
            CHARMODE = True
        elif a == '-w':
            CHARMODE = False
        elif a == '-oprefix':
            mode = 1
        else:
            sys.stderr.write("?don't recognize: ")
            sys.stderr.write(a)
            sys.stderr.write('\n')
            sys.exit(1)

def main():
    global OutputPrefix, BusyBody, Filename
    command_line()
    outputs = dict()
    with open(Filename) as fin:
        for lin in fin:
            line,dialect = lin.strip().split('\t');
            if CHARMODE:
                oldline = line
                line = ""
                for word in oldline.split():
                    line += ('<w> ')
                    for ch in word:
                        line += (ch)
                        line += (' ')
                    line += ('</w> ')

            if dialect in outputs:
                fout = outputs[dialect]
            else:
                fout = open(OutputPrefix + '-' + dialect + '.txt', 'w')
                outputs[dialect] = fout

            if BusyBody :
                fout.write('<s> ')
            fout.write(line)
            if BusyBody :
                fout.write(' </s>')
            fout.write('\n')
    for fout in outputs.values():
        fout.close()


main()
