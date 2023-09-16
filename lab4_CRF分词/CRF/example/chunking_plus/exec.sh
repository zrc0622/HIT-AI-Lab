#!/bin/sh
../../crf_learn -t -c 4.0 template train.data model
../../crf_test -m model test.data
rm -f model
