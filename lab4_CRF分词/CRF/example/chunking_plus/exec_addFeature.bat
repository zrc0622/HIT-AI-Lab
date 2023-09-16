..\..\crf_learn -t -c 4.0 template5 train.data model5
..\..\crf_test -m model5 test.data -o result5
perl conlleval.pl -d "	" <result5 >result5.info

