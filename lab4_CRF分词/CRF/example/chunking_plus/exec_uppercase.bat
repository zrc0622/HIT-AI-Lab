 ..\..\crf_learn -t -c 4.0 template1 train1.data model1
 ..\..\crf_test -m model1 test1.data -o result1
perl conlleval.pl -d "	" <result1 >result1.info

