 ..\..\..\crf_learn -t -c 4.0 template_mine train_data_mine.txt model_mine
 ..\..\..\crf_test -m model_mine test_data_mine.txt -o result
perl conlleval.pl -d "	" <result >result.info
