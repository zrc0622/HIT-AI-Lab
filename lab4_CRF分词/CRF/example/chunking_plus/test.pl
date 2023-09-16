use strict;
my $str="this a number, that is a word";
my @num=($str=~/ (\w) /g);
print "@num\n";
