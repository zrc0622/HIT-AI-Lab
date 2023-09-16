use strict;
my $infile=@ARGV[0];
my $outfile=@ARGV[1];

open IN,"$infile"
or die "Could not open the input file:$infile!";
print "The input file is $infile, ";
my $suffix;
my $suffix_num;
if($infile=~s/\.(\w+)$//){
  $suffix=$1;
}
if($infile=~s/(\d+)$//){
  $suffix_num=$1;
}
$suffix_num++;
if($outfile eq undef){
	$outfile=$infile;
	$outfile=$infile.$suffix_num.'.'.$suffix;	
	}

open OUT,">$outfile"
or die "Could not open the input file:$infile!";
print "the output file is $outfile.\n";
##########################################################
my $preline;
while(my $aline=<IN>){
  $aline=~s/^\s+//gs;
  $aline=~s/\s+$//gs;
  if($aline eq undef){
    $preline=undef;
    print OUT "\n";
    next;
  }
  $aline=~s/ +/ /gs;
  my @current_array=split(/ /,$aline);
  if(@current_array[0] =~/^[A-Z]/){
    @current_array[-2].=' 1' ;
    $aline=join(' ',@current_array);
  }
  else{
    @current_array[-2].=' 0'  ;
    $aline=join(' ',@current_array);
  }
  print OUT "$aline\n";
}

close IN;
close OUT;


