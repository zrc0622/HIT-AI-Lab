use strict;
my $infile=@ARGV[0];
my $outfile=@ARGV[1];

if($infile eq undef){
  $infile="train.data";
}
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
$suffix_num++;
if($outfile eq undef){
	$outfile=$infile;
	$outfile=$infile.$suffix_num.'.'.$suffix;	
	}


open OUT,">$outfile"
or die "Could not open the input file:$infile!";
print "the output file is $outfile.\n";
###########################################################################
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
  my $token_type=&gettype(@current_array[0]);
  @current_array[-2].=" $token_type";
  $aline=join(' ',@current_array);

  print OUT "$aline\n";
}

close IN;
close OUT;
#########################################################################
#sub
sub gettype
{

  my ($aline)=@_;
  my $rect;
  my $taline=$aline;
  $taline=~s/[0-9a-zA-Z\/\-\_\@]//gs;
  my $tempstr=$aline;
  $tempstr=~s/[0-9]//gs;

  if($aline=~/^\[h\]/gs){
    $rect="[h]";
    }
  elsif($aline=~/^\[hh\]/gs){
    $rect="[hh]";
    }


    elsif($aline=~/^[\.|\,|\?|\:|\;|\|!]$/||$aline=~/^\.{3}/){
    $rect="punc";
    }
    elsif($aline=~/^[\'|\"|\[|\]|\/|\\]$/){
    $rect="pun";
    }
    elsif($tempstr eq undef||$tempstr eq '+'||$tempstr eq '-'||$tempstr eq '.'
    ||($tempstr eq '$')||($tempstr eq '#')||($tempstr eq '%')||($tempstr eq '/')){
    $rect="num";
    }

    elsif(($taline eq undef) || ($taline eq "\'" && $aline=~/(s|t|m|ve|d|ll|re)$/i)){

    if($aline eq lc($aline)){
     $rect="lc";
     }
     elsif($aline eq uc($aline)){
     $rect="uc";

     }
     elsif($aline eq ucfirst($aline)){
     $rect="ucf";
     }
     else{
     $rect="mix";
     }
    }
    else{
    if(length($aline)==1){
    $rect="sign";}
    else{
    $rect="unknown";}
    }
    return $rect;
}
#########################################################################


