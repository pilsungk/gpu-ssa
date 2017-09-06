#!/usr/bin/perl -wnla
BEGIN {
  $count = 0;
  $sum_x1 = 0;
  $sum_x2 = 0;
  $sum_x3 = 0;

  $sumv_x1 = 0;
  $sumv_x2 = 0;
  $sumv_x3 = 0;

  # skip the first line
}

if (@F == 9)  {
    (undef, undef, undef, undef, undef, undef, $x1, $x2, $x3)=@F;
    if (($x1+$x2+$x3) == 1800) {
        $sum_x1 = $sum_x1 + $x1;
        $sum_x2 = $sum_x2 + $x2;
        $sum_x3 = $sum_x3 + $x3;

        $x1_array[$count] = $x1;
        $x2_array[$count] = $x2;
        $x3_array[$count] = $x3;

        $count += 1;
    }
}


END {
#  print ("sum_x1 = $sum_x1, sum_x2 = $sum_x2, sum_x3 = $sum_x3.\n");

  print ("Data processed = $count\n");

  $mean_x1 = $sum_x1/$count;
  $mean_x2 = $sum_x2/$count;
  $mean_x3 = $sum_x3/$count;

  print ("Mean:");
  printf ("\tx1 = %.2f, x2 = %.2f, x3 = %.2f\n\n", $mean_x1, $mean_x2, $mean_x3);

  for ($index = 0; $index < $count; $index++)
  {
      $sumv_x1 += ($mean_x1 - $x1_array[$index])**2;
      $sumv_x2 += ($mean_x2 - $x2_array[$index])**2;
      $sumv_x3 += ($mean_x3 - $x3_array[$index])**2;
  }

  $var_x1 = $sumv_x1/$count;
  $var_x2 = $sumv_x2/$count;
  $var_x3 = $sumv_x3/$count;

  $std_x1 = sqrt($var_x1);
  $std_x2 = sqrt($var_x2);
  $std_x3 = sqrt($var_x3);

  print ("Variance:");
  printf ("\tx1 = %.2f, x2 = %.2f, x3 = %.2f\n\n", $var_x1, $var_x2, $var_x3);
  print ("Standard Deviation:");
  printf ("\tx1 = %.4f, x2 = %.4f, x3 = %.4f\n\n", $std_x1, $std_x2, $std_x3);
}
