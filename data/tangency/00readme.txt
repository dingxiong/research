Mon Sep 21 14:01:11 EDT 2015
============================================================
100 : orbits with T < 100 for N = 32
120 : orbits with T < 120 for N = 32
anglePOs64: the first 200 rpos and ppos

============================================================
Log:

== Sat Jan 30 13:55:51 EST 2016 ==

The angle distribution of rpo 33, 36, 59, 60, 79, 81, 109, 114
are wired, so I suspect their FV do not converge. I use a smaller
tolerrance rtol = 1e-12 to recalculate FV. All converge and now
it looks norm.

