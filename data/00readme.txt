Author: Xiong Ding
Fri Dec 12 11:34:10 EST 2014
==============================
The format of Flqouet exponents:
[mu, w]: Lambda = exp(mu*T + 1i * w)

==============================
copyh5.sh : script to combine data for T < 100 and  100 < T < 120
mergeh5.sh : script to merge two h5 files by some selection rule
readksh5.m : Matlab script to export h5 contents to Matlab structure.

converge: 
	  contains the information about convergence of PED on 
	  upos.
120 : angle distriubtion for T < 120

Note: in the H5 files, nstp is double.

============================
Mon Jan  5 12:46:44 EST 2015

ks22h1t120.h5, ks22h1t120x64.h5 : Ruslan's original data
ks22h02t120x64.h5 : the refined the data with h = 0.02
ks22h005t120x64.h5 : the refined the data with h = 0.005
ks22h001t120x64.h5 : the refined the data with h = 0.001


ppo: 421 426 430 432 437 440 441 442 443 450 61 65 458 70 462 81 466 83
     470 472 479 481 103 105 108 109 112 490 117 120 495 496 497 129 502
     133 509 146 147 156 525 160 528 163 165 534 173 538 174 541 186 549
     188 557 198 202 563 564 207 567 208 209 568 571 216 574 219 220 578
     222 580 227 584 228 587 593 239 595 241 596 602 255 608 615 266 617
     619 269 271 625 277 280 637 640 304 653 308 658 660 661 664 665 320
     670 325 328 675 679 336 681 338 341 687 691 692 693 352 695 355 356
     357 700 359 701 703 705 706 707 712 373 374 375 377 387 726 731 732
     394 397 398 403 404 413 420 761 764 765 767 772 773 775 778 782 783
     784 789 790 791 792 802 805 809 812 815 818 819 820 822 823 824 828 
     831 835 836 839 (up to 839)
