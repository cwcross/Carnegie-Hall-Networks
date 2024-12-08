# Carnegie-Hall-Networks
Using data from the New York Philharmonic, these networks encompass composers (nodes) and their shared performances (edges) by the New York Philharmonic at Carnegie Hall since its formation.


The first network covers all performances since Carnegie Hall's formation. It contains connections between composers when they have been performed at the same event a minumum of 8 times. 

The other networks contain the exact same, but for a specific time period. Each time period has been selected so that there are 5 time periods with a roughly equivalent number of performances. There is a connection between composers if they have been featured at the same event as another composer a minumum of 5 times.

The 5 time periods and number of performances are the following:
```
year                  # of performances      file number
(1922.0, 1934.0]      4512                   2
(1943.0, 1953.0]      4424                   4
(1891.9, 1922.0]      4418                   1
(1953.0, 2022.0]      4325                   5
(1934.0, 1943.0]      4239                   3
```
