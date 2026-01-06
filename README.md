To the finished and published version: https://github.com/vi-rwth/FESTA.git \
To the publication: https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c01022

Main differences:
- no PLUMED dependancy anymore:
  - if no FES-file provided: Histogram creation from COLVAR-file
  - read columns manually setable for COLVAR- and FES-files
- pre-sorting of trajectory and therefore significant speed increase
- compatible with MLIPs CFG-format (https://doi.org/10.1063/5.0155887) read+write
- multiple trajectory- and COLVAR-files can be concatenated
- (minor) FES-png now shows the true polygon outlines
- (minor) possible to only generate FES-png without frame separation ("preview mode")

This is purely experimental and highly work-in-progress

Possible future additions:
- output of only unique structures
