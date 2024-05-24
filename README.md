# Footprint Analysis

## Synopsis
Bidimensional footprint analysis for Eddy Covariance data.

The project consists of: 

1. lower-level modules with the basic functions needed in order to produce 1D and 2D footprints.

## Instalation
The following Python libraries will be required:

  - python=3.9
  - numpy
  - scipy
  - gdal
  - proj
  - pyproj
  - https://github.com/hectornieto/pytseb

With `conda`, you can create a complete environment with
```
conda env create -f environment.yml
```

Then you can install the package using pip:

```
pip install ./
```

## License
footprint_analysis: a Python bidimensional footprint analysis for Eddy Covariance data.

Copyright 2024 Hector Nieto and contributors.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
