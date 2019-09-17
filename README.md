# Region Detection Rate
Repository with code and examples for the evaluation of the region detection rate.

## Installation
```
git clone https://github.com/pLeminoq/region-detection-rate.git
mkdir build
cd build
cmake ..
make
```
The installation required [Boost](https://www.boost.org/) and [OpenCV](https://opencv.org/) (I think at least version 3.0.0).


## Usage
The created binary needs a file which maps a ground truth image to a prediction in each line.
For an example have a look at one of the [mapping.txt](examples/from_datasets/mapping.txt) files in the examples.
Then run:
```
./eval <mapping-file>
```
For example:

```
./eval ../examples/from_datasets/mapping.txt
```
The flag _-v_ or _--verbose_ can be used for a more verbose mode.
In this mode, the results for the measure will be printed for every image in the mapping as well as statistics for all regions and their intersections.
Additionally, images with detected regions and false predictions marked will be displayed.
If the 's' key is pressed, these images will be saved.
On any other key press, these images will not be saved in the images for the next mapping will be displayed.
