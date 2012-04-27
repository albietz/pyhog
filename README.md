# Python HOG features extractor

This is a python module (C++ with boost.python) for computing HOG features (*[Dalal and Triggs, CVPR 2005](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)*) from images, based on code from [Piotr Dollár's toolbox](http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html).

To compile the module, try something similar to:

---
g++ pyhog.cpp -shared -o pyhog.so -I/usr/include/python2.7 -lpython2.7 -lboost_python -fPIC
---

## Example

Here's an example of use with numpy. There is no native numpy support for now, so you need to convert the image to a string and then reshape the resulting array.

```python
import pyhog
import Image
import numpy as np

img = np.array(Image.open('myImage.png'))
width = img.shape[1]
height = img.shape[0]
channels = img.shape[2]
sbin = 8  # size of spatial bins in pixels
obin = 9  # number of initial orientation bins (gives 4*obin normalized bins)

hg, w, h, b = pyhog.hog(img.tostring(), width, height, channels, sbin, obin)
hog = np.array(hg).reshape(b, h, w)  # this means hog is accessed with hog[bin,y,x]
```
