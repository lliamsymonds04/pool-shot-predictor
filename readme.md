# Pool Shot Predictor
* Uses OpenCV and python to detect a pool table and balls
* Uses the 'ghost ball' method to line up a shot
* Shows the shot with the least angle difference, in theory the easiest shot to make
* Handles simple collision scenarios

## Demo
[![YouTube](http://i.ytimg.com/vi5Lc4tfRdDl0/hqdefault.jpg)](https://www.youtube.com/watch?v=5Lc4tfRdDl0)

## Downloading
* Python 3.12 or higher
* Download the code base
* run ```pip install -r requirements.txt```

## Usage
* Run **main.py**
* Press *q* or stop to close
* Sample images are provided in the images folder

## Limitations
* The vision can struggle to determine if a shadow is a black ball, most of the time this is excluded.
* As both the table and balls are green it can be tricky to differentiate them
* Sometimes a full ball will be detected as a striped ball due to placement of the white center:w
