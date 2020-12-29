# ModernSlideShower
Image viewer with smooth image mothion and easy navigation

[Intro video](https://www.youtube.com/watch?v=kGz4jEd-cZQ)

## Description
ModernSlideShower is an image viewer aimed at ease of use and good visual quality.
Key features:
* **Unique fullscreen interface.** In fullscreen mouse cursor is never shown. All navigation is done without seeing mouse cursor.
* **Support for big image sets.** Perfectly workable with lists of millions af images.
* **Easy mouse navigation.** You can smoothly zoom, pan, switch images with only left and right mouse buttons. Mouse wheel can also be used for switching images.
* **Rich image switching arsenal.** 
  * Previous/next image.
  * First image in current folder
  * First image in previous/next folder
  * Random image
    * Random image in the whole list
    * Random image in current folder
    * First image in random folder
    * Random image in random folder
* **Automatic image swithing (slideshow),** where you can change switching speed on the go and visual indicator till next switch.
* **Image rotation on the screen.** Option to rotate by 90° angles, as well as by arbitrary angle.
* **Lossless jpeg rotation.** Using free third-party utility [jpegtran](https://sourceforge.net/projects/libjpeg-turbo/files/2.0.5/)
* **Basic image editing**
  * Levels adjustment for all colors at once as well as for R, G, B separately.
  * Image resize, crop, and rotation
* **Fractal mode**, showind mandelbrot set with ability to zoom up to 2^100 with usable framerate on average hardware.


## Installation
1. Install Python (~30 MB installer size)
https://www.python.org/downloads/

1. Download all project files and unpack them into a folder of your choice.

1. Install required Python modules (~100 MB).

``` pip install -r requirements.txt``` 

  Or just double-click **install requirements.cmd** if you are using Windows.
  
  To be able to save losslessly rotated jpeg files, put [jpegtran.exe](https://sourceforge.net/projects/libjpeg-turbo/files/2.0.5/) into folder with this program (Windows only)

## Usage
Run program by launching ModernSlideShower.py

To open a folder (and all of its subfolders) with images, pass it as an argument:

``` ModernSlideShower.py d:\Pics\Elephants```

Alternately, under windows, drag-and-drop a folder on a shortcut of ModernSlideShower

To run in windowed mode (rather than fullscreen, which is default, pass -f command line ardument:

``` ModernSlideShower.py d:\Pics\Elephants -f``` 

