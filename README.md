# Motion Detection App

This is a simple real-time motion detection program using Python and OpenCV. The application allows you to adjust various parameters for customizing motion detection, as well as facial and eye detection.

## Requirements
This code was created based on Python 3.11.
Make sure you have Python and the required libraries installed. You can install them using the following command:

```bash
pip install opencv-python numpy pillow
```
## Usage

Run the script, and the application will open in a graphical user interface window. From there, you can adjust the following parameters:

- **Transparency of the area within contours:** Controls the opacity of the area within detected contours.
- **Paint area within contours:** Enables/disables painting the area within contours.
- **Motion sensor sensitivity:** Adjusts the sensitivity of the motion sensor.
- **Morphology iterations (erode/dilate):** Controls the number of iterations for morphological operations.
- **Contour thickness:** Defines the thickness of the drawn contours.
- **Information text size:** Adjusts the size of the text displaying detection information.
- **Maximum contour lines:** Limits the maximum number of lines drawn in the contour.

Additionally, you can activate face and eye detection, as well as highlight the detection.
