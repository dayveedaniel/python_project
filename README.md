**Labs 1 and 2 - Image processing**


Test the program for different sizes of processed image files. For testing take files with sizes: 10240 x 7680, 12800 x 9600, 20480 x 15360. Get the average value of the work of each image processing procedure at three times restarting the program.
Load a color image.
Get the values of intensity ,                                
where - intensity of pixel v, - value of red component of pixel , - value of green component of pixel , - value of blue component of pixel .
Set the value of scalar quantity - Threshold (any value from 100 to 200). Set the intensity values that are less than Threshold to 0, and those that are greater than Threshold to 1.
Perform erosion operation [https://habr.com/ru/post/113626/ or https://intuit.ru/studies/courses/10621/1105/lecture/17989?page=4] on the obtained matrix of 0 and 1. Note: You need to set the erosion step (any value from 1, 2 or 3). Get an image from the result by setting black pixels (0, 0, 0, 0) instead of values 0, and white pixels instead of values 1. Save the result to a file.
Evaluate the performance of the specified image processing algorithm for 2, 4, 6, 8, 10, 12, 14, and 16 threads per CPU.
