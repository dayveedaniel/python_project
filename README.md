**Labs 1 and 2 - Image processing**


Test the program for different sizes of processed image files. For testing take files with sizes: 10240 x 7680, 12800 x 9600, 20480 x 15360. Get the average value of the work of each image processing procedure at three times restarting the program.
Load a color image.
Get the values of intensity ,                                
where - intensity of pixel v, - value of red component of pixel , - value of green component of pixel , - value of blue component of pixel .
Set the value of scalar quantity - Threshold (any value from 100 to 200). Set the intensity values that are less than Threshold to 0, and those that are greater than Threshold to 1.
Perform erosion operation [https://habr.com/ru/post/113626/ or https://intuit.ru/studies/courses/10621/1105/lecture/17989?page=4] on the obtained matrix of 0 and 1. Note: You need to set the erosion step (any value from 1, 2 or 3). Get an image from the result by setting black pixels (0, 0, 0, 0) instead of values 0, and white pixels instead of values 1. Save the result to a file.
Evaluate the performance of the specified image processing algorithm for 2, 4, 6, 8, 10, 12, 14, and 16 threads per CPU.


**Labs 3 - Image processing**

Test the program for different sizes of processed image files. For testing take files with sizes: 10240 x 7680, 12800 x 9600, 20480 x 15360. Get the average value of the work of each image processing procedure at three times restarting the program.
Load a color image.
On the basis of the image build the next level of the Gaussian pyramid (i.e. reduce the size of the image by half).



**Labs 5 - Working with the Apache Spark distributed computing framework**

Program A (working with Spark SQL component) is a set of queries to the database 'brooklyn_sales_map.csv'.
Program B (working with Spark MLlib component) builds three machine learning models: logistic regression, decision tree and random forest for the data set specified in the variant. For each algorithm, the best model should be obtained by finding the best set of its parameters. For the logistic regression algorithm, these parameters are maxIter=10...10000, regParam>0 (0.1, 0.5, 1, ...), elasticNetParam=0...1. For the decision tree maxDepth > 0 (3, 5, 9, 12, ...). For the random forest maxDepth > 0 (3, 5, 9, 12, ...) and numTree > 0 (5, 11, 25, ...). Obtain the values of Confusion Matrix (Confusion Matrix), Fidelity (Accuracy), Completeness (Recall) and Precision (Precision) for all models.



**Labs 6 - Fundamentals of deep learning**

Write 3 programs that perform training of neural networks 
Perform loading and preprocessing of data from the data sets. 
Divide each sample into training, test and validation samples. 
Train a set of neural network architectures, differing by different set of parameters: number of layers, number of neurons (feature maps) in the layers:
Select neural network architectures that on the one hand allow to obtain models with the best performance quality metrics, on the one hand, and on the other hand, are not redundant and on the other hand are not redundant and not overtrained.
Obtain the curves of Loss, Accuracy (for the for classifiers), mean absolute error (MAE - for regressors).
Draw conclusions on the results of model building.
