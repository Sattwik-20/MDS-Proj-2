# MDS-Proj-2
The Codes were written and run on Google Colab. Follow the following steps to run the code.
1. For each code put the respective file in a directory and use the directory name in respective arguments of function-:

  In all the .py put the data_set.csv and use the directory name in                  pd.read_csv() function.

2. Copy and paste the code in the code block of Google Colab and run it. You will find a set of clean_data.csv file created and also in output you will have the accuracy and loss plots.
3. run Only_Loss_Function.py - for Model with only the loss function and its respective ouput plots
4. Run With_Regularizer to see the different result
5. To find the answer to the test data point run, Test_Data_Point to find the answer.

Remember to change the directory name accordingly.

Walk Through of the Code-:
1.First the data is cleaned and reorganised. We replace the ? values with average values of the respective feature
2. Normalizing the given data
3. Choosing the target variable symbolling and converting it into an encoded vector.
4. Then we split the data into training and testing sets.
5. We train the model with the training data and make predictions
6. using the predicted and truth values from the test data we evaluate the accruaracy and loss values
7. in case of without regularizer in model definition we dont out the kernel.regularizer argument.
8. In case of with we simply provide it as an argument
9. Finally we plot the Accuracy and Losses for observational purposes.
