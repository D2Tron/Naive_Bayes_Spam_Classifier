All code exists in the code folder. All code was done in python. There is a main file called main.py.
There are two other files called training.py and classify.py.

To execute the program, must use command line arguments.

To execute the training program, use this command:
python code/main.py training -i spam.csv -os spamprob.csv -oh hamprob.csv

To execute the classify program, use this command:
python code/main.py classify -i <test csv file> -is spamprob.csv -ih hamprob.csv -o outfile.csv
*Here, the test csv file would be whatever your test csv file is, mine was called test.csv