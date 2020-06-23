Plan and Write is a project I've been working on slowly mainly for my dissertation
however due to pandemic events, project work was put on hold
now that university has finished, I am able to continue working on it

This project aims to produce text based stories by first generating a plot based on a users title input
from this plot a 5 sentence, 25 word total, story will be generated

the work in this project aims to replicate that of Plan and Write (Yao et al 2019)
however with the exception of moving from pytorch library, to TensorFlow

currently I'm in the process of writing the planning frame work after using the RAKE algorithm
to extract key word plots from the ROCStories dataset (a collection of near 100,000 verified and checked 5 sentence stories)

this project mainly is serviced as a learning project, allowing myself to expand my knowledge on Artificial intelligence, Machine learning and TensorFlow in general

this project has no current executable, but requires TensorFlow 2.2 the latest stable release at the time of writing

I suggest running plotTrainerV2 as it shows more promising results
it can be run in vscode, and can be queried by in code setting the Query boolean to true and changing the input_raw value on line 185 to any text value excluding numeric values

to train the network which must be done before use, set Query to False and on line 161 set the number of epochs the network should run for
