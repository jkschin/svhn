# svhn
Based on the CIFAR10 example on TensorFlow. Reconfiguring it to run my own images and data.

I removed the "LICENSE HEADERS" in the code because it was getting kinda cluttered. My apologies if I violate any license laws. I'm new to this open source thing, so drop me an email at samuelchin91@gmail.com and tell me that I'm doing it wrong and this is not allowed. This code is for personal use and nothing else.

Notable Improvements

1. Refactored all flags to be put in python flags file.
2. Made a small hack to evaluate the test set sequentially (we lose out on speed greatly). This way, we can single out those images that have been classified wrongly. Still imperfect and a work in progress.
3. Previous CIFAR10 evaluation code runs the test set in batches. In my opinion, this is unnecessary. We only want to run every image in the test set once.
4. Refactored code to allow for 3 channel and 1 channel images.

Successful Runs:

1. Street View House Number Data Set from http://ufldl.stanford.edu/housenumbers/ 
  * After training, I cropped my own images in. 100% accuracy. Wow!
2. My self-engineered Wally data set.
  * 100% accuracy too, but the data set creation is off.
3. Assortment of others that yield promising results.

P.S. I apologize for messy code. This is experimental. Trying to hack stuff together to make it work. Expect a refactored general purpose module soon! 
