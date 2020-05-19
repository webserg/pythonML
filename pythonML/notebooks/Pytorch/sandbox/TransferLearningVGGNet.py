# VGGNet is great because it's simple and has great performance, coming in second in the ImageNet competition. The idea here is that we keep all the convolutional layers, but replace the final fully-connected layer with our own classifier. This way we can use VGGNet as a fixed feature extractor for our images then easily train a simple classifier on top of that.
#
# Use all but the last fully-connected layer as a fixed feature extractor.
# Define a new, final classification layer and apply it to a task of our choice!
# You can read more about transfer learning from the CS231n Stanford course notes.
# https://cs231n.github.io/transfer-learning/