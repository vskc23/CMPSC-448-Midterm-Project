import numpy as np

#  get accuracy
def compare_with_test_set(correct_set,predicted_data):
    total = 0
    correct = 0
    for predicted_word, correct_word in zip(predicted_data, correct_set):
        total = total + 1
        if predicted_word[1] == correct_word[1]:
            correct = correct + 1

    accuracy = (correct / total) * 100
    return accuracy

# read_data() function from LR/new.py
def read_data(train_datapath,include_y=True):
    """Read sentences from given corpus data"""
    # Master list to hold all the sentences
    sentences = [] 
    # Read the corpus data
    with open(train_datapath) as f:
        content = f.readlines()
    # Create a local list to hold each sentence
    sentence = []
    for line in content:
        if line !='\n':
            # Remove the trailing newline character and split the line at whitespace
            line = line.strip() 
            # Get the word and pos tag
            word = line.split()[0].lower() 
            if include_y:
                pos = ""
                # Get the pos tag
                pos = line.split()[1] 
                # Append the word and pos tag as a tuple to the local sentence
                sentence.append((word, pos)) 
            else:
                sentence.append(word)
        else:
            # Append the local sentence to the master list
            continue
    print("-> %d sentences are read from '%s'." % (len(sentence), train_datapath))
    return sentence

def read_new_data(test_datapath):
    """Read sentences from given corpus data"""
    # Master list to hold all the sentences
    # Read the corpus data
    with open(test_datapath) as f:
        content = f.readlines()
    # Create a local list to hold each sentence
    sentence = []
    for line in content:
          # Remove the trailing newline character and split the line at whitespace
            line = line.strip() 
            line_len = len(line.split())
            if line_len < 2:
                continue
            # Get the word and pos tag
            word = line.split()[0].lower() 
            pos = ""
            # Get the pos tag
            pos = line.split()[1] 
            # Append the word and pos tag as a tuple to the local sentence
            sentence.append((word, pos)) 
    print("-> %d sentences are read from '%s'." % (len(sentence), test_datapath))
    return sentence

def main():
    # define the path to the training data
    VAL_DATA = "../dataset/test_labelled.txt"
    TEST_DATA = "../dataset/predicted_tags_new.txt"

    correct_set = read_data(VAL_DATA)
    predicted_data = read_new_data(TEST_DATA)

    accuracy = compare_with_test_set(correct_set,predicted_data)

    print("Accuracy: ",accuracy)

if __name__ == '__main__':
    main()