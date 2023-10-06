import numpy as np

#  get accuracy
def compare_with_test_set(correct_set,predicted_data):
    total = 0
    correct = 0
    for predicted_sentence, correct_sentence in zip(predicted_data, correct_set):
        for predicted_word, correct_word in zip(predicted_sentence, correct_sentence):
            total = total + 1
            if predicted_word[1] == correct_word[1]:
                correct = correct + 1

    accuracy = (correct / total) * 100
    return accuracy