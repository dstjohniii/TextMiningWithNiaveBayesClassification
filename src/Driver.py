import numpy as np
import math

def main():
    #Test Sentences
    sentences = []
    sentences.append("the puppy ate my computer science homework.")
    sentences.append("i went to school for a computer science convention.")
    sentences.append("i took a long drive last night to clear my mind")
    sentences.append("i spent my time watching tv while i was working")
    sentences.append("computers are my favorite puppy they are they are")
    
    problem(sentences, 'BinaryData.csv', convert_to_keywords_binary, bayes_classification)
    problem(sentences, 'NonBinaryData.csv', convert_to_keywords_count, bayes_continuous_classification)

def problem(sentences, file_name, keyword_function, classification_function):
    dataset = np.genfromtxt(file_name, delimiter=',', skip_header=True, usecols=(0, 1, 2, 3, 4, 5, 6))
    
    for sentence in sentences:
        keywords_present = keyword_function(sentence)
        result_binary = classification_function(dataset, keywords_present, True)
        print(sentence)
        print(f"is sentence cs? {result_binary}")
        print()
    mismatch = calculate_training_mismatch(dataset, classification_function, False)
    print(f"mismatch percentage: {mismatch}%")
    print()

def calculate_training_mismatch(dataset, class_function, print_flag):
    class_index = np.shape(dataset)[1]-1
    
    count = 0
    for row_idx in range(np.shape(dataset)[0]):
        result_count = class_function(dataset, dataset[row_idx,:class_index-1], print_flag)
        if result_count == dataset[row_idx, class_index]:
            count += 1
    return (1 - (count / np.shape(dataset)[0])) * 100
    
def bayes_classification(dataset, keywords, print_flag):
    class_index = np.shape(dataset)[1]-1
    cs_dataset = dataset[dataset[:, class_index] > 0]
    cs = bayes_classification_inner(dataset, keywords, cs_dataset, class_index)
    
    non_cs_dataset = dataset[dataset[:,class_index]==0]
    non_cs = bayes_classification_inner(dataset, keywords, non_cs_dataset, class_index)
    
    if print_flag:
        print(f"cs:\t{cs}")
        print(f"non_cs:\t{non_cs}")
    
    return cs > non_cs
    
def bayes_classification_inner(dataset, keywords, cs_dataset, class_index):
    num_rows = np.shape(cs_dataset)[0]
    class_cond = 1
    for idx in range(class_index-1):
        temp = np.count_nonzero(cs_dataset[:, idx] == keywords[idx]) / num_rows
        
        #Smooth if 0
        if temp == 0:
            # find max number in column 
            max_col = dataset[:, idx].max()
            temp = 1 / (num_rows + max_col + 1)
        class_cond *= temp
    
    prior = num_rows / np.shape(dataset)[0]
    
    return class_cond * prior

def bayes_continuous_classification(dataset, keywords, print_flag):
    class_index = np.shape(dataset)[1]-1
    cs_dataset = dataset[dataset[:, class_index] > 0]
    cs = bayes_continuous_classification_inner(dataset, keywords, cs_dataset, class_index)
    
    non_cs_dataset = dataset[dataset[:,class_index]==0]
    non_cs = bayes_continuous_classification_inner(dataset, keywords, non_cs_dataset, class_index)
    
    if print_flag:
        print(f"cs:\t{cs}")
        print(f"non_cs:\t{non_cs}")
    
    return cs > non_cs

def bayes_continuous_classification_inner(dataset, keywords, cs_dataset, class_index):
    num_rows = np.shape(cs_dataset)[0]
    class_cond = 1
    for idx in range(class_index-1):
        mean = cs_dataset[:, idx].mean()
        std_dev = cs_dataset[:, idx].std()
        x = keywords[idx]
        if std_dev != 0:
            temp = 1/(std_dev * math.sqrt(2 * math.pi)) * math.e**(-0.5*((x - mean)/std_dev)**2)
        else:
            temp = 0
            
        #Smooth if 0
        if temp == 0:
            # find max number in column 
            max_col = dataset[:, idx].max()
            temp = 1 / (num_rows + max_col + 1)
        class_cond *= temp
    
    prior = num_rows / np.shape(dataset)[0]
    
    return class_cond * prior
    
def convert_to_keywords_binary(test):
    test_array = []
    test = test.lower()
    test_array.append((0,1) ["computer" in test])
    test_array.append((0,1) ["science" in test])
    test_array.append((0,1) ["they" in test])
    test_array.append((0,1) ["school" in test])
    test_array.append((0,1) ["puppy" in test])
    test_array.append((0,1) ["software" in test])
    return test_array 
    
def convert_to_keywords_count(test): 
    test_array = []
    test = test.lower()
    test_array.append(test.count("computer"))
    test_array.append(test.count("science"))
    test_array.append(test.count("they"))
    test_array.append(test.count("school"))
    test_array.append(test.count("puppy"))
    test_array.append(test.count("software"))
    return test_array 
    
# run program
main()