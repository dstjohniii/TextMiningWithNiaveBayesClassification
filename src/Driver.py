import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def main():
    np.random.seed(4)

    # Get Data
    biniary_dataset = np.genfromtxt('BinaryData.csv', delimiter=',', skip_header=True, usecols=(0, 1, 2, 3, 4, 5, 6))
    non_biniary_dataset = np.genfromtxt('NonBinaryData.csv', delimiter=',', skip_header=True, usecols=(0, 1, 2, 3, 4, 5, 6))
    
    #Test Sentences
    
    test1 = "the puppy ate my computer science homework"
    
    #Convert to keywords
    keywords_present= convert_to_keywords_binary(test1)
    keywords_count = convert_to_keywords_count(test1)
    
    result_binary = bayes_classification(biniary_dataset, keywords_present)
    print(test1)
    print(f"is sentence cs? {result_binary}")
    print()
    
    result_count = bayes_continuous_classification(non_biniary_dataset, keywords_count)
    print(test1)
    print(f"is sentence cs? {result_count}")
    print()

def problem1(sentences):
    biniary_dataset = np.genfromtxt('BinaryData.csv', delimiter=',', skip_header=True, usecols=(0, 1, 2, 3, 4, 5, 6))
    keywords_present = convert_to_keywords_binary(test1)
    result_binary = bayes_classification(biniary_dataset, keywords_present)
    print(test1)
    print(f"is sentence cs? {result_binary}")
    print()

def problem2(sentences):
    non_biniary_dataset = np.genfromtxt('NonBinaryData.csv', delimiter=',', skip_header=True, usecols=(0, 1, 2, 3, 4, 5, 6))
    keywords_count = convert_to_keywords_count(test1)
    result_count = bayes_continuous_classification(non_biniary_dataset, keywords_count)
    print(test1)
    print(f"is sentence cs? {result_count}")
    print()
    
    
def bayes_classification(dataset, keywords):
    class_index = np.shape(dataset)[1]-1
    cs_dataset = dataset[dataset[:, class_index] > 0]
    cs = bayes_classification_inner(dataset, keywords, cs_dataset, class_index)
    
    non_cs_dataset = dataset[dataset[:,class_index]==0]
    non_cs = bayes_classification_inner(dataset, keywords, non_cs_dataset, class_index)
    
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

def bayes_continuous_classification(dataset, keywords):
    class_index = np.shape(dataset)[1]-1
    cs_dataset = dataset[dataset[:, class_index] > 0]
    cs = bayes_continuous_classification_inner(dataset, keywords, cs_dataset, class_index)
    
    non_cs_dataset = dataset[dataset[:,class_index]==0]
    non_cs = bayes_continuous_classification_inner(dataset, keywords, non_cs_dataset, class_index)
    
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