import ktrain
from ktrain import text

def perform_data_preprocessing(transformer, X_train, y_train, X_test, y_test):
    # Preprocess the training data using the provided transformer
    train = transformer.preprocess_train(X_train.to_list(), y_train.to_list())
    
    # Preprocess the testing data using the provided transformer
    val = transformer.preprocess_test(X_test.to_list(), y_test.to_list())
    
    # Return the preprocessed training and testing data
    return train, val
