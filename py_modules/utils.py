import numpy as np


def to_categorical(labels):
    return np.argmax(labels, axis=1)


def to_one_hot(labels, n_classes=None):
    if n_classes is None:
        n_classes = labels.max() + 1
    one_hot = np.zeros((labels.size, n_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def train_test_split(X, y, test_size=0.2):
    """
    Splits the dataset into training and testing sets.
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    if not X.shape[0] == y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def batch_generator(X, y, batch_size, shuffle=True):
    """
    Generates batches of data.
    """
    n_samples = X.shape[0]
    
    if shuffle:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        if shuffle:
            yield X[indices[start:end]], y[indices[start:end]]
        else:
            yield X[start:end], y[start:end]
