import numpy as np

def train_simple_cnn_with_backprop(X, y, epochs, learning_rate, kernel_size=3, num_filters=1):
    '''
    Trains a simple CNN with one convolutional layer, ReLU activation, flattening, and a dense layer with softmax output using backpropagation.

    Assumes X has shape (n_samples, height, width) for grayscale images and y is one-hot encoded with shape (n_samples, num_classes).

    Parameters:
    X : np.ndarray, input data
    y : np.ndarray, one-hot encoded labels
    epochs : int, number of training epochs
    learning_rate : float, learning rate for weight updates
    kernel_size : int, size of the square convolutional kernel
    num_filters : int, number of filters in the convolutional layer

    Returns:
    W_conv, b_conv, W_dense, b_dense : Trained weights and biases for the convolutional and dense layers
    '''
    n_samples, height, width = X.shape
    num_classes = y.shape[1]

    # Initialize weights and biases
    W_conv = np.random.randn(kernel_size, kernel_size, num_filters) * 0.01
    b_conv = np.zeros(num_filters)
    output_height = height - kernel_size + 1
    output_width = width - kernel_size + 1
    flattened_size = output_height * output_width * num_filters
    W_dense = np.random.randn(flattened_size, num_classes) * 0.01
    b_dense = np.zeros(num_classes)

    # Your code here - implement forward pass, backpropagation, and weight updates
    for _ in range(epochs):
        for s in range(n_samples):
            x_s = X[s]
            y_s = y[s]
            conv_out = np.zeros((output_height,output_width,num_filters))
            for i in range(output_height):
                for j in range(output_width):
                    patch = x_s[i:i+kernel_size,j:j+kernel_size]
                    conv_out[i,j,:] = np.tensordot(patch,W_conv,axes=([0,1],[0,1]))+b_conv
            relu_out = np.maximum(0,conv_out)
            flat = relu_out.flatten()

            logits = flat@W_dense +b_dense
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits/np.sum(exp_logits)

            dlogits = probs -y_s
            dW_dense = np.outer(flat,dlogits)
            db_dense = dlogits

            dflat = dlogits@W_dense.T
            drelu = dflat.reshape(relu_out.shape)
            dconv = drelu*(conv_out>0)

            dW_conv = np.zeros_like(W_conv)
            db_conv = np.sum(dconv,axis=(0,1))
            for i in range(output_height):
                for j in range(output_width):
                    patch = x_s[i:i+kernel_size,j:j+kernel_size]
                    dW_conv += patch[:,:,np.newaxis]*dconv[i,j,:]
            
            W_conv -= learning_rate*dW_conv
            b_conv -= learning_rate*db_conv
            W_dense -= learning_rate*dW_dense
            b_dense -= learning_rate*db_dense



    
    return W_conv, b_conv, W_dense, b_dense