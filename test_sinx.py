from utils.training import train 
from models.sin_x import create_model
from utils.dataloader import get_mnist 

def main():

    error_list = [] 
    for i in range(20):
        (X_train, y_train), (X_test, y_test), num_classes = get_mnist() 
        model = create_model() 
        test_error, history = train(model, (X_train, y_train), (X_test, y_test), epochs=10)
        error_list.append(test_error) 

    print "20 runs over.." 
    print error_list 

if __name__ == '__main__':
    main() 



