from mlp_names import *


with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
    m, X_test, Y_test,int_to_char,char_to_int,block_size,training_names,test_names = pickle.load(f)

m.generate_name()