
import os
import cv2
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import time

def char_list_():
    char_list = [' ',
                '#',
                "'",
                '(',
                ')',
                '+',
                ',',
                '-',
                '.',
                '/',
                '0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                ':',
                'A',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G',
                'H',
                'I',
                'J',
                'K',
                'L',
                'M',
                'N',
                'O',
                'P',
                'Q',
                'R',
                'S',
                'T',
                'U',
                'V',
                'W',
                'X',
                'Y',
                'a',
                'b',
                'c',
                'd',
                'e',
                'g',
                'h',
                'i',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
                'u',
                'v',
                'w',
                'x',
                'y',
                'z',
                'Â',
                'Ê',
                'Ô',
                'à',
                'á',
                'â',
                'ã',
                'è',
                'é',
                'ê',
                'ì',
                'í',
                'ò',
                'ó',
                'ô',
                'õ',
                'ù',
                'ú',
                'ý',
                'ă',
                'Đ',
                'đ',
                'ĩ',
                'ũ',
                'Ơ',
                'ơ',
                'ư',
                'ạ',
                'ả',
                'ấ',
                'ầ',
                'ẩ',
                'ậ',
                'ắ',
                'ằ',
                'ẵ',
                'ặ',
                'ẻ',
                'ẽ',
                'ế',
                'ề',
                'ể',
                'ễ',
                'ệ',
                'ỉ',
                'ị',
                'ọ',
                'ỏ',
                'ố',
                'ồ',
                'ổ',
                'ỗ',
                'ộ',
                'ớ',
                'ờ',
                'ở',
                'ỡ',
                'ợ',
                'ụ',
                'ủ',
                'Ứ',
                'ứ',
                'ừ',
                'ử',
                'ữ',
                'ự',
                'ỳ',
                'ỵ',
                'ỷ',
                'ỹ']

    return char_list

char_list = char_list_()
(min_height, max_height, min_width, max_width) = (94, 376, 955, 2694)

TIME_STEPS = 240
max_label_len = TIME_STEPS

def Model_RCNN(char_list):
    # OUR FULL MODEL OF CRNN AND LSTM

    # input with shape of height=32 and width=128 
    inputs = Input(shape=(118,2167,1))
    
    # Block 1
    x = Conv2D(64, (3,3), padding='same')(inputs)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_1 = x 

    # Block 2
    x = Conv2D(128, (3,3), padding='same')(x)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_2 = x

    # Block 3
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_3 = x

    # Block4
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,x_3])
    x = Activation('relu')(x)
    x_4 = x

    # Block5
    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_5 = x

    # Block6
    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,x_5])
    x = Activation('relu')(x)

    # Block7
    x = Conv2D(1024, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 1))(x)
    x = Activation('relu')(x)

    # pooling layer with kernel size (2,2) to make the height/2 #(1,9,512)
    x = MaxPool2D(pool_size=(3, 1))(x)
    
    # # to remove the first dimension of one: (1, 31, 512) to (31, 512) 
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)
    
    # # # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)

    # # this is our softmax character proprobility with timesteps 
    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

    # model to be used at test time

    act_model = Model(inputs, outputs)
    return act_model

act_model = Model_RCNN(char_list)
act_model.load_weights(os.path.join('checkpoint_weights_expand_12_12.hdf5'))

def Get_path_img():
    path = 'E:\\Yumi\\Do_an_2\\data_7_12_light_6'
    path_img = ''
    for i in os.listdir(path):
        path_img = i
    return path +'\\'+ path_img

def pre_process():
    #lists for validation dataset
    valid_img =[]
    resize_max_width=0

    i=0
    path = Get_path_img()
    #path = "E:\\Yumi\\data_30_11\\Image00016.JPG"
    #path = "E:\\Yumi\\img_pro\\vietnamese-handwriting-recognition-ocr-main\\data_expand\\0150_samples.png"
    #path = "C:\\Users\\My_Laptop\\OneDrive - vnu.edu.vn\\Desktop\\chu-ky-dep-ten-tuan.jpg"
    #path ='E:\\Yumi\\Do_an_2\\data_7_12_cropped\\0001_abc_12.jpg'

    # print(f_name)
    # read input image and convert into gray scale image
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    #img = img[700:930,99:1200]
    #img = img[420:600,190:1000] #data 07-12
    #img = img[420:650,190:1000]
    #img = img[420:650,220:1000]
    img = img[420:620,190:1000] #new 14_12
    height, width = img.shape
    # in this dataset, we don't need to do any resize at all here.
    img = cv2.resize(img,(int(118/height*width),118))

    height, width = img.shape
    if img.shape[1] > resize_max_width:
        resize_max_width = img.shape[1]
        
    img = np.pad(img, ((0,0),(0, 2167-width)), 'median')

    # YOUR PART: Blur it
    img = cv2.GaussianBlur(img, (5,5), 0)

    # YOUR PART: Threshold the image using adapative threshold
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 4) #for abb
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4) #defauld
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    # add channel dimension
    img = np.expand_dims(img , axis = 2)

    # Normalize each image
    img = img/255.

    valid_img.append(img)
    valid_img = np.array(valid_img)

    return img, valid_img

#img, valid_img = pre_process()
time_start = time.time()

def predict_name():

    img, valid_img = pre_process()

    NO_PREDICTS = 100
    OFFSET=0
    # prediction = act_model.predict(valid_img[OFFSET:OFFSET+NO_PREDICTS])
    prediction = act_model.predict(valid_img[0:1])

    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1], greedy=True)[0][0])
    # see the results
    all_predictions =[]
    i = 0
    for x in out:
    #  print("original_text  = ", valid_orig_txt[i+OFFSET])
      #  print("predicted text = ", end = '')
        pred = ""
        for p in x:  
            if int(p) != -1:
                pred += char_list[int(p)]
      #  print(pred)
        all_predictions.append(pred) 
        i+=1
    return str(all_predictions[0])

print(predict_name())
time_end = time.time()
print(round(time_end-time_start,4))
