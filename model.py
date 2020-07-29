
from keras import Input, Model

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Activation, Reshape, BatchNormalization, Add, \
    Conv2DTranspose, GroupNormalization
from keras.layers import Concatenate, Add, Multiply, UpSampling2D, Lambda
from keras.regularizers import l2

G=1

def GetModel(input_shape):
    input_shape= Input(shape=input_shape)

    #第一个
    c1_1=input_shape
    

    c1_2= Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c1_1)
    c1_2= GroupNormalization(G)(c1_2)
    c1_2= Activation('relu')(c1_2)


    c1_3 = Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c1_2)
    c1_3 = GroupNormalization(G)(c1_3)

    #residual propagation
    #yf=w2*relu*(w1*x)+ws*X
    temp_c1_1 = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding='same')(c1_1) #ws*X
    #X = Add()([X, X_shortcut])
    yf_1=Add()([temp_c1_1,c1_3])#temp=w2*relu*(w1*x)+ws*X
    c1_3_f=Activation('relu')(yf_1)#最后添加relu c1_3_final 最终的c1_3


    #residual feedback  然后直接到c1_3_f 算一圈完毕
    ##yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(3, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_1)#G(yf)

   # yb_1=(Activation("sigmoid")(t2)+1)*c1_1#x=c1_1    报错，卷积层中无法进行加减乘除运算,要用对应的层
    t1=Activation("sigmoid")(t2)
    t1=Multiply()([t1,c1_1])
    yb_1=Add()([t1,c1_1])

    c1_2= Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_1)
    c1_2=GroupNormalization(G)(c1_2)
    c1_2=Activation('relu')(c1_2)

    c1_3_f=Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c1_2)
    c1_3_f=GroupNormalization(G)(c1_3_f)

    #在做一次残差网络
    # c1_3_f=Activation('relu')(c1_3_f)
    temp_c1_1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c1_1)
    c1_3_f = Add()([temp_c1_1, c1_3_f])
    c1_3_f=Activation('relu')(c1_3_f)







    #第二个

    c2_1=MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same')(c1_3_f)
    #print(c2_1)

    c2_2 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c2_1)
    c2_2 = GroupNormalization(G)(c2_2)
    c2_2 = Activation('relu')(c2_2)

    c2_3 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c2_2)
    c2_3 = GroupNormalization(G)(c2_3)
    #c2_3 = Activation(activation='relu')(c2_3) #注意这里没有relu

    # residual propagation
    # yf=w2*relu*(w1*x)+ws*X
    temp_c2_1 = Conv2D(64, kernel_size=(1, 1), strides=(1,1), padding='same')(c2_1)  # Ws*X    同维度
    # X = Add()([X, X_shortcut])
    yf_2 = Add()([temp_c2_1, c2_3])  # temp=w2*relu*(w1*x)+ws*X
    c2_3_f = Activation('relu')(yf_2)  # 最后添加relu c2_3_final 最终的c2_3

    # residual feedback  ????
    # yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_2)  # G(yf)
    #yb_2 = (Activation("sigmoid")(t2) + 1) * c2_1  # x=c2_1

    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c2_1])
    yb_2 = Add()([t1, c2_1])



    c2_2 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_2)
    c2_2 = GroupNormalization(G)(c2_2)
    c2_2 = Activation('relu')(c2_2)


    c2_3_f = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c2_2)
    c2_3_f = GroupNormalization(G)(c2_3_f)

    # 在做一次残差网络
    # c2_3_f=Activation('relu')(c2_3_f)
    temp_c2_1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(c2_1)
    c2_3_f = Add()([temp_c2_1, c2_3_f])
    c2_3_f = Activation('relu')(c2_3_f)

    #第三个
    c3_1=MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same')(c2_3_f)


    c3_2 = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c3_1)
    c3_2 = GroupNormalization(G)(c3_2)
    c3_2 = Activation('relu')(c3_2)

    c3_3 = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c3_2)
    c3_3 = GroupNormalization(G)(c3_3)

    # c3_3 = Activation(activation='relu')(c3_3) #注意这里没有relu

    # residual propagation
    # yf=w2*relu*(w1*x)+ws*X
    temp_c3_1 = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same')(c3_1)  # Ws*X    同维度
    # X = Add()([X, X_shortcut])
    yf_3 = Add()([temp_c3_1, c3_3])  # temp=w2*relu*(w1*x)+ws*X
    c3_3_f = Activation('relu')(yf_3)  # 最后添加relu c3_3_final 最终的c3_3

    # residual feedback  ????
    # yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(64, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_3)  # G(yf)
    #yb_3 = (Activation("sigmoid")(t2) + 1) * c3_1  # x=c3_1

    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c3_1])
    yb_3 = Add()([t1, c3_1])


    c3_2 = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_3)

    c3_2 = GroupNormalization(G)(c3_2)
    c3_2 = Activation('relu')(c3_2)

    c3_3_f = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c3_2)
    c3_3_f = GroupNormalization(G)(c3_3_f)


    # 在做一次残差网络
    # c3_3_f=Activation('relu')(c3_3_f)
    temp_c3_1 = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(c3_1)
    c3_3_f = Add()([temp_c3_1, c3_3_f])
    c3_3_f = Activation('relu')(c3_3_f)

    #第四个
    c4_1=MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same')(c3_3_f)

    c4_2 = Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c4_1)
    c4_2 = GroupNormalization(G)(c4_2)
    c4_2 = Activation('relu')(c4_2)

    c4_3 = Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c4_2)
    c4_3 = GroupNormalization(G)(c4_3)
    # c4_3 = Activation(activation='relu')(c4_3) #注意这里没有relu


    # residual propagation
    # yf=w2*relu*(w1*x)+ws*X
    temp_c4_1 = Conv2D(256, kernel_size=(1, 1), strides=(1,1), padding='same')(c4_1)  # Ws*X    同维度
    # X = Add()([X, X_shortcut])
    yf_4 = Add()([temp_c4_1, c4_3])  # temp=w2*relu*(w1*x)+ws*X
    c4_3_f = Activation('relu')(yf_4)  # 最后添加relu c4_3_final 最终的c4_3

    # residual feedback  ????
    # yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_4)  # G(yf)
    #yb_4 = (Activation("sigmoid")(t2) + 1) * c4_1  # x=c4_1

    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c4_1])
    yb_4 = Add()([t1, c4_1])


    c4_2 = Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_4)

    c4_2 = GroupNormalization(G)(c4_2)
    c4_2 = Activation('relu')(c4_2)

    c4_3_f = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c4_2)
    c4_3_f = GroupNormalization(G)(c4_3_f)


    # 在做一次残差网络
    # c4_3_f=Activation('relu')(c4_3_f)
    temp_c4_1 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(c4_1)
    c4_3_f = Add()([temp_c4_1, c4_3_f])
    c4_3_f = Activation('relu')(c4_3_f)

    #第五个
    c5_1=MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same')(c4_3_f)

    c5_2 = Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c5_1)
    c5_2 = GroupNormalization(G)(c5_2)
    c5_2 = Activation('relu')(c5_2)

    c5_3 = Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c5_2)
    c5_3 = GroupNormalization(G)(c5_3)
    # c5_3 = Activation(activation='relu')(c5_3) #注意这里没有relu

    # residual propagation
    # yf=w2*relu*(w1*x)+ws*X
    temp_c5_1 = Conv2D(512, kernel_size=(1, 1), strides=(1,1), padding='same')(c5_1)  # Ws*X    同维度
    # X = Add()([X, X_shortcut])
    yf_5 = Add()([temp_c5_1, c5_3])  # temp=w2*relu*(w1*x)+ws*X
    c5_3_f = Activation('relu')(yf_5)  # 最后添加relu c5_3_final 最终的c5_3

    # residual feedback  ????
    # yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(256, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_5)  # G(yf)
    yb_5 = (Activation("sigmoid")(t2) + 1) *c5_1  # x=c5_1

    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c5_1])
    yb_5 = Add()([t1, c5_1])

    c5_2 = Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_5)

    c5_2 = GroupNormalization(G)(c5_2)
    c5_2 = Activation('relu')(c5_2)

    c5_3_f = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c5_2)
    c5_3_f = GroupNormalization(G)(c5_3_f)

    # 在做一次残差网络
    # c5_3_f=Activation('relu')(c5_3_f)
    temp_c5_1 = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same')(c5_1)
    c5_3_f = Add()([temp_c5_1, c5_3_f])
    c5_3_f = Activation('relu')(c5_3_f)


    c5_3_f=Conv2DTranspose(256,kernel_size=(3,3),strides=(2,2),padding='same',name='Conc2DTranspose_1')(c5_3_f)
    c5_3_f=GroupNormalization(G)(c5_3_f)
    c5_3_f=Activation('relu')(c5_3_f)

    #第六个--第一个

    c6_1=Concatenate()([c4_3_f,c5_3_f])




    c6_2=Conv2D(128,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c6_1)
    c6_2 = GroupNormalization(G)(c6_2)
    c6_2 = Activation('relu')(c6_2)

    c6_3 = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(c6_2)
    c6_3 = GroupNormalization(G)(c6_3)

    # residual propagation
    # yf=w2*relu*(w1*x)+ws*X
    temp_c6_1 = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same')(c6_1)  # Ws*X    同维度
    # X = Add()([X, X_shortcut])
    yf_6 = Add()([temp_c6_1, c6_3])  # temp=w2*relu*(w1*x)+ws*X
    #c6_3_f = Activation('relu')(yf_6)  # 最后添加relu c4_3_final 最终的c4_3



    # residual feedback  ????
    # yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(512, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_6)  # G(yf)

   ## yb_6 = (Activation("sigmoid")(t2) + 1) * c6_1  # x=c5_1

    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c6_1])
    yb_6 = Add()([t1, c6_1])



    c6_2 = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(yb_6)
    #
    c6_2 = GroupNormalization(G)(c6_2)
    c6_2 = Activation('relu')(c6_2)

    c6_3_f = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c6_2)
    c6_3_f = GroupNormalization(G)(c6_3_f)

    # 在做一次残差网络
    # c6_3_f=Activation('relu')(c6_3_f)
    temp_c6_1 = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(c6_1)  # Ws*X    同维度
    c6_3_f = Add()([temp_c6_1, c6_3_f])
    c6_3_f = Activation('relu')(c6_3_f)


    c6_3_f=Conv2DTranspose(128,kernel_size=(3,3),strides=(2,2),padding='same')(c6_3_f)
    c6_3_f=GroupNormalization(G)(c6_3_f)
    c6_3_f=Activation('relu')(c6_3_f)


    #第七个--第二个
    c7_1=Concatenate()([c3_3_f,c6_3_f])

    c7_2=Conv2D(64,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c7_1)
    c7_2=GroupNormalization(G)(c7_2)
    c7_2=Activation('relu')(c7_2)

    c7_3=Conv2D(64,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c7_2)
    c7_3=GroupNormalization(G)(c7_3)

    # residual propagation
    # yf=w2*relu*(w1*x)+ws*X
    temp_c7_1 = Conv2D(64, kernel_size=(1, 1), strides=(1,1), padding='same')(c7_1)  # Ws*X    同维度
    # X = Add()([X, X_shortcut])
    yf_7 = Add()([temp_c7_1, c7_3])  # temp=w2*relu*(w1*x)+ws*X
    c7_3_f = Activation('relu')(yf_7)  # 最后添加relu c4_3_final 最终的c4_3

    # residual feedback  ????
    # yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(256, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_7)  # G(yf)
    yb_7 = (Activation("sigmoid")(t2) + 1) * c7_1  # x=c5_1

    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c7_1])
    yb_7 = Add()([t1, c7_1])



    c7_2 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(yb_7)
    c7_2 = GroupNormalization(G)(c7_2)
    c7_2 = Activation('relu')(c7_2)

    c7_3_f = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c7_2)
    c7_3_f = GroupNormalization(G)(c7_3_f)


    # 在做一次残差网络
    # c7_3_f=Activation('relu')(c7_3_f)
    temp_c7_1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(c7_1)  # Ws*X    同维度
    c7_3_f = Add()([temp_c7_1, c7_3_f])
    c7_3_f = Activation('relu')(c7_3_f)

    c7_3_f=Conv2DTranspose(64,kernel_size=(3,3),strides=(2,2), padding='same')(c7_3_f)
    c7_3_f=GroupNormalization(G)(c7_3_f) #GroupNormalization(G) 会报错，不接受(none,none,dims)
    c7_3_f=Activation('relu')(c7_3_f)

    #第八--第三
    c8_1=Concatenate()([c2_3_f,c7_3_f])

    c8_2=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c8_1)
    c8_2=GroupNormalization(G)(c8_2)
    c8_2=Activation('relu')(c8_2)

    c8_3=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c8_2)
    c8_3=GroupNormalization(G)(c8_2)


    # residual propagation
    # yf=w2*relu*(w1*x)+ws*X
    temp_c8_1 = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding='same')(c8_1)  # Ws*X    同维度
    # X = Add()([X, X_shortcut])
    yf_8 = Add()([temp_c8_1, c8_3])  # temp=w2*relu*(w1*x)+ws*X
    c8_3_f = Activation('relu')(yf_8)  # 最后添加relu c4_3_final 最终的c4_3

    # residual feedback  ????
    # yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_8)  # G(yf)
    yb_8 = (Activation("sigmoid")(t2) + 1) * c8_1  # x=c5_1

    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c8_1])
    yb_8 = Add()([t1, c8_1])


    c8_2 = Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(yb_8)

    c8_2 = GroupNormalization(G)(c8_2)
    c8_2 = Activation('relu')(c8_2)

    c8_3_f = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c8_2)
    c8_3_f = GroupNormalization(G)(c8_3_f)

    # 在做一次残差网络
    # c8_3_f=Activation('relu')(c8_3_f)
    temp_c8_1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c8_1)  # Ws*X    同维度
    c8_3_f = Add()([temp_c8_1, c8_3_f])
    c8_3_f = Activation('relu')(c8_3_f)



    c8_3_f=Conv2DTranspose(32,kernel_size=(3,3),strides=(2,2), padding='same')(c8_3_f)
    c8_3_f=GroupNormalization(G)(c8_3_f)
    c8_3_f=Activation('relu')(c8_3_f)
    #第九--第四
    c9_1=Concatenate()([c1_3_f,c8_3_f])

    c9_2=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c9_1)
    c9_2=GroupNormalization(G)(c9_2)
    c9_2=Activation('relu')(c9_2)

    c9_3=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c9_2)
    c9_3=GroupNormalization(G)(c9_3)


    # residual propagation
    # yf=w2*relu*(w1*x)+ws*X
    temp_c9_1 = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding='same')(c9_1)  # Ws*X    同维度
    # X = Add()([X, X_shortcut])
    yf_9 = Add()([temp_c9_1, c9_3])  # temp=w2*relu*(w1*x)+ws*X
    c9_3_f = Activation('relu')(yf_9)  # 最后添加relu c4_3_final 最终的c4_3

    # residual feedback  ????
    # yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(64, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_9)  # G(yf)
    yb_9 = (Activation("sigmoid")(t2) + 1) * c9_1  # x=c5_1

    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c9_1])
    yb_9 = Add()([t1, c9_1])

    c9_2 = Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(yb_9)
    c9_2 = GroupNormalization(G)(c9_2)
    c9_2 = Activation('relu')(c9_2)

    c9_3_f = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c9_2)
    c9_3_f = GroupNormalization(G)(c9_3_f)

    # 在做一次残差网络
    temp_c9_1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c9_1)  # Ws*X    同维度
    # c9_3_f=Activation('relu')(c9_3_f)
    c9_3_f = Add()([temp_c9_1, c9_3_f])
    c9_3_f = Activation('relu')(c9_3_f)



    #last
    #last=Conv2D(1,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2,activation='sigmoid',kernel_regularizer=l2(0.0005))(c9_3_f)
    last = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='sigmoid')(c9_3_f)

    model=Model(inputs=input_shape,outputs=last)

    return model


# model=GetModel((512,512,3))
# model.summary()
