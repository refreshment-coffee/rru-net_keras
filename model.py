from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization,Conv2DTranspose, GroupNormalization
from keras.layers import Concatenate, Add, Multiply
from keras.regularizers import l2

G=1#    G=Batch_Size

def GetModel(input_shape):
    input_shape= Input(shape=input_shape)

    #第一个
    c1_1=input_shape
    

    c1_2= Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c1_1)
    c1_2= GroupNormalization(G)(c1_2)
    c1_2= Activation('relu')(c1_2)


    c1_3 = Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c1_2)
    c1_3 = GroupNormalization(G)(c1_3)
    #注意c1_3不要激活函数

    #residual propagation
    #yf=w2*relu*(w1*x)+ws*X
    temp_c1_1 = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding='same')(c1_1) #ws*X
    yf_1=Add()([temp_c1_1,c1_3])#temp=w2*relu*(w1*x)+ws*X
    yf_1=Activation('relu')(yf_1)       #未知要不要用激活函数


    #residual feedback  然后直接到c1_3_f 算一圈完毕
    ##yb=(sigmoid(G(yf)+1)*x
    t2 = Conv2D(3, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_1)#G(yf)
   # yb_1=(Activation("sigmoid")(t2)+1)*c1_1#x=c1_1    报错，卷积层中无法进行加减乘除运算,要用对应的层
    #yb_1=(Activation("sigmoid)(t2))*c1_1+c1_1 化解成这样


    t1=Activation("sigmoid")(t2)
    t1=Multiply()([t1,c1_1])
    yb_1=Add()([t1,c1_1])

    c1_1=yb_1 #修改最初的输入



    c1_2= Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_1)
    c1_2=GroupNormalization(G)(c1_2)
    c1_2=Activation('relu')(c1_2)

    c1_3_f=Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c1_2)
    c1_3_f=GroupNormalization(G)(c1_3_f)


    #在做一次残差网络
    temp_c1_1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c1_1)
    c1_3_f = Add()([temp_c1_1, c1_3_f])
    c1_3_f=Activation('relu')(c1_3_f)



    #第二个

    c2_1=MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same')(c1_3_f)

    c2_2 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c2_1)
    c2_2 = GroupNormalization(G)(c2_2)
    c2_2 = Activation('relu')(c2_2)

    c2_3 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(c2_2)
    c2_3 = GroupNormalization(G)(c2_3)

    # residual propagation
    temp_c2_1 = Conv2D(64, kernel_size=(1, 1), strides=(1,1), padding='same')(c2_1)  # Ws*X    同维度
    yf_2 = Add()([temp_c2_1, c2_3])  # temp=w2*relu*(w1*x)+ws*X
    yf_2=Activation("relu")(yf_2)


    # residual feedback
    t2 = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_2)  # G(yf)
    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c2_1])
    yb_2 = Add()([t1, c2_1])


    c2_1=yb_2


    c2_2 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_2)
    c2_2 = GroupNormalization(G)(c2_2)
    c2_2 = Activation('relu')(c2_2)


    c2_3_f = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c2_2)
    c2_3_f = GroupNormalization(G)(c2_3_f)

    # 在做一次残差网络
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



    # residual propagation
    temp_c3_1 = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same')(c3_1)  # Ws*X    同维度
    yf_3 = Add()([temp_c3_1, c3_3])  # temp=w2*relu*(w1*x)+ws*X
    yf_3 = Activation('relu')(yf_3)

    # residual feedback
    t2 = Conv2D(64, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_3)  # G(yf)
    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c3_1])
    yb_3 = Add()([t1, c3_1])


    c3_1 = yb_3


    c3_2 = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_3)
    c3_2 = GroupNormalization(G)(c3_2)
    c3_2 = Activation('relu')(c3_2)

    c3_3_f = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c3_2)
    c3_3_f = GroupNormalization(G)(c3_3_f)


    # 在做一次残差网络
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

    # residual propagation
    temp_c4_1 = Conv2D(256, kernel_size=(1, 1), strides=(1,1), padding='same')(c4_1)  # Ws*X    同维度
    yf_4 = Add()([temp_c4_1, c4_3])  # temp=w2*relu*(w1*x)+ws*X
    yf_4 = Activation('relu')(yf_4)

    # residual feedback
    t2 = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_4)  # G(yf)
    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c4_1])
    yb_4 = Add()([t1, c4_1])


    c4_1=yb_4


    c4_2 = Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_4)
    c4_2 = GroupNormalization(G)(c4_2)
    c4_2 = Activation('relu')(c4_2)

    c4_3_f = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c4_2)
    c4_3_f = GroupNormalization(G)(c4_3_f)


    # 在做一次残差网络
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


    # residual propagation
    temp_c5_1 = Conv2D(512, kernel_size=(1, 1), strides=(1,1), padding='same')(c5_1)  # Ws*X    同维度
    yf_5 = Add()([temp_c5_1, c5_3])  # temp=w2*relu*(w1*x)+ws*X
    yf_5 = Activation('relu')(yf_5)

    # residual feedback
    t2 = Conv2D(256, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_5)  # G(yf)
    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c5_1])
    yb_5 = Add()([t1, c5_1])


    c5_1=yb_5


    c5_2 = Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=2)(yb_5)
    c5_2 = GroupNormalization(G)(c5_2)
    c5_2 = Activation('relu')(c5_2)

    c5_3_f = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c5_2)
    c5_3_f = GroupNormalization(G)(c5_3_f)

    # 在做一次残差网络
    temp_c5_1 = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same')(c5_1)
    c5_3_f = Add()([temp_c5_1, c5_3_f])
    c5_3_f = Activation('relu')(c5_3_f)


    c5_3_f=Conv2DTranspose(256,kernel_size=(3,3),strides=(2,2),padding='same',name='Conc2DTranspose_1')(c5_3_f)
    c5_3_f=GroupNormalization(G)(c5_3_f)

    #第六个--第一个
    c6_1=Concatenate()([c4_3_f,c5_3_f])
    c6_1 = Activation('relu')(c6_1)

    c6_2=Conv2D(128,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c6_1)
    c6_2 = GroupNormalization(G)(c6_2)
    c6_2 = Activation('relu')(c6_2)

    c6_3 = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(c6_2)
    c6_3 = GroupNormalization(G)(c6_3)

    # residual propagation
    temp_c6_1 = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same')(c6_1)  # Ws*X    同维度
    yf_6 = Add()([temp_c6_1, c6_3])  # temp=w2*relu*(w1*x)+ws*X
    yf_6=Activation('relu')(yf_6)




    # residual feedback
    t2 = Conv2D(512, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_6)  # G(yf)
    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c6_1])
    yb_6 = Add()([t1, c6_1])


    c6_1=yb_6


    c6_2 = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(yb_6)
    c6_2 = GroupNormalization(G)(c6_2)
    c6_2 = Activation('relu')(c6_2)

    c6_3_f = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c6_2)
    c6_3_f = GroupNormalization(G)(c6_3_f)

    # 在做一次残差网络
    temp_c6_1 = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(c6_1)  # Ws*X    同维度
    c6_3_f = Add()([temp_c6_1, c6_3_f])
    c6_3_f = Activation('relu')(c6_3_f)


    c6_3_f=Conv2DTranspose(128,kernel_size=(3,3),strides=(2,2),padding='same')(c6_3_f)
    c6_3_f=GroupNormalization(G)(c6_3_f)



    #第七个--第二个
    c7_1=Concatenate()([c3_3_f,c6_3_f])
    c7_1 = Activation('relu')(c7_1)


    c7_2=Conv2D(64,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c7_1)
    c7_2=GroupNormalization(G)(c7_2)
    c7_2=Activation('relu')(c7_2)

    c7_3=Conv2D(64,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c7_2)
    c7_3=GroupNormalization(G)(c7_3)

    # residual propagation
    temp_c7_1 = Conv2D(64, kernel_size=(1, 1), strides=(1,1), padding='same')(c7_1)  # Ws*X    同维度
    yf_7 = Add()([temp_c7_1, c7_3])  # temp=w2*relu*(w1*x)+ws*X
    yf_7 = Activation('relu')(yf_7)

    # residual feedback
    t2 = Conv2D(256, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_7)  # G(yf)
    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c7_1])
    yb_7 = Add()([t1, c7_1])


    c7_1=yb_7


    c7_2 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(yb_7)
    c7_2 = GroupNormalization(G)(c7_2)
    c7_2 = Activation('relu')(c7_2)

    c7_3_f = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c7_2)
    c7_3_f = GroupNormalization(G)(c7_3_f)


    # 在做一次残差网络
    temp_c7_1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(c7_1)  # Ws*X    同维度
    c7_3_f = Add()([temp_c7_1, c7_3_f])
    c7_3_f = Activation('relu')(c7_3_f)

    c7_3_f=Conv2DTranspose(64,kernel_size=(3,3),strides=(2,2), padding='same')(c7_3_f)
    c7_3_f=GroupNormalization(G)(c7_3_f)


    #第八--第三
    c8_1=Concatenate()([c2_3_f,c7_3_f])
    c8_1 = Activation('relu')(c8_1)

    c8_2=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c8_1)
    c8_2=GroupNormalization(G)(c8_2)
    c8_2=Activation('relu')(c8_2)

    c8_3=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c8_2)
    c8_3=GroupNormalization(G)(c8_2)


    # residual propagation
    temp_c8_1 = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding='same')(c8_1)  # Ws*X    同维度
    yf_8 = Add()([temp_c8_1, c8_3])  # temp=w2*relu*(w1*x)+ws*X
    yf_8 = Activation('relu')(yf_8)

    # residual feedback
    t2 = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_8)  # G(yf)
    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c8_1])
    yb_8 = Add()([t1, c8_1])


    c8_1=yb_8


    c8_2 = Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(yb_8)
    c8_2 = GroupNormalization(G)(c8_2)
    c8_2 = Activation('relu')(c8_2)

    c8_3_f = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c8_2)
    c8_3_f = GroupNormalization(G)(c8_3_f)

    # 在做一次残差网络
    temp_c8_1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c8_1)  # Ws*X    同维度
    c8_3_f = Add()([temp_c8_1, c8_3_f])
    c8_3_f = Activation('relu')(c8_3_f)


    c8_3_f=Conv2DTranspose(32,kernel_size=(3,3),strides=(2,2), padding='same')(c8_3_f)
    c8_3_f=GroupNormalization(G)(c8_3_f)

    #第九--第四
    c9_1=Concatenate()([c1_3_f,c8_3_f])
    c9_1 = Activation('relu')(c9_1)

    c9_2=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c9_1)
    c9_2=GroupNormalization(G)(c9_2)
    c9_2=Activation('relu')(c9_2)

    c9_3=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2)(c9_2)
    c9_3=GroupNormalization(G)(c9_3)


    # residual propagation
    temp_c9_1 = Conv2D(32, kernel_size=(1, 1), strides=(1,1), padding='same')(c9_1)  # Ws*X    同维度
    yf_9 = Add()([temp_c9_1, c9_3])  # temp=w2*relu*(w1*x)+ws*X
    yf_9= Activation('relu')(yf_9)

    # residual feedback
    t2 = Conv2D(64, kernel_size=(1, 1), strides=(1,1), padding='same')(yf_9)  # G(yf)
    t1 = Activation("sigmoid")(t2)
    t1 = Multiply()([t1, c9_1])
    yb_9 = Add()([t1, c9_1])


    c9_1=yb_9


    c9_2 = Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same', dilation_rate=2)(yb_9)
    c9_2 = GroupNormalization(G)(c9_2)
    c9_2 = Activation('relu')(c9_2)

    c9_3_f = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2)(c9_2)
    c9_3_f = GroupNormalization(G)(c9_3_f)

    # 在做一次残差网络
    temp_c9_1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c9_1)  # Ws*X    同维度
    c9_3_f = Add()([temp_c9_1, c9_3_f])
    c9_3_f = Activation('relu')(c9_3_f)


    #last
    #last=Conv2D(1,kernel_size=(3,3),strides=(1,1), padding='same',dilation_rate=2,activation='sigmoid',kernel_regularizer=l2(0.0005))(c9_3_f)
    last = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='sigmoid')(c9_3_f)

    model=Model(inputs=input_shape,outputs=last)

    return model
