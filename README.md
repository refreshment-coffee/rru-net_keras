# rru-net_keras

refer to http://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Bi_RRU-Net_The_Ringed_Residual_U-Net_for_Image_Splicing_Forgery_Detection_CVPRW_2019_paper.html

!!!!! GroupNormalization is in GN.PY put it down "keras.layers.normalization" down BN

problems: two formulas: yf = F(x,{Wi}) + Ws ∗ x, yb = (s(G(yf)) + 1) ∗ x G is a linear projection, which is used to change the dimensions of yf G is what mean ? I can't know,I regard G as W (I think the function of G is same as W)
