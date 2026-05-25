import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler

data=pd.read_csv(r"D:\ML Project\Dataset Creation +EDA+Preprocessing\Data_Feature_Engineered.csv")

data_Y1=data["class"]
data_X=data.drop(columns=["class"])


########### Label Encoding of Y ###################

Y_Encoder=LabelEncoder()

Y_Encoder.fit(data_Y1)

data_Y=pd.Series(Y_Encoder.transform(data_Y1),name='class')   # Since transform removes the header


############# Perform Train-Test-Validation Split #####################

X_train1,X_test,Y_train1,Y_test=train_test_split(data_X,data_Y,test_size=0.1,random_state=50)

X_train,X_val,Y_train,Y_val=train_test_split(X_train1,Y_train1,test_size=0.11111,random_state=50)

Y_train=pd.Series(Y_train)
Y_val=pd.Series(Y_val)
Y_test=pd.Series(Y_test)



############ Normalize Feature Data Splits (To remove any feature dominance issues) ###################

normalizer=StandardScaler().set_output(transform='pandas')

X_train=normalizer.fit_transform(X_train)
X_val=normalizer.transform(X_val)
X_test=normalizer.transform(X_test)


############# Exporting Splits #########################################

X_train.to_csv(r"D:\ML Project\Data Splits\X_train.csv",index=False)
X_val.to_csv(r"D:\ML Project\Data Splits\X_val.csv",index=False)
X_test.to_csv(r"D:\ML Project\Data Splits\X_test.csv",index=False)
Y_train.to_csv(r"D:\ML Project\Data Splits\Y_train.csv",index=False)
Y_val.to_csv(r"D:\ML Project\Data Splits\Y_val.csv",index=False)
Y_test.to_csv(r"D:\ML Project\Data Splits\Y_test.csv",index=False)
