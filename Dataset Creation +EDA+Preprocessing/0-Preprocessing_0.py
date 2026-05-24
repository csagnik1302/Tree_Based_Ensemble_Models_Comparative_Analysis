import pandas as pd

data=pd.read_csv(r"D:\ML Project\SDSS_DR18.csv")


data1=data.drop(columns=['objid', 'specobjid', 'run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid'])   
# Removing unnecessary columns which only contain serial number/device data that was used to capture data. These type of data will only add npise or unnecessary patterns in the model so its better to remove them


# print(data1)

# print(data1.columns)

data1.to_csv(r"D:\ML Project\Data.csv",index=False)