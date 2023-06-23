import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tensorflow import keras
from tensorflow.keras import layers

hotel = pd.read_csv('Module 4 (Deep Learning)\hotel.csv')

X = hotel.copy()
y = X.pop('is_canceled')

X['arrival_date_month'] = X['arrival_date_month'].map({'January': 1, 'February': 2, 'March': 3,
                                                       'April': 4, 'May': 5, 'June': 6, 'July': 7,
                                                       'August': 8, 'September': 9, 'October': 10,
                                                       'November': 11, 'December': 12})

features_num = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
                "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]
features_cat = ["hotel", "arrival_date_month", "meal", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type"]

transformer_num = make_pipeline(SimpleImputer(strategy="constant"), StandardScaler())
transformer_cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                                OneHotEncoder(handle_unknown='ignore'))

preprocessor = make_column_transformer((transformer_num, features_num), (transformer_cat, features_cat))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]

model = LogisticRegression()
model.fit(X_train, y_train)

# Complete the code below this line
y_train_pred = model.predict_proba(X_train)[:, 1]
y_valid_pred = model.predict_proba(X_valid)[:, 1]
train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_valid, y_valid_pred)

# Check the ROC AUC scores
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC: {test_auc:.4f}")