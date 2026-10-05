import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

DATA_DIR = r"c:\Users\Javier\Desktop\kaggle\02_spaceship_titanic\data"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
n_train = len(train)
n_test = len(test)
df = pd.concat([train, test], sort=False).reset_index(drop=True)

df['GroupId_int'] = df['PassengerId'].apply(lambda x: int(x.split('_')[0]))
df['GroupId'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
df['GroupNum'] = df['PassengerId'].apply(lambda x: int(x.split('_')[1]))
df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

df['LastName'] = df['Name'].str.split().str[-1]
df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count').fillna(1)

cabin_split = df['Cabin'].str.split('/', expand=True)
df['Deck'] = cabin_split[0]
df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
df['Side'] = cabin_split[2]

expenses = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in expenses:
    df.loc[df['Age'] < 13, col] = 0
    df.loc[df['CryoSleep'] == True, col] = 0
    df[col] = df[col].fillna(0)

df['TotalExpenses'] = df[expenses].sum(axis=1)
df['NoExpenses'] = (df['TotalExpenses'] == 0).astype(int)
df.loc[df['TotalExpenses'] > 0, 'CryoSleep'] = df.loc[df['TotalExpenses'] > 0, 'CryoSleep'].fillna(False)

group_hp = df.groupby('GroupId')['HomePlanet'].transform(lambda s: s.ffill().bfill())
df['HomePlanet'] = df['HomePlanet'].fillna(group_hp)
last_hp = df.groupby('LastName')['HomePlanet'].transform(lambda s: s.ffill().bfill())
df['HomePlanet'] = df['HomePlanet'].fillna(last_hp)
df.loc[(df['HomePlanet'].isna()) & (df['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
df.loc[(df['HomePlanet'].isna()) & (df['Deck'] == 'G'), 'HomePlanet'] = 'Earth'
df['HomePlanet'] = df['HomePlanet'].fillna('Earth')

group_side = df.groupby('GroupId')['Side'].transform(lambda s: s.ffill().bfill())
df['Side'] = df['Side'].fillna(group_side).fillna('Missing')
group_deck = df.groupby('GroupId')['Deck'].transform(lambda s: s.ffill().bfill())
df['Deck'] = df['Deck'].fillna(group_deck).fillna('Missing')
group_cabnum = df.groupby('GroupId')['CabinNum'].transform(lambda s: s.ffill().bfill())
df['CabinNum'] = df['CabinNum'].fillna(group_cabnum).fillna(-1)
group_dest = df.groupby('GroupId')['Destination'].transform(lambda s: s.ffill().bfill())
df['Destination'] = df['Destination'].fillna(group_dest).fillna('TRAPPIST-1e')

df.loc[df['HomePlanet'] == 'Earth', 'VIP'] = df.loc[df['HomePlanet'] == 'Earth', 'VIP'].fillna(False)
df.loc[df['CryoSleep'] == True, 'VIP'] = df.loc[df['CryoSleep'] == True, 'VIP'].fillna(False)
df['VIP'] = df['VIP'].fillna(False).astype(int)

group_cryo = df.groupby('GroupId')['CryoSleep'].transform(lambda s: s.ffill().bfill())
df['CryoSleep'] = df['CryoSleep'].fillna(group_cryo).fillna(False).astype(int)

df['Age'] = df['Age'].fillna(df.groupby(['HomePlanet', 'Deck'])['Age'].transform('median')).fillna(df['Age'].median())

df['IsChild'] = (df['Age'] < 13).astype(int)
df['IsTeen'] = ((df['Age'] >= 13) & (df['Age'] < 18)).astype(int)
df['IsSenior'] = (df['Age'] > 55).astype(int)

df['NegativeAmenities'] = df['RoomService'] + df['Spa'] + df['VRDeck']
df['PositiveAmenities'] = df['FoodCourt'] + df['ShoppingMall']
df['NegativeAmenities_log'] = np.log1p(df['NegativeAmenities'])
df['PositiveAmenities_log'] = np.log1p(df['PositiveAmenities'])
df['NegativeAmenityRatio'] = df['NegativeAmenities'] / (df['TotalExpenses'] + 1.0)
df['PositiveAmenityRatio'] = df['PositiveAmenities'] / (df['TotalExpenses'] + 1.0)
df['ZeroNegativeAmenities'] = (df['NegativeAmenities'] == 0).astype(int)
df['ZeroPositiveAmenities'] = (df['PositiveAmenities'] == 0).astype(int)

df['IsCryo_HighDeck'] = ((df['CryoSleep'] == 1) & (df['Deck'].isin(['A', 'B', 'C', 'D', 'F']))).astype(int)
df['IsSpender_ZeroNeg'] = ((df['TotalExpenses'] > 0) & (df['NegativeAmenities'] == 0)).astype(int)

def get_type(row):
    if row['Age'] < 13: return 0
    elif row['CryoSleep'] == 1: return 1
    elif row['TotalExpenses'] > 0: return 2
    else: return 3
df['PassengerType'] = df.apply(get_type, axis=1)

df['AmenitiesCount'] = (df[expenses] > 0).sum(axis=1)

for col in expenses + ['TotalExpenses']:
    df[f'{col}_log'] = np.log1p(df[col])
    df[f'{col}_pct'] = df[col] / (df['TotalExpenses'] + 1.0)
    df[f'Has_{col}'] = (df[col] > 0).astype(int)

df['SpendPerAge'] = df['TotalExpenses'] / (df['Age'] + 1.0)

df['GroupTotalExpense'] = df.groupby('GroupId')['TotalExpenses'].transform('sum')
df['GroupMeanExpense'] = df.groupby('GroupId')['TotalExpenses'].transform('mean')
df['GroupExpenseRatio'] = df['TotalExpenses'] / (df['GroupTotalExpense'] + 1.0)
df['GroupCryoCount'] = df.groupby('GroupId')['CryoSleep'].transform('sum')
df['GroupCryoRate'] = df.groupby('GroupId')['CryoSleep'].transform('mean')

max_deck_cabin = df.groupby('Deck')['CabinNum'].transform('max')
df['CabinPosInDeck'] = (df['CabinNum'] / (max_deck_cabin + 1.0)).fillna(0.0)
df['CabinRegion100'] = (df['CabinNum'] // 100).astype(int)
df['CabinRegion300'] = (df['CabinNum'] // 300).astype(int)

df['DeckSide'] = df['Deck'] + '_' + df['Side']
df['DeckPlanet'] = df['Deck'] + '_' + df['HomePlanet']
df['SidePlanet'] = df['Side'] + '_' + df['HomePlanet']
df['CryoDeck'] = df['CryoSleep'].astype(str) + '_' + df['Deck']
df['CryoSide'] = df['CryoSleep'].astype(str) + '_' + df['Side']
df['CryoDeckSide'] = df['CryoSleep'].astype(str) + '_' + df['Deck'] + '_' + df['Side']

cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'DeckSide', 'DeckPlanet', 'SidePlanet', 'CryoDeck', 'CryoSide', 'CryoDeckSide']
for c in cat_cols:
    df[c] = df[c].astype('category').cat.codes

features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'IsChild', 'IsTeen', 'IsSenior', 'PassengerType',
    'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log',
    'TotalExpenses_log', 'NoExpenses', 'AmenitiesCount', 'SpendPerAge',
    'NegativeAmenities_log', 'PositiveAmenities_log', 'NegativeAmenityRatio', 'PositiveAmenityRatio',
    'ZeroNegativeAmenities', 'ZeroPositiveAmenities', 'IsCryo_HighDeck', 'IsSpender_ZeroNeg',
    'Has_RoomService', 'Has_FoodCourt', 'Has_ShoppingMall', 'Has_Spa', 'Has_VRDeck',
    'RoomService_pct', 'FoodCourt_pct', 'ShoppingMall_pct', 'Spa_pct', 'VRDeck_pct',
    'GroupSize', 'IsAlone', 'FamilySize', 'GroupId_int',
    'GroupTotalExpense', 'GroupMeanExpense', 'GroupExpenseRatio',
    'GroupCryoCount', 'GroupCryoRate',
    'Deck', 'CabinNum', 'Side', 'CabinRegion100', 'CabinRegion300', 'DeckSide', 'CabinPosInDeck',
    'DeckPlanet', 'SidePlanet', 'CryoDeck', 'CryoSide', 'CryoDeckSide'
]

X_train = df.iloc[:n_train][features].copy().replace([np.inf, -np.inf], 0.0).fillna(0.0)
y_train = df.iloc[:n_train]['Transported'].astype(int).copy()
X_test = df.iloc[n_train:][features].copy().replace([np.inf, -np.inf], 0.0).fillna(0.0)
test_ids = df.iloc[n_train:]['PassengerId'].values

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
test_preds_cb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_lgb = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    
    cb = CatBoostClassifier(iterations=750, learning_rate=0.03, depth=6, random_seed=42+fold, verbose=0)
    cb.fit(X_tr, y_tr)
    test_preds_cb += cb.predict_proba(X_test)[:, 1] / 10
    
    xgb = XGBClassifier(n_estimators=350, learning_rate=0.025, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42+fold)
    xgb.fit(X_tr, y_tr)
    test_preds_xgb += xgb.predict_proba(X_test)[:, 1] / 10
    
    lgb = LGBMClassifier(n_estimators=400, learning_rate=0.025, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42+fold, verbose=-1)
    lgb.fit(X_tr, y_tr)
    test_preds_lgb += lgb.predict_proba(X_test)[:, 1] / 10

# Test blend with 0.50 CB + 0.30 XGB + 0.20 LGB
blend_prob = 0.50 * test_preds_cb + 0.30 * test_preds_xgb + 0.20 * test_preds_lgb
preds_bool = (blend_prob > 0.485).astype(bool)

out_sub = r"c:\Users\Javier\Desktop\kaggle\02_spaceship_titanic\submissions\submission_57f_optimal_th485.csv"
sub_df = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
sub_df.to_csv(out_sub, index=False)
print("Saved to", out_sub)
print("True counts:", sub_df['Transported'].value_counts().to_dict())
