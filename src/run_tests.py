from BorutaShap import BorutaShap, load_data
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor



def Test_Models(data_type, models):

    X,y = load_data(data_type=data_type)

    for key, value in models.items():

        print('Testing: ' + str(key))
        # no model selected default is Random Forest, if classification is False it is a Regression problem
        Feature_Selector = BorutaShap(
            model=value,
            importance_measure='shap',
            classification=(data_type != 'regression')
        )

        Feature_Selector.fit(X=X, y=y, n_trials=5, random_state=0, train_or_test = 'train')

            
        # Returns Boxplot of features disaplay False or True to see the plots for automation False
        Feature_Selector.plot(X_size=12, figsize=(12,8),
                     y_scale='log', which_features='all', display=False)
        




if __name__ == "__main__":
    
    tree_classifiers = {'tree-classifier':DecisionTreeClassifier(), 'forest-classifier':RandomForestClassifier(),
                        'xgboost-classifier':XGBClassifier(),'lightgbm-classifier':LGBMClassifier(),
                        'catboost-classifier':CatBoostClassifier()}


    tree_regressors = {'tree-regressor':DecisionTreeRegressor(), 'forest-regressor':RandomForestRegressor(),
                       'xgboost-regressor':XGBRegressor(),'lightgbm-regressor':LGBMRegressor(),
                       'catboost-regressor':CatBoostRegressor()}

    
    Test_Models('regression', tree_regressors)
    Test_Models('classification', tree_classifiers)
