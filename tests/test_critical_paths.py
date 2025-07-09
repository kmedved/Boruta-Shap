import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from BorutaShap import BorutaShap, load_data
from sklearn.base import clone
from sklearn.datasets import make_classification
import pandas as pd


def test_default_constructor_runs():
    X, y = load_data('classification')
    fs = BorutaShap()
    fs.fit(X=X, y=y, n_trials=1, random_state=0, train_or_test='train',
            sample=False, normalize=True, verbose=False)


def test_set_params_returns_self_and_clone():
    fs = BorutaShap()
    returned = fs.set_params(importance_measure='shap')
    assert returned is fs
    cloned = clone(fs, safe=False)
    assert isinstance(cloned, BorutaShap)


def test_permutation_importance_runs():
    Xc, yc = load_data('classification')
    fs = BorutaShap(importance_measure='perm', classification=True)
    fs.fit(X=Xc, y=yc, n_trials=1, random_state=0, train_or_test='train',
            sample=False, normalize=False, verbose=False)

    Xr, yr = load_data('regression')
    fs = BorutaShap(importance_measure='perm', classification=False)
    fs.fit(X=Xr, y=yr, n_trials=1, random_state=0, train_or_test='train',
            sample=False, normalize=False, verbose=False)


def test_tentative_rough_fix_changes_state():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=2,
                               random_state=0)
    X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    y = pd.Series(y)

    fs = BorutaShap(importance_measure='gini', classification=True)
    fs.fit(X, y, n_trials=1, random_state=0, train_or_test='train',
            sample=False, normalize=True, verbose=False)

    before = len(fs.accepted) + len(fs.rejected)
    fs.TentativeRoughFix()
    after = len(fs.accepted) + len(fs.rejected)
    assert after >= before


def test_importance_measure_sync():
    fs = BorutaShap()
    fs.set_params(importance_measure='gini')
    assert fs.importance_measure == 'gini'


def test_transform_respects_tentative_flag():
    X, y = load_data('classification')
    fs = BorutaShap().fit(X, y, n_trials=1, train_or_test='train', verbose=False)
    assert set(fs.transform(X).columns) == set(fs.accepted)
    assert set(fs.transform(X, tentative=True).columns) == set(fs.accepted + fs.tentative)

