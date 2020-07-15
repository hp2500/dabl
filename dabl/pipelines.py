from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

enable_hist_gradient_boosting


def get_fast_classifiers(n_classes):
    """Get a list of very fast classifiers.

    Parameters
    ----------
    n_classes : int
        Number of classes in the dataset. Used to decide on the complexity
        of some of the classifiers.


    Returns
    -------
    fast_classifiers : list of sklearn estimators
        List of classification models that can be fitted and evaluated very
        quickly.
    """
    return [
        # These are sorted by approximate speed
        DummyClassifier(strategy="prior"),
        GaussianNB(),
        make_pipeline(MinMaxScaler(), MultinomialNB()),
        DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
        DecisionTreeClassifier(max_depth=max(5, n_classes),
                               class_weight="balanced"),
        DecisionTreeClassifier(class_weight="balanced",
                               min_impurity_decrease=.01),
        LogisticRegression(C=.1,
                           solver='lbfgs',
                           multi_class='auto',
                           class_weight='balanced',
                           max_iter=1000),
        # FIXME Add warm starting here?
        LogisticRegression(C=1,
                           solver='lbfgs',
                           multi_class='auto',
                           class_weight='balanced',
                           max_iter=1000)
    ]


def get_fast_regressors():
    """Get a list of very fast regressors.

    Returns
    -------
    fast_regressors : list of sklearn estimators
        List of regression models that can be fitted and evaluated very
        quickly.
    """
    return [
        DummyRegressor(),
        DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=5),
        Ridge(alpha=10),
        Lasso(alpha=10)
    ]


def get_any_classifiers(portfolio='baseline'):
    """Return a portfolio of classifiers.

    Returns
    -------
    classifiers : list of sklearn estimators
        List of classification models.
    """
    baseline = [
        LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=.1, solver='lbfgs', multi_class='multinomial'),
        RandomForestClassifier(max_features=None, n_estimators=100),
        RandomForestClassifier(max_features='sqrt', n_estimators=100),
        RandomForestClassifier(max_features='log2', n_estimators=100),
        SVC(C=1, gamma=0.03, kernel='rbf'),
        SVC(C=1, gamma='scale', kernel='rbf'),
        HistGradientBoostingClassifier()
    ]

    mixed = [
        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       max_bins=256,
                                       max_iter=200,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=50),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.32778941525789984,
                               min_samples_split=5,
                               n_estimators=300),
        SVC(C=52.368035023140784, gamma=0.008051730038808798),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       max_bins=128,
                                       max_depth=8,
                                       max_iter=350,
                                       max_leaf_nodes=16,
                                       min_samples_leaf=4),
        SVC(C=2.5918689981661567,
            coef0=0.3186996400686849,
            gamma=0.0016271844595562733),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.07198156955074353,
                               min_samples_split=6,
                               n_estimators=500),
        HistGradientBoostingClassifier(l2_regularization=0.0001,
                                       learning_rate=0.01,
                                       max_bins=16,
                                       max_depth=9,
                                       max_iter=500,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=33),
        SVC(C=1622.763547526942,
            coef0=0.2316204140676945,
            gamma=0.042609827284645976),
        SVC(C=1.799125831143992,
            coef0=0.7926565732345652,
            gamma=0.01858955180141993,
            kernel='poly'),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.27977761318292416,
                               n_estimators=500),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       max_bins=64,
                                       max_depth=18,
                                       max_iter=350,
                                       max_leaf_nodes=32,
                                       min_samples_leaf=7),
        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       max_bins=256,
                                       max_depth=15,
                                       max_iter=350,
                                       max_leaf_nodes=8),
        HistGradientBoostingClassifier(l2_regularization=1e-08,
                                       max_bins=256,
                                       max_depth=13,
                                       max_iter=450,
                                       max_leaf_nodes=64,
                                       min_samples_leaf=14),
        RandomForestClassifier(criterion='entropy',
                               max_features=5.545027374453948e-05,
                               min_samples_split=11,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.5778510197529033,
                               min_samples_split=6,
                               n_estimators=300),
        SVC(C=3539.053405327911, gamma=0.2574787996118815),
        SVC(C=0.19056746772632044,
            coef0=0.3565163593285343,
            degree=5,
            gamma=0.015503770572916192,
            kernel='sigmoid'),
        SVC(C=100000.0, gamma=0.00241115807647),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       max_bins=64,
                                       max_depth=12,
                                       max_iter=200,
                                       max_leaf_nodes=64),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       max_bins=32,
                                       max_depth=18,
                                       max_iter=450,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=5),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       max_bins=8,
                                       max_depth=20,
                                       max_iter=150,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=13),
        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       max_bins=256,
                                       max_depth=19,
                                       max_iter=300,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=11),
        SVC(C=0.12779439580461893,
            coef0=-0.007860547843195675,
            degree=2,
            gamma=0.011489094638370643,
            kernel='sigmoid'),
        RandomForestClassifier(max_features=0.7583040407073237,
                               min_samples_split=20,
                               n_estimators=100),
        SVC(C=5.3824120488001554, gamma=0.06797377929157812),
        HistGradientBoostingClassifier(l2_regularization=1e-10,
                                       max_bins=32,
                                       max_depth=9,
                                       max_iter=350,
                                       max_leaf_nodes=16,
                                       min_samples_leaf=18),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.05298984247042704,
                               min_samples_leaf=2,
                               min_samples_split=7,
                               n_estimators=500),
        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       max_bins=256,
                                       max_depth=4,
                                       max_iter=350,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=11),
        HistGradientBoostingClassifier(l2_regularization=1e-05,
                                       max_bins=256,
                                       max_depth=16,
                                       max_iter=400,
                                       max_leaf_nodes=64,
                                       min_samples_leaf=10),
        HistGradientBoostingClassifier(l2_regularization=1e-09,
                                       max_bins=8,
                                       max_depth=7,
                                       max_iter=150,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=27),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.3391406102951785,
                               min_samples_split=19,
                               n_estimators=100),
        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       max_bins=32,
                                       max_depth=12,
                                       max_iter=200,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=3)
    ]

    hgb = [
        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       max_bins=256,
                                       max_iter=200,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=50),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       max_bins=64,
                                       max_depth=5,
                                       max_leaf_nodes=4),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       max_bins=64,
                                       max_depth=18,
                                       max_iter=350,
                                       max_leaf_nodes=32,
                                       min_samples_leaf=7),
        HistGradientBoostingClassifier(l2_regularization=1e-07,
                                       max_bins=16,
                                       max_depth=19,
                                       max_iter=500,
                                       max_leaf_nodes=8,
                                       min_samples_leaf=27),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       max_bins=256,
                                       max_depth=16,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=8),
        HistGradientBoostingClassifier(l2_regularization=1e-07,
                                       max_bins=256,
                                       max_depth=16,
                                       max_iter=350,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=13),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       learning_rate=0.01,
                                       max_bins=8,
                                       max_depth=19,
                                       max_iter=350,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=1e-07,
                                       max_bins=16,
                                       max_depth=3,
                                       max_iter=350,
                                       max_leaf_nodes=16,
                                       min_samples_leaf=14),
        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       learning_rate=0.01,
                                       max_bins=64,
                                       max_depth=9,
                                       max_iter=450,
                                       max_leaf_nodes=32,
                                       min_samples_leaf=12),
        HistGradientBoostingClassifier(l2_regularization=0.0001,
                                       max_bins=128,
                                       max_depth=20,
                                       max_iter=500,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       max_bins=16,
                                       max_depth=2,
                                       max_iter=300,
                                       max_leaf_nodes=16,
                                       min_samples_leaf=8),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       max_bins=256,
                                       max_depth=20,
                                       max_iter=400,
                                       max_leaf_nodes=64,
                                       min_samples_leaf=5),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       learning_rate=1.0,
                                       max_bins=32,
                                       max_depth=2,
                                       max_iter=150,
                                       max_leaf_nodes=32),
        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       max_bins=256,
                                       max_depth=8,
                                       max_iter=500,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=9),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       max_bins=256,
                                       max_depth=3,
                                       max_iter=400,
                                       max_leaf_nodes=16,
                                       min_samples_leaf=11),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=1.0,
                                       max_bins=16,
                                       max_depth=2,
                                       max_iter=200,
                                       max_leaf_nodes=64,
                                       min_samples_leaf=2),
        HistGradientBoostingClassifier(l2_regularization=1e-08,
                                       max_bins=8,
                                       max_depth=6,
                                       max_iter=500,
                                       max_leaf_nodes=32,
                                       min_samples_leaf=15),
        HistGradientBoostingClassifier(l2_regularization=1e-05,
                                       max_bins=16,
                                       max_iter=400,
                                       max_leaf_nodes=128,
                                       min_samples_leaf=48),
        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       max_bins=64,
                                       max_depth=14,
                                       max_iter=200,
                                       max_leaf_nodes=16,
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       max_bins=16,
                                       max_depth=4,
                                       max_iter=450,
                                       max_leaf_nodes=64,
                                       min_samples_leaf=19),
        HistGradientBoostingClassifier(l2_regularization=0.0001,
                                       learning_rate=0.01,
                                       max_bins=64,
                                       max_depth=19,
                                       max_iter=200,
                                       max_leaf_nodes=32,
                                       min_samples_leaf=8),
        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       max_bins=16,
                                       max_depth=17,
                                       max_iter=50,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=23),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       max_bins=128,
                                       max_depth=16,
                                       max_leaf_nodes=8,
                                       min_samples_leaf=5),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       max_bins=8,
                                       max_depth=20,
                                       max_iter=150,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=13),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       max_bins=256,
                                       max_depth=15,
                                       max_iter=200,
                                       max_leaf_nodes=64),
        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       max_bins=256,
                                       max_depth=18,
                                       max_iter=450,
                                       max_leaf_nodes=64,
                                       min_samples_leaf=9),
        HistGradientBoostingClassifier(l2_regularization=100.0,
                                       max_bins=64,
                                       max_depth=7,
                                       max_iter=350,
                                       max_leaf_nodes=8,
                                       min_samples_leaf=5),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       max_bins=64,
                                       max_depth=5,
                                       max_iter=450,
                                       max_leaf_nodes=32,
                                       min_samples_leaf=15),
        HistGradientBoostingClassifier(l2_regularization=1e-05,
                                       max_bins=256,
                                       max_depth=16,
                                       max_iter=400,
                                       max_leaf_nodes=64,
                                       min_samples_leaf=10),
        HistGradientBoostingClassifier(l2_regularization=1e-10,
                                       max_bins=64,
                                       max_depth=2,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       learning_rate=0.01,
                                       max_bins=4,
                                       max_depth=18,
                                       max_iter=200,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=39),
        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       max_bins=128,
                                       max_depth=12,
                                       max_iter=300,
                                       max_leaf_nodes=4,
                                       min_samples_leaf=3)
    ]

    svc = [
        SVC(C=52.368035023140784, gamma=0.008051730038808798),
        SVC(C=149.07622270551335, gamma=0.05610768111553853),
        SVC(C=1.68536554317688,
            coef0=0.2168646578884883,
            gamma=0.0008080068502590277),
        SVC(C=81.8664880584341,
            coef0=0.1339044447397313,
            degree=4,
            gamma=0.6339071538529285),
        SVC(C=1.799125831143992,
            coef0=0.7926565732345652,
            gamma=0.01858955180141993,
            kernel='poly'),
        SVC(C=55762.3529353618,
            coef0=-0.8056114085510306,
            gamma=3.187772482265977e-05,
            kernel='sigmoid'),
        SVC(C=34.18479740302528,
            coef0=0.465809282171058,
            gamma=0.025017141595224057),
        SVC(C=4.104647380564808,
            coef0=-0.724712336449596,
            degree=5,
            gamma=0.2926981232494074),
        SVC(C=1.0357760788047117,
            coef0=0.420858393631554,
            degree=4,
            gamma=6.420989750880023),
        SVC(C=0.42681252219264904,
            coef0=0.23495235580748663,
            degree=4,
            gamma=0.0419665675168468),
        SVC(C=128.0, gamma=0.03125),
        SVC(C=1.0827206202000408, gamma=0.012405921979379314),
        SVC(C=49.496117538847585,
            coef0=-0.36888378135766664,
            degree=5,
            gamma=0.0013908089928613187),
        SVC(C=3.6080862947181074,
            coef0=-0.5499049407262779,
            degree=1,
            gamma=0.03110141304760934),
        SVC(C=0.14135351008197172,
            coef0=-0.6091263079805762,
            degree=5,
            gamma=3.1086101104977364e-05,
            kernel='poly'),
        SVC(C=53212.0921773362,
            coef0=-0.20185133000171707,
            degree=4,
            gamma=7.496099998815888e-05),
        SVC(C=0.12779439580461893,
            coef0=-0.007860547843195675,
            degree=2,
            gamma=0.011489094638370643,
            kernel='sigmoid'),
        SVC(C=1622.763547526942,
            coef0=0.2316204140676945,
            gamma=0.042609827284645976),
        SVC(C=4.184538758259627,
            coef0=-0.6638392568697213,
            degree=4,
            gamma=0.01078805639317049),
        SVC(C=0.5294291732233543, gamma=0.17699127262468486),
        SVC(C=100000.0, gamma=0.00241115807647),
        SVC(C=33.252648089739836, gamma=2.1212339071044592),
        SVC(C=9.62953169042274, gamma=0.037086661651597304),
        SVC(C=513.6546159384284,
            coef0=-0.16364828944715204,
            degree=5,
            gamma=0.3170962383864124),
        SVC(C=8.537098039116069, gamma=0.014430579841442782),
        SVC(C=32.184662024224764,
            coef0=0.8359658612498964,
            degree=2,
            gamma=0.11650886331310951,
            kernel='poly'),
        SVC(C=6.5031187555491305,
            coef0=0.3569381453006406,
            degree=4,
            gamma=0.15364377981481867),
        SVC(C=12825.233283804411,
            coef0=-0.5596962381502244,
            degree=1,
            gamma=0.007137693124484331),
        SVC(C=0.9963167675010125,
            coef0=0.8350264578298481,
            degree=2,
            gamma=0.0015426280688508745),
        SVC(C=4.523994003904736,
            coef0=0.20331467465699182,
            degree=5,
            gamma=0.033987763661532146),
        SVC(C=3.081620245614723,
            coef0=-0.053322514098660845,
            degree=5,
            gamma=0.31447231452702795,
            kernel='poly'),
        SVC(C=15.406711505563193,
            coef0=-0.8097233063524558,
            degree=2,
            gamma=0.007206235544735408)
    ]

    rf = [
        RandomForestClassifier(criterion='entropy',
                               max_features=0.13891783454814322,
                               min_samples_leaf=2,
                               min_samples_split=9,
                               n_estimators=300),
        RandomForestClassifier(max_features=0.050069426976632525,
                               min_samples_split=3,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.38146409165028106,
                               min_samples_split=12,
                               n_estimators=300),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.20254434685778855,
                               min_samples_split=3,
                               n_estimators=500),
        RandomForestClassifier(max_features=0.09371003097672581,
                               min_samples_split=5,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.4203242802359065,
                               min_samples_split=9,
                               n_estimators=100),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.8014260101515722,
                               min_samples_leaf=6,
                               min_samples_split=18,
                               n_estimators=300),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.17540145080676417,
                               n_estimators=300),
        RandomForestClassifier(max_features=0.12496709200924094,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.02701567860915255,
                               min_samples_split=7,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_depth=9,
                               max_features=0.25,
                               min_samples_split=3,
                               n_estimators=500),
        RandomForestClassifier(max_features=0.08362628567277053,
                               min_samples_leaf=2,
                               min_samples_split=4,
                               n_estimators=300),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.45369329148302284,
                               min_samples_split=3,
                               n_estimators=100),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.014380612412532856,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.9028655839413416,
                               min_samples_leaf=2,
                               min_samples_split=3,
                               n_estimators=500),
        RandomForestClassifier(max_features=0.477758211021571,
                               n_estimators=500),
        RandomForestClassifier(max_features=0.8965933352549531,
                               min_samples_leaf=18,
                               min_samples_split=4,
                               n_estimators=100),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.32405539446276466,
                               min_samples_split=6,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.2628558322023252,
                               min_samples_leaf=3,
                               min_samples_split=19,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.05298984247042704,
                               min_samples_leaf=2,
                               min_samples_split=7,
                               n_estimators=500),
        RandomForestClassifier(max_features=0.05946371474549883,
                               n_estimators=300),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.13728248146122612,
                               n_estimators=500),
        RandomForestClassifier(max_features=0.4127164296489969,
                               min_samples_split=6,
                               n_estimators=100),
        RandomForestClassifier(criterion='entropy',
                               max_depth=12,
                               max_features=0.1,
                               min_samples_split=4,
                               n_estimators=512),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.6077306020179516,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.07198156955074353,
                               min_samples_split=6,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.26491494231808,
                               min_samples_split=3,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.13878410049133105,
                               min_samples_leaf=2,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.4385263141334529,
                               min_samples_leaf=4,
                               min_samples_split=7,
                               n_estimators=300),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.11082304637318625,
                               min_samples_leaf=19,
                               min_samples_split=18,
                               n_estimators=100),
        RandomForestClassifier(max_features=0.003021649218545863,
                               min_samples_split=10,
                               n_estimators=500),
        RandomForestClassifier(criterion='entropy',
                               max_features=0.15300932068501327,
                               min_samples_split=3,
                               n_estimators=500)
    ]

    portfolios = {
        'baseline': baseline,
        'mixed': mixed,
        'svc': svc,
        'hgb': hgb,
        'rf': rf
    }

    return (portfolios[portfolio])
