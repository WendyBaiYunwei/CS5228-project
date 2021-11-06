from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# l = pd.read_csv('./small_test.csv').columns.tolist()
# print(l)
# exit()
basic = ['dereg_value', 'mileage', 'omv', 'age']
types = ['bus/mini bus', 'hatchback', 'luxury sedan', 'mid-sized sedan', 'mpv', 'sports car', 'stationwagon', 'suv', 'truck', 'van', 'auto', 'manual']
makes = ['alfa romeo', 'alpine', 'aston martin', 'audi', 'austin', 'bentley', 'bmw', 'byd', 'cadillac', 'chery', 'chevrolet', 'chrysler', 'citroen', 'cupra', 'daf', 'daihatsu', 'daimler', 'datsun', 'dodge', 'dongfeng', 'ferrari', 'fiat', 'ford', 'foton', 'golden dragon', 'hafei', 'higer', \
'hino', 'honda', 'hummer', 'hyundai', 'infiniti', 'international', 'isuzu', 'iveco', 'jaguar', 'jeep', 'joylong', 'kia', 'lamborghini', \
'land rover', 'lexus', 'lotus', 'man', 'maserati', 'maxus', 'maybach', 'mazda', 'mclaren', 'mercedes', 'mercedes-benz', 'mg', 'mini', 'mitsubishi', 'mitsuoka', 'morgan', 'morris', 'nissan', 'opel', 'perodua', 'peugeot', 'porsche', 'proton', 'renault', 'riley', 'rolls', 'rolls-royce', 'rover', 'ruf', 'saab', 'scania', 'seat', 'sinotruk', 'skoda', 'smart', 'ssangyong', 'subaru', 'suzuki', 'tesla', 'toyota', 'triumph', 'ud', 'volkswagen', 'volvo', 'yutong']
models = ['107', '116d', '116i', '118i', '120i', '125i', '12c', '1300', '135i', '14.28', '159', '190e', '1m', '2', '200', '2008', '200sx', '207', '207cc', '208', '216', '216d', '216i', '218d', '218i', '219', '220', '220i', '225xe', '228i', '230', '230i', '250', '280s', '280zx', '3', '3000gt', '3008', '300c', '300ce', '300gd', '300sel', '300sl', '308', '315', '316i', '318ci', '318i', '320ce', '320d', '320i', '323i', '325ci', '325i', '328i', '330ci', '330e', '330i', '335i', '340i', '348', '360', '3800s', '420i', '428i', '430i', '435i', '440i', '450slc', '458', '488', '500', '5008', '508', '520d', '520i', '523i', '525i', '528i', '530e', '530i', '535i', '540i', '570s', '575m', '599', '600lt', '612', '62', '630ci', '630i', '640i', '650i', '650s', '718', '720s', '730d', \
'730i', '730li', '740e', '740i', '740li', '750i', '750li', '840i', '850i', '86', '900s', '911', '924', '940', '944', '9��3��', '9��5��', 'a1', 'a110', 'a156', 'a170', 'a180', 'a200', 'a250', 'a3', 'a35', 'a4', 'a45', 'a5', 'a6', 'a7', 'a8', 'accent', 'accord', 'activehybrid', 'actros', 'actyon', 'adam', 'aero', 'airtrek', 'airwave', 'alhambra', 'allion', 'almera', 'alphard', 'alpina', 'amg', 'apv', 'aqua', 'arona', 'arteon', 'astra', 'asx', 'ateca', 'atlas', 'attrage', 'aumark', 'avante', 'avanza', 'aventador', 'aveo', 'axela', 'axia', 'axor', 'b160', 'b170', 'b180', 'b200', 'beetle', 'bentayga', 'benz', 'berlingo', 'bj1041', 'bj6800', 'boxster', 'bravo', 'brz', 'c-hr', 'c160', 'c180', 'c200', 'c250', 'c3', 'c300', 'c350', 'c4', 'c5', 'cabstar', 'caddy', 'california', 'camry', 'capri', 'captur', 'carens', 'cascada', 'cayenne', 'cayman', 'cc', 'cefiro', 'celica', 'century', 'cerato', 'cherokee', 'ciaz', 'citan', 'city', 'civic', 'cl550', 'cla180', 'cla200', 'cla250', 'clc180k', 'clio', 'clk230', 'clk280', 'cls250', 'cls350', 'cls400', 'cls500', 'coaster', 'colt', 'combo', \
'compass', 'condor', 'continental', 'cooper', 'copen', 'corolla', 'corona', 'corsa', 'coupe', 'cr-v', 'cr-z', 'crossland', 'crossroad', \
'crown', 'cruze', 'ct', 'cts', 'cullinan', 'cx-3', 'cx-7', 'cyh52s', 'cyh52t', 'cyz52k', 'cyz52l', 'cyz52r', 'dawn', 'db11', 'db9', 'dbs', 'dbx', 'defender', 'discovery', 'dispatch', 'doblo', 'double-cab', 'ds3', 'ds4', 'ds5', 'dualis', 'dyna', 'e', 'e-pace', 'e-tron', 'e-type', 'e180', 'e200', 'e220', 'e230', 'e250', 'e280', 'e300', 'e350', 'e6', 'eclipse', 'elantra', 'elgrand', 'elise', 'eos', 'epica', \
'eqv', 'es', 'esprit', 'esquire', 'estima', 'europa', 'every', 'evolution', 'exiga', 'exora', 'expert', 'f-pace', 'f-type', 'f12berlinetta', 'f355', 'f430', 'f512m', 'f612s', 'f8', 'fa', 'fairlady', 'fd8jlka', 'fe83', 'ff', 'fiesta', 'fiorino', 'fit', 'fj', 'fluence', 'flying', 'fm370', 'focus', 'forester', 'fortuner', 'fortwo', 'freed', 'freelander', 'frr34s', 'frr90', 'fs1elkm', 'fs1etka', 'fsr34s', 'fto', 'ftr34p', 'fuso', 'fvr34', 'fvr90', 'fxz77m', 'fy1etkm', 'fy1euka', 'g10', 'g350', 'g420', 'galaxy', 'gallardo', 'gen', 'genesis', 'getz', 'gh8jrma', 'ghibli', 'ghost', 'giulia', 'giulietta', 'gl400', 'gla180', 'gla200', 'gla250', 'gladiator', 'glb200', 'glc200', 'glc250', 'glc300', 'gle400', 'gle450', 'golf', 'gr', 'grace', 'gran', 'grancabrio', 'grand', 'grandis', 'grandland', 'gransport', 'granturismo', 'gs', 'gt', 'gtc4lusso', 'gto', 'gtr', 'h3', 'harrier', 'healey', 'hiace', 'hilux', 'himiko', 'hkl6540', 'hr-v', 'hs', 'huracan', \
'i-pace', 'i3', 'i30', 'i45', 'i8', 'ibiza', 'impala', 'impreza', 'insight', 'insignia', 'integra', 'ioniq', 'iq', 'is', 'isis', 'ist', \
'jade', 'jazz', 'jetta', 'jimny', 'journey', 'juke', 'k2500', 'kadjar', 'kamiq', 'kangoo', 'karmann', 'karoq', 'keb4x2', 'kelisa', 'kenari', 'kestrel', 'kib4x2', 'kicks', 'klq6916', 'kodiaq', 'kombi', 'kona', 'kub4x2', 'kuga', 'lafesta', 'lancer', 'land', 'latio', 'lc500', 'leaf', 'legacy', 'leon', 'levante', 'levorg', 'liteace', 'lj80', 'ls350', 'ls460', 'ls500', 'lt133p', 'lt134p', 'lt434p', 'lx570', 'm135i', 'm140i', 'm2', 'm3', 'm4', 'm5', 'm6', 'm8', 'macan', 'magentis', 'malibu', 'march', 'mark', 'materia', 'maybach.1', 'megane', 'meriva', 'mg.1', 'midget', 'mini.1', 'minor', 'minz', 'mito', 'mk', 'ml250', 'ml300', 'ml350', 'ml400', 'mobilio', 'mokka', 'mondeo', 'mr2', 'mrs', 'mulsanne', 'murano', 'mustang', 'mx-5', 'myvi', 'n-box', 'n-van', 'navara', 'nemo', 'nhr85a', 'nhr87a', 'niro', 'njr85a', 'njr88a', 'nmr85', 'nnr85', 'noah', 'note', 'nouera', 'npr71l', 'npr75', 'npr85', 'nqr75u', 'nv100', 'nv150', 'nv200', 'nv250', 'nv350', 'nx', 'octavia', 'odyssey', 'one', 'optima', 'optra', 'orlando', 'outback', 'outlander', 'p124', 'p1800', 'p360', 'panamera', 'partner', \
'passat', 'patriot', 'persona', 'phantom', 'picanto', 'picnic', 'pkc37', 'polo', 'portofino', 'prado', 'premio', 'presage', 'previa', 'prin', 'prius', 'proace', 'probox', 'pulsar', 'q2', 'q3', 'q30', 'q5', 'q50', 'q60', 'q7', 'q8', 'qashqai', 'quattroporte', 'qx30', 'qx50', 'qx80', 'r280l', 'r300l', 'r350l', 'r8', 'ractis', 'raize', 'range', 'ranger', 'rapid', 'rapide', 'rav4', 'rc', 'rcz', 'regiusace', 'renegade', 'rio', 'rockstar', 'rosa', 'royce', 'rs', 'rt35s', 'rush', 'rx', 'rx-8', 'rx7', 's-max', 's-type', 's2000', 's280', 's3', 's300l', 's320l', 's350', 's4', 's40', 's400l', 's450', 's5', 's500', 's500l', 's560', 's6', 's60', 's600l', 's80', 's90', 'saga', 'santa', 'scenic', 'scirocco', 'scout', 'seltos', 'serena', 'series', 'sf90', 'sh1eeka', 'sh1eelg', 'sharan', 'shuttle', 'sienta', 'silver', 'single-cab', 'sitrak', 'sj41', 'skyline', 'sl280', 'sl320', 'sl350', 'sl400', 'sl500', 'slk200', 'slk250', 'slk280', 'slk350', 'solio', 'sonata', 'sonic', 'sorento', 'soul', 'sovereign', 'space', 'spark', 'spider', 'spitfire', 'sportage', 'sportsvan', 'sprinter', 'sq5', 'star', 'starex', 'starion', 'starlet', 'stavic', 'stelvio', 'stepwagon', 'stinger', 'stonic', 'stream', 'sunny', 'super', 'superb', 'swift', 'sx4', 'sylphy', 't-cross', 't11', 't3', 'tarraco', 'taycan', 'teana', 'terios', 'testarossa', 'tfr54h', 'tfr86', 'tfr87', 'tfs87j', \
'tgs', 'tiguan', 'tivoli', 'toledo', 'touareg', 'touran', 'townace', 'toyoace', 'trafic', 'trajet', 'trakker', 'transporter', 'trf87j', \
'triton', 'trucks', 'tt', 'tucson', 'tudor', 'tuscani', 'urus', 'urvan', 'ux', 'v220', 'v250', 'v250d', 'v260l', 'v40', 'v50', 'v60', 'v70', 'v8', 'v80', 'v90', 'vanguard', 'vanquish', 'vantage', 'vellfire', 'veloster', 'venue', 'verna', 'vezel', 'viano', 'vios', 'vitara', 'vito', 'vitz', 'viva', 'vivaro', 'vivio', 'voxy', 'wish', 'wraith', 'wrangler', 'wrx', 'x-trail', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'xc40', 'xc60', 'xc90', 'xe', 'xf', 'xj', 'xk', 'xml6103', 'xml6701', 'xv', 'xzu605r', 'xzu710r', 'yaris', 'yeti', 'z3', 'z4', 'zafira', 'zk6100h', 'zk6938h', 'zoe', 'zs']

l = basic.copy()
l.extend(makes)
l.extend(models)
l.extend(types)
if 'Unnamed: 0' in l:
    l.remove('Unnamed: 0')
if 'Unnamed: 0.1' in l:
    l.remove('Unnamed: 0.1')
X_train = pd.read_csv('small_train.csv')[l]
X_test = pd.read_csv('small_test.csv')[l]
y_train = pd.read_csv('small_train_y.csv')['price']
y_test = pd.read_csv('small_test_y.csv')['price']
# X_train = pd.read_csv('./new_data/small_train.csv')[l]
# X_test = pd.read_csv('./new_data/small_test.csv')[l]
# y_train = pd.read_csv('./new_data/small_train_y.csv')['price']
# y_test = pd.read_csv('./new_data/small_test_y.csv')['price']

model = RandomForestRegressor(n_estimators = 500, random_state=0)
# model = RandomForestRegressor(n_estimators = 300)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
print(mse)

print(model.feature_importances_)
# diff = abs(y_pred - y_test)
# X_test = pd.concat([X_test, y_test], axis=1)
# X_test['predicted_price'] = y_pred
# X_test['difference'] = diff
# X_test=X_test.sort_values(by=['difference'],ascending=False)
# X_test.to_csv('poor_results.csv')

# model = RandomForestRegressor(n_estimators = 700, max_features = 7)
# df = pd.read_csv('numerical_cleaned_train.csv')
# l = df.columns.tolist()
# l.remove('price')
# X = df[l]
# y = df['price']
# model.fit(X, y)
# final_X = pd.read_csv('numerical_cleaned_test.csv')
# y = model.predict(final_X)
# y = pd.DataFrame(y)
# y.to_csv('predictions.csv')