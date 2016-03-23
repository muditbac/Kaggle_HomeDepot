print(__doc__)

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
import pandas as pd
from local_paths import *

data_X = pd.read_csv(OUTPUT_PATH + 'X_features.csv')

train_y = data_X['relevance'].values.reshape(len(data_X))
train_X = data_X.drop(['relevance'], axis=1).values

# Build a forest and compute the feature importances

rfr = RandomForestRegressor(n_estimators = 50, n_jobs = -1, random_state = 2016, verbose = 1)

rfr.fit(train_X, train_y)
importances = rfr.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfr.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

'''
Output

1. feature 9 (0.146532)
2. feature 13 (0.023000)
3. feature 10 (0.018158)
4. feature 23 (0.017810)
5. feature 24 (0.015264)
6. feature 63 (0.014973)
7. feature 14 (0.014944)
8. feature 15 (0.014400)
9. feature 45 (0.014015)
10. feature 35 (0.013730)
11. feature 62 (0.012649)
12. feature 18 (0.012579)
13. feature 12 (0.011545)
14. feature 41 (0.011404)
15. feature 43 (0.010947)
16. feature 2 (0.010430)
17. feature 58 (0.010351)
18. feature 40 (0.010289)
19. feature 44 (0.010250)
20. feature 48 (0.010077)
21. feature 53 (0.009675)
22. feature 29 (0.009572)
23. feature 37 (0.009539)
24. feature 33 (0.009527)
25. feature 56 (0.009493)
26. feature 61 (0.009430)
27. feature 19 (0.009398)
28. feature 38 (0.009393)
29. feature 46 (0.009384)
30. feature 57 (0.009250)
31. feature 20 (0.009109)
32. feature 49 (0.009077)
33. feature 54 (0.009033)
34. feature 1 (0.009019)
35. feature 60 (0.009010)
36. feature 59 (0.008988)
37. feature 0 (0.008968)
38. feature 55 (0.008965)
39. feature 17 (0.008879)
40. feature 36 (0.008872)
41. feature 50 (0.008788)
42. feature 52 (0.008750)
43. feature 51 (0.008698)
44. feature 21 (0.008641)
45. feature 28 (0.008629)
46. feature 42 (0.008603)
47. feature 47 (0.008481)
48. feature 32 (0.008476)
49. feature 30 (0.008314)
50. feature 34 (0.008312)
51. feature 16 (0.008140)
52. feature 27 (0.008113)
53. feature 22 (0.008067)
54. feature 25 (0.008011)
55. feature 31 (0.007944)
56. feature 39 (0.007796)
57. feature 26 (0.007786)
58. feature 64 (0.007422)
59. feature 110 (0.006149)
60. feature 66 (0.006021)
61. feature 111 (0.005912)
62. feature 113 (0.005910)
63. feature 108 (0.005887)
64. feature 112 (0.005870)
65. feature 106 (0.005840)


'''