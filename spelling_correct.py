import enchant
from local_paths import *

__author__ = 'mudit'

spell_checker = enchant.DictWithPWL('en_US', INPUT_PATH + 'attributes.csv', INPUT_PATH + 'product_description.csv')
