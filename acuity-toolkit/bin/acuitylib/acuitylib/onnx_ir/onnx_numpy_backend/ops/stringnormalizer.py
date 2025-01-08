from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def StringNormalizer(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    case_change_action = attr.get('case_change_action', None)
    is_case_sensitive = attr.get('is_case_sensitive', 0) == 1
    locale = attr.get('locale', 'en_US')
    if is_case_sensitive:
        stopwords = [sw.decode('utf-8') for sw in attr.get('stopwords', [])]
    else:
        stopwords = [sw.decode('utf-8').lower() for sw in attr.get('stopwords', [])]
    for location in np.ndindex(x.shape):
        v = x[location]
        if stopwords != None:
            cw = v if is_case_sensitive else v.lower()
            if cw not in stopwords:
                if case_change_action == 'LOWER':
                    v = v.lower()
                elif case_change_action == 'UPPER':
                    v = v.upper()
            else:
                v = cw
        else:
            if case_change_action == 'LOWER':
                v = v.lower()
            elif case_change_action == 'UPPER':
                v = v.upper()
        x[location] = v

    if stopwords != None:
        for stpw in stopwords:
            index = np.argwhere(x==stpw)
            x = np.delete(x, index)
    return x
