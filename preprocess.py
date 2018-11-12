import pandas as pd
import numpy as np
import collections
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


ALL_USED_COLUMNS = ["week", "call_time", "list_type", "re_call_date", "address",
                    "kabushiki_code", "establishment", "shihonkin", "employee_num",
                    "kojokazu", "jigyoshokazu", "zenkikessan_uriagedaka",
                    "zenkikessan_riekikin", "tokikessan_uriagedaka",
                    "tokikessan_riekikin", "tokiuriage_shinchoritsu",
                    "zenkiuriage_shinchoritsu", "tokirieki_shinchoritsu",
                    "zenkirieki_shinchoritsu", "birthday", "danjokubun",
                    "eto_meisho", "tosankeireki", "race_area"]


MAIN_ADDR_LIST = ['宮城県', '東京都', '愛知県', '大阪府', '埼玉県', '神奈川県', '千葉県']

MAIN_CHARGER_LIST = ['y-miyajima', 'oosaki', 'w-matsumoto', 'sakazaki', 'shimako', 'm-furukawa', 'a-ichikawa',
                     't-yamaguchi', 's-kinoshita', 'y-ozaki', 'miyake', 'a-go', 'omura', 'hoshina', 'y-endo',
                     'mik-sato', 'soma', 'h-oono', 'arakaki', 's-ootsu', 'saida']

MAIN_INDUSTRY_LIST = ["721", "621", "631", "6821", "722", "6911", "6921", "641", "6812", "6941", "5229",
                      "831", "661", "9299", "5419", "651", "9121", "782", "5019", "6811", "5599", "6099",
                      "795", "5432", "841", "8821", "5319", "833", "5329", "4711"]


class CategoricalEncoder():
    """
    Create a sklearn label encoder object for parsing input data
    """
    def __init__(self, onehot_encode=True, categories=None):
        if categories is None:
            categories = 'auto'
        else:
            try:
                categories = list(categories)
            except:
                raise ValueError("Input categories should be iterable or None")
        if onehot_encode:
            self._encoder = OneHotEncoder(sparse=False, categories=categories)
        else:
            self._encoder = LabelEncoder()
        self._onehot_encode = onehot_encode

    @property
    def category_names(self):
        """
        Return unique category names
        """
        try:
            if self._onehot_encode:
                cate_names = self._encoder.categories_
            else:
                cate_names = self._encoder.classes_
            return cate_names
        except:
            raise RuntimeError("Encoder not fit to data yet! Run 'parse' first")

    @property
    def get_encoder(self):
        return self._encoder

    def parse(self, data):
        data = np.asarray(data)
        res = self._encoder.fit_transform(np.expand_dims(data, 1))
        print("Categories: ", self.category_names)
        return res


class NumericalEncoder():
    """
    Create a numeric encoder to parse numerical data
    """
    def __init__(self, valid_range=None, normalize=True):
        """
        valid_range: tuple of float to indicate low and high percentile threshold for valid value range
        normalize: normalize input value or not
        """
        if valid_range is None:
            self._valid_range = (0, 1.)
        else:
            self._valid_range = valid_range
        self._normalize = normalize
        self._mean = None
        self._std = None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_mean_and_std(self, data):
        """
        Get mean value and std value in given data
        """
        valid_data = data.replace(" ", np.nan)
        valid_data = valid_data.dropna().astype(float)
        low_cut, high_cut = np.quantile(valid_data, (self._valid_range[0], self._valid_range[1]))
        valid_data = valid_data[(valid_data > low_cut) & (valid_data < high_cut)]
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        self._mean = mean
        self._std = std

    def parse(self, data, rep_na="ave"):
        """
        Parse numeric value in give data. If normalize is True, normalize data by mean value and std value
        after removing outliers (top 5% and bottom 5%)
        """
        if isinstance(rep_na, (float, int)):
            alternative = rep_na
        elif rep_na.lower() == "ave":
            if self._mean is None:
                self._get_mean_and_std(data)
            alternative = self._mean
        else:
            raise ValueError("rep_na should a numeric value or 'ave'!")
        if isinstance(data, pd.Series):
            valid_data = data.replace(" ", np.nan)
            valid_data = valid_data.fillna(alternative).astype(float)
        else:
            valid_data = np.asarray(data)
        if self._normalize:
            valid_data -= self._mean
            valid_data /= self._std
            print("mean value: ", self._mean)
            print("mean std: ", self._std)
        return valid_data


class LikelihoodEncoder():
    """
    Use likelihood encoder to convert categorical data
    """
    def __init__(self):
        """
        Args:
            rep_na: method to replace na. If 'ave', use average likelihood as instead; if 'min', use 0 to replace
        """
        self._prob_dict = None

    def _data_occurrence(self, data):
        """
        Count occurrence of each element in df series
        """
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        data = data.fillna("unknown")
        data_dict = dict(collections.Counter(data))
        return data_dict

    def _data_occurrence_prob(self, data):
        data_dict = self._data_occurrence(data)
        # if self._rep_na == "ave":
        #     self._rep = 1 / (len(data_dict) - 1)
        prob_dict = {}
        total_valid_num = sum(data_dict.values())
        for key in data_dict:
            prob_dict[key] = data_dict[key] / total_valid_num
        self._prob_dict = prob_dict

    def parse(self, data):
        if self._prob_dict is None:
            self._data_occurrence_prob(data)
        encoded_data = [self._prob_dict.get(x, 0.) for x in data]
        if isinstance(data, pd.Series):
            encoded_data = pd.Series(encoded_data)
        return encoded_data


def preprocess_numerical_value(data, valid_range=None, rep_na="ave", normalize=True, category_name=None, use_log=False):
    """
    Preprocess numeric input data
    Args:
        data: input pd series data
        valid_range: tuple of float to indicate low and high percentile threshold for valid value range
        rep_na: values for replacing na. Default is 'unknown'.
        normalize: normalize input value or not
        category_name: input data category name
        use_log: use log to convert data, ONLY available when normalize is False
    """
    if valid_range is None:
        valid_range = (0.05, 0.95)
    encoder = NumericalEncoder(valid_range=valid_range, normalize=normalize)
    data_encoded = encoder.parse(data, rep_na=rep_na)
    if use_log and (not normalize):
        data_encoded = np.log(data_encoded + 1)  # +1 to avoid 0
    # covert to ndarray and shape to (-1, 1)
    data_encoded = np.expand_dims(np.asarray(data_encoded), 1)
    return (data_encoded, encoder, category_name)


def preprocess_categorical_value(data,
                                 rep_na=None,
                                 keep_name_list=None,
                                 onehot_encode=True,
                                 categories=None,
                                 other_name_rep=None):
    """
    Preprocess categorical input value
    Args:
        data: input pandas Series data
        rep_na: values for replacing na. Default is 'unknown'.
        keep_name_list: list of name to remain, others will be replaced by 'other'
        onehot_encode: return label in onehot encoding
        categories: category name for parsing onehot encode
        other_name_rep: replacement name for items out of keep name list
    Returns:
        encoded data and encoder object
    """
    if rep_na is None:
        rep_na = "unknown"
    if other_name_rep is None:
        other_name_rep = "others"
    encoder = CategoricalEncoder(onehot_encode=onehot_encode, categories=categories)
    # replace other labels with other
    if not keep_name_list is None:
        data[~ data.isna() & ~ data.isin(keep_name_list)] = other_name_rep
    valid_data = data.fillna(rep_na).astype(str)
    encoded_data = encoder.parse(valid_data)
    return (encoded_data, encoder, encoder.category_names)


def parse_cate_exist(data):
    """
    Parse categorical data into existence using 0 or 1
    """
    if isinstance(data, pd.Series):
        data = data.replace(" ", np.nan)
        data[~data.isna()] = 1
        data[data.isna()] = 0
        return data
    else:
        ret = []
        for x in data:
            if re.match("\s+", x):
                ret.append(0)
            elif x:
                ret.append(1)
            else:
                ret.append(0)
        return ret


def parse_num_exist(data):
    """
    Parse numerical data int existence using 0 or 1
    """
    if isinstance(data, pd.Series):
        data = data.replace(" ", np.nan)
        data[data > 0] = 1
        return data
    else:
        ret = []
        for x in data:
            if x > 0:
                value = 1
            elif x == 0:
                value = 0
            else:
                value = 2
            ret.append(value)
        return ret


def preprocess_cate_to_binary_encode(data):
    """
    Parse data to 1 if item exists, otherwise 0
    """
    data = parse_cate_exist(data)
    return preprocess_categorical_value(data)


def preprocess_num_to_binary_encode(data):
    """
    Parse data to 1 if item exists and its value is larger than 0; if nan, use -1 to indicate
    """
    data = parse_num_exist(data)
    return preprocess_categorical_value(data, rep_na=2)


def preprocess_charger(data, main_charger_list=None):
    """
    Preprocess charger_id
    """
    if main_charger_list is None:
        main_charger_list = MAIN_CHARGER_LIST
    data_cp = data.copy()
    for i, charger in enumerate(main_charger_list):
        data_cp[data_cp.str.contains(charger, na=False)] = i + 1
    data_cp[data_cp.str.match("\w+", na=False)] = 0
    encoder = CategoricalEncoder()
    encoded_data = encoder.parse(data_cp)
    total_charger_list = ["others"] + main_charger_list
    return (encoded_data, encoder, total_charger_list)


def preprocess_address(data, main_address_list=None):
    """
    Preprocess address
    Args:
        data: address pd series data
        main_address_list: main county address remained, otherwise used others instead
    Returns:
         onehot encoded county address
         address encoder
         total_address_list: main simplified addresses
    """
    if main_address_list is None:
        main_address_list = MAIN_ADDR_LIST
    data_cp = data.copy()
    for i, address in enumerate(main_address_list):
        data_cp[data_cp.str.contains(address, na=False)] = i + 2
    data_cp[data_cp.str.match(".*[県府都道].*", na=False)] = 1
    data_cp = data_cp.fillna("unknown")
    data_cp[data_cp == "unknown"] = 0
    # print("Addr statistic", dict(collections.Counter(data)))
    encoder = CategoricalEncoder()
    encoded_data = encoder.parse(data_cp)
    total_address_list = ["unknown", "others"] + main_address_list
    return (encoded_data, encoder, total_address_list)


def preprocess_calltime(data, triple_encode=False):
    """
    Preprocess call time
    Args:
        data: call time pd series data
        triple_encode: encode call time into 3 part or not: morning, afternoon, night
    Returns:
        one-hot encoded date time
        calltime encoder
        calltime_list
    """
    TIME_INTERVAL = ["morning", "afternoon", "night"]
    def _parse_time(x):
        conv_x = []
        for i in x:
            if 7 < i < 12:
                conv_x.append(0)
            elif 12 < i < 18:
                conv_x.append(1)
            else:
                conv_x.append(2)
        return np.array(conv_x)

    calltime = [int(x[:2]) for x in data]
    if not triple_encode:
        categories = np.expand_dims(np.arange(1, 25), 0)
        encoder = CategoricalEncoder(categories=categories)
        encoded_calltime = encoder.parse(calltime)
    else:
        categories = np.expand_dims(np.arange(len(TIME_INTERVAL)), 0)
        encoder = CategoricalEncoder(categories=categories)
        call_time_interval = _parse_time(calltime)
        print(call_time_interval)
        encoded_calltime = encoder.parse(call_time_interval)
    return (encoded_calltime, encoder, categories)


def preprocess_date_to_duration(data):
    """
    Preprocess date to duration, like convert birthday into age
    """
    age = []
    data = data.fillna("unknown")
    for i in data:
        if i == "unknown":
            age.append(i)
        else:
            age.append(2018 - int(str(i)[:4]))
    age = pd.Series(age).replace("unknown", np.nan)
    return preprocess_numerical_value(age, normalize=False)


def preprocess_gender(data):
    """
    Preprocess gender with fule: 1 for man, 0 for female and 2 for unknown,
    and return in onehot encoded data
    """
    (encoded_data, encoder, categories) = preprocess_categorical_value(data, keep_name_list=["0", "1"],
                                                                       other_name_rep=2, rep_na=2)
    encoded_categories = ["female", "male", "unknown"]
    return (encoded_data, encoder, encoded_categories)


def preprocess_industry(data, main_industry_list=None):
    """
    Preprocess industry code
    """
    if main_industry_list is None:
        main_industry_list = MAIN_INDUSTRY_LIST
    industry_items = ["industry_code1", "industry_code2", "industry_code3"]
    encoded_data_list = []
    encoder = CategoricalEncoder()
    for item in industry_items:
        data_cp = data[item].copy()
        data_cp = data_cp.fillna("unknown").astype(str)
        for i, industry in enumerate(main_industry_list):
            data_cp[data_cp.str.contains(industry, na=False)] = i + 1
        data_cp[data_cp.str.match("\d+", na=False)] = 0
        data_cp[data_cp == "unknown"] = 0
        encoded_data = encoder.parse(data_cp)
        encoded_data_list.append(encoded_data)
    encoded_data = np.sum(encoded_data_list, axis=0)
    encoded_data[:, 0] = (encoded_data[:, 0] > 0).astype(int)
    total_charger_list = ["unknown", "others"] + main_industry_list
    return (encoded_data, encoder, total_charger_list)


PREPROCESSING_FACTORY = {"week":
                             {"fn": preprocess_categorical_value,
                              "params": {"categories": [['土曜日', '日曜日', '月曜日', '木曜日',
                                                         '水曜日', '火曜日', '金曜日']]}},
                         "charger_id":
                             {"fn": preprocess_charger},
                         "call_time":
                             {"fn": preprocess_calltime,
                              "params": {"triple_encode": False}},
                         "list_type":
                             {"fn": preprocess_categorical_value,
                              "params": {"categories": [['源泉', '管S']]}},
                         "re_call_date":
                             {"fn": preprocess_cate_to_binary_encode},
                         "address":
                             {"fn": preprocess_address},
                         "kabushiki_code":
                             {"fn": preprocess_cate_to_binary_encode},
                         "establishment":
                             {"fn": preprocess_date_to_duration},
                         "shihonkin":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "employee_num":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         # "kojokazu":
                         #     {"fn": preprocess_num_to_binary_encode},
                         "kojokazu":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         # "jigyoshokazu":
                         #     {"fn": preprocess_num_to_binary_encode},
                         "jigyoshokazu":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "industry_code":
                             {"cols": ["industry_code1", "industry_code2", "industry_code3"],
                              "fn": preprocess_industry},
                         "zenkikessan_uriagedaka":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "zenkikessan_riekikin":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "tokikessan_uriagedaka":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "tokikessan_riekikin":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "tokiuriage_shinchoritsu":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "zenkiuriage_shinchoritsu":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "tokirieki_shinchoritsu":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "zenkirieki_shinchoritsu":
                             {"fn": preprocess_numerical_value,
                              "params": {"normalize": False, "use_log": False}},
                         "birthday":
                             {"fn": preprocess_date_to_duration},
                         "danjokubun":
                             {"fn": preprocess_gender},
                         "eto_meisho":
                             {"fn": preprocess_categorical_value},
                         "tosankeireki":
                             {"fn": preprocess_cate_to_binary_encode},
                         "race_area":
                             {"fn": preprocess_categorical_value}}


def test():
    filename = "train_call_history.csv"
    df = pd.read_csv(filename, encoding="utf8")
    x = ALL_USED_COLUMNS[23]
    data = df[x]
    fn = PREPROCESSING_FACTORY
    (encoded, encoder, categories) = fn[x]["fn"](data, **fn[x].get("params", {}))
    print(x, data[:5], encoded[:5], categories)