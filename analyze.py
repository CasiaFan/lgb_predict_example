import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt

filename_train = "train_call_history.csv"
filename_test = "test_call_history.csv"
df_train = pd.read_csv(filename_train, encoding="utf8")
df_test = pd.read_csv(filename_test, encoding="utf8")

def analyze_address(df):
    # get county address
    addrs = df["address"].fillna("unknown")
    coarse_addrs = [re.search(r"(.*?[県府都]).*", x).group(1) for x in addrs if re.search(r"(.*[県府都]).*", x)]
    coarse_addr_count = dict(collections.Counter(coarse_addrs))
    main_addrs = [x for x in coarse_addr_count.items() if x[1] > 1000]
    print("addr: ", list(coarse_addr_count.keys()))
    print("main_addr", main_addrs)
    # plt.bar(list(range(len(coarse_addr_count))), list(coarse_addr_count.values()))
    # plt.show()
    # ['宮城県', '東京都', '愛知県', '大阪府', '埼玉県', '神奈川県', '千葉県', '其他']
    return coarse_addr_count

def analyze_calltime(df):
    # get time (24h one-hot encode)
    call_time = df["call_time"].fillna("unknown")
    call_time = [x[:2] for x in call_time]
    time_count = dict(collections.Counter(call_time))
    print("call_time: ", list(time_count.keys()))
    # plt.bar(list(time_count.keys()), list(time_count.values()))
    # plt.show()
    return time_count

def analyze_week(df):
    # get week (7d one-hot encode )
    week = df["week"].fillna("unknown")
    week_count = dict(collections.Counter(week))
    print("week: ", list(week_count.keys()))
    # plt.bar(list(week_count.keys()), list(week_count.values()))
    # plt.show()
    return week_count

# # get date (month one-hot encode) X no need
# call_date = df["call_date"].fillna("unknown")
# call_mon = [re.search(r"\d+/(\d+)/\d+", x).group(1) for x in call_date]
# call_mon_dict = dict(collections.Counter(call_mon))
# print("mon", list(call_mon_dict.keys()))
# plt.bar(list(call_mon_dict.keys()), list(call_mon_dict.values()))
# plt.show()

# # service X no need
# service = df_train["service"]
# service_dict = dict(collections.Counter(service))
# print(service_dict)

def analyze_recall(df):
    # recall
    recall_date = df["re_call_date"].fillna("0")
    recall_count = len(recall_date[recall_date != "0"])
    total_count = len(recall_date)
    print(recall_count, total_count)

def analyze_kabushiki(df):
    # kabushiki_code
    kab = df["kabushiki_code"].fillna("0")
    kab_count = len(kab[kab != "0"])
    print("{}/{}".format(kab_count, len(kab)))
    print(np.sum(df["result"][kab != "0"]))

def analyze_shihonkin(df):
    # shihonkin
    sk = df["shihonkin"].fillna("0").astype(int)
    sk_min = min(sk)
    sk_max = max(sk)
    sk_bins = [0, 10000, 20000, 30000, 40000, 50000, 1e8]
    sk_groups = sk.groupby(pd.cut(sk, sk_bins))
    print(sk_groups.count())

def analyze_charger_id(df):
    ch = list(df["charger_id"])
    print(len(set(ch)), len(ch))
#
# # employee num
# emp_num = df["employee_num"].fillna("0").astype(int)
# emp_min = min(emp_num)
# emp_max = max(emp_num)
# # print(np.quantile(emp_num, np.arange(0., 1., 0.1)))
# emp_bins = [0, 10, 25, 50, 100, 200, 1e6]
# emp_groups = emp_num.groupby(pd.cut(emp_num, emp_bins))
# print(emp_groups.count())
#
# kojokazu 0/1
def analyze_kojo(df):
    kojo_num = np.array(df["kojokazu"].fillna("0").astype(int))
    print(len(kojo_num[kojo_num == 0]), len(kojo_num))

def analyze_position(df):
    pos = np.array(df["position"])
    print(set(pos))
    print(dict(collections.Counter(pos)))
    print(np.sum(df["result"][df["position"] == '代表取締役']))
    print(np.sum(df["result"][df["position"].isna()]))
    print(sum(df["result"]))
    print(len(df["result"]))

def analyze_jukyo(df):
    jukyo = np.array(df["jukyo"])
    print(dict(collections.Counter(jukyo)))

def analyze_saishugakureki_gakko(df):
    gakko = np.array(df["saishugakureki_gakko"].fillna("unknown"))
    uni = [x for x in gakko if re.search(r".*大学.*", x)]
    print(len(uni), len(gakko))

# # jigyoshokazu 0/1/n
# jiguo_num = df["jigyoshokazu"].fillna("0").astype(int)
# print(np.quantile(jiguo_num, np.arange(0, 1, 0.1)))
# jigyo_dict = dict(collections.Counter(jiguo_num))
# print("jigyo: ", jigyo_dict)
# # plt.bar(list(jigyo_dict.keys()), list(jigyo_dict.values()))
# # plt.show()
#
# # industry ??
# ind = df[["industry_code1", "industry_code2", "industry_code3"]].fillna("unknown")
# ind_total = np.asarray(ind).reshape(-1)
# ind_dict = collections.Counter(ind_total)
# ind_dict = sorted(ind_dict.items(), key=lambda x: x[1])[::-1]
# # print(len(ind_dict))
# # print(ind_dict[:20])
#
# # atsukaihin count ??
#
# # eigyoshumokumeisho count??
#
# # shiiresakimeisho count ??
#
# # hambaisakimeisho count ??
#
# # zenkikessan_uriagedaka: real value (norm) / onehot-encode?
# zenki_uri = df["zenkikessan_uriagedaka"].fillna("0").astype(float)
# # print(np.quantile(zenki_uri, np.arange(0, 1., 0.1)))
# zenki_uri_bins = [0, 5e5, 1e6, 2e6, 1e10]
# zenki_uri_group = zenki_uri.groupby(pd.cut(zenki_uri, zenki_uri_bins))
# print(zenki_uri_group.count())
#
# zenki_rie = df["zenkikessan_riekikin"].fillna("0").astype(float)
# zenki_rie_bins = [-1e7, 0, 1e4, 5e4, 1e5, 1e8]
# # zenkikessan_riekikin
# zenki_uri_group = zenki_uri.groupby(pd.cut(zenki_uri, zenki_uri_bins))
# print(zenki_uri_group.count())
#
# zenki_shi = df["zenkiuriage_shinchoritsu"].fillna("0").astype(float)
# # print(np.quantile(zenki_shi, np.arange(0, 1, 0.1)))
# zenki_shi_bins = [-100, 0, 80, 100, 120, 200]
# zenki_shi_group = zenki_shi.groupby(pd.cut(zenki_shi, zenki_shi_bins))
# print(zenki_shi_group.count())
#
# zenkirie_shi = df["zenkirieki_shinchoritsu"].fillna("0").astype(float)
# # print(np.quantile(zenkirie_shi, np.arange(0, 1.0, 0.1)))
# zenkirie_bins = [-1e4, 0, 50, 100, 200, 1e4]
# zenkirie_group = zenkirie_shi.groupby(pd.cut(zenkirie_shi, zenkirie_bins))
# print(zenkirie_group.count())
#
# # tokikessan_uriagedaka
# toki_uri = df["tokikessan_uriagedaka"].fillna("0").astype(float)
# # print(np.quantile(toki_uri, np.arange(0, 1, 0.1)))
# toki_uri_bins = [0, 1e6, 15e5, 2e6, 5e6, 1e8]
# toki_uri_group = toki_uri.groupby(pd.cut(toki_uri, toki_uri_bins))
# print(toki_uri_group.count())
#
# toki_rie = df["tokikessan_riekikin"].fillna("0").astype(float)
# # print(np.quantile(toki_rie, np.arange(0, 1.0, 0.1)))
# toki_rie_bins = [-1e8, 0, 1e4, 5e4, 1e5, 1e8]
# toki_rie_group = toki_rie.groupby(pd.cut(toki_rie, toki_rie_bins))
# print(toki_rie_group.count())
#
# toki_uri_shi = df["tokiuriage_shinchoritsu"].fillna("0").astype(float)
# # print(np.quantile(toki_uri_shi, np.arange(0, 1., 0.1)))
# toki_uri_shi_bins = [0, 80, 100, 120, 1e3]
# toki_uri_shi_group = toki_uri_shi.groupby(pd.cut(toki_uri_shi, toki_uri_shi_bins))
# print(toki_uri_shi_group.count())

# hitoriataririgekkan_uriagekinhitai ??

# hitoriataririgekkan_riekikingaku ??

def analyze_age(df):
    # age
    birthday = df["birthday"].dropna().astype(str)
    year = np.array([str(x)[:4] for x in birthday]).astype(int)
    age = 2018 - year
    age_dict = dict(collections.Counter(age))
    print(age_dict)
    # plt.bar(list(age_dict.keys()), list(age_dict.values()))
    # plt.show()
    return age_dict

def analyze_gender(df):
    # gender
    gender = df["danjokubun"].fillna(-1)
    gender_dict = dict(collections.Counter(gender))
    print(gender_dict)
    return gender_dict

def analyze_eto(df):
    # eto_meisho
    zodiac = df["eto_meisho"].fillna("unknown")
    zodiac_dict = dict(collections.Counter(zodiac))
    print(zodiac_dict)
    return zodiac_dict

def analyze_tosan(df):
    # tosankeireki
    tosan = df["tosankeireki"].fillna(-1)
    tosan_dict = dict(collections.Counter(tosan))
    print(tosan_dict)
    return tosan_dict

def analyze_area(df):
    # race_area
    race = df["race_area"].fillna("other")
    race_dict = dict(collections.Counter(race))
    print(race_dict)
    # plt.bar(list(race_dict.keys()), list(race_dict.values()))
    # plt.show()
    return race_dict

def analyze_type(df):
    list_type = df["list_type"].fillna("unknown")
    list_dict = dict(collections.Counter(list_type))
    # plt.bar(list(list_dict.keys()), list(list_dict.values()))
    # plt.show()
    return list_dict

def analyze_company_code(df):
    code = list(df["company_code"])
    print(len(set(code)), len(code))

analyze_saishugakureki_gakko(df_train)
# charger_id = df_train["charger_id"]
# charger_dict = dict(collections.Counter(charger_id))
# sorted_charger = dict(sorted(charger_dict.items(), key=lambda x: x[1])[::-1])
# print(sorted_charger.items())
#
# industry_code = df_train[["industry_code1", "industry_code2", "industry_code3"]]
# industry_code_dict = dict(collections.Counter(industry_code["industry_code1"].fillna(0)))
# industry_code_dict.update(dict(collections.Counter(industry_code["industry_code2"].fillna(0))))
# industry_code_dict.update(dict(collections.Counter(industry_code["industry_code3"].fillna(0))))
# sorted_industry_code = dict(sorted(industry_code_dict.items(), key=lambda x: x[1])[::-1])
# print(sorted_industry_code.items())

# label = df["result"].fillna(0)
# print("{}/{}".format(sum(label), len(label)))
# x = analyze_type(df_train)
# y = analyze_type(df_test)
# x = dict(sorted(x.items()))
# y = dict(sorted(y.items()))
# plt.subplot(2,1,1)
# plt.bar(list(x.keys()), list(x.values()))
# plt.subplot(2,1,2)
# plt.bar(list(y.keys()), list(y.values()))
# plt.show()