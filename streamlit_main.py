import streamlit as st
from streamlit import session_state as ss
from datetime import datetime as dt
from datetime import timedelta
import requests
import json
import pandas as pd
import polars as pl
import pickle


st.set_page_config(page_title="School Data", layout="wide", page_icon="🏫")


# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Password authentication
if not st.session_state["authenticated"]:
    given_password = st.text_input("Enter password", key="password_input", type="password")
    if given_password == st.secrets['PASSWORD']:
        st.session_state["authenticated"] = True
        st.rerun()
    elif given_password:
        st.write("Incorrect password. Please try again.")
    st.stop()


st.title("School Data")

with st.sidebar:
    cookie = st.text_input("Auth Cookie (ask Cope, only lasts an hour or 2)")
    if cookie:
        st.success("Cookie loaded")
    """
    I've provided a tool to generate new datasets, as well as a pre-scraped &
    cleaned dataset for you from 2016-12-21 to 2025-07-16.

    Note the download button that shows up over the datasets as you hover over them.

    You can specify a rate, and the dataset will be filtered to only include coupons
    with a coupon rate higher than the specified rate. I've also sorted them by
    what I am calling "reward", which is: `(coupon rate - specified rate) * par amount`.
    Let me know if you want me to sort it some other way, or want more conditions or
    something.

    You'll have to ask me for a new auth cookie every time you want to use it,
    as they expire after about an hour or two, and I haven't quite implemented
    automatic authentication yet. I can later, if you want.
    """


now = dt.now().strftime("%Y-%m-%d")
search_url = "https://workspace.refinitiv.com/api/tm3-backend/muni-data-analysis/deal-search/search"
detail_url = 'https://workspace.refinitiv.com/api/tm3-backend/muni-data-analysis/common/deal-analysis?dealId={deal_id}&evaluationDate='+now

def search_req_body(saleDateFrom, saleDateTo):
    return {
        "saleDateFrom": saleDateFrom.strftime("%Y-%m-%d"),
        "saleDateTo": saleDateTo.strftime("%Y-%m-%d"),
        "evaluationDate": now,
        "either": False,
        "both": True,
        "moodyLong": True,
        "moodyShort": False,
        "sandpLong": True,
        "sandpShort": False,
        "moodyRatingFrom": {
            "value": "UR",
            "label": "UR"
        },
        "moodyRatingTo": {
            "value": "Aaa",
            "label": "Aaa"
        },
        "moodyUnderlyingFrom": {
            "value": "UR",
            "label": "UR"
        },
        "moodyUnderlyingTo": {
            "value": "Aaa",
            "label": "Aaa"
        },
        "sandpRatingFrom": {
            "value": "UR",
            "label": "UR"
        },
        "sandpRatingTo": {
            "value": "AAA",
            "label": "AAA"
        },
        "sandpUnderlyingFrom": {
            "value": "UR",
            "label": "UR"
        },
        "sandpUnderlyingTo": {
            "value": "AAA",
            "label": "AAA"
        },
        "useOfProceeds": {
            "label": "EDUC Charter school",
            "value": "E5"
        },
        "cusips": [],
        "isEntitledToSp": False,
        "isAllBondInsurance": False,
        "isAllIssueType": False
    }

if 'search_log' not in ss:
    ss.search_log = []
if 'detail_log' not in ss:
    ss.detail_log = []
# Indexed by daterange
if 'search_log_data' not in ss:
    ss.search_log_data = {}
# Indexed by deal_id
if 'detail_log_data' not in ss:
    ss.detail_log_data = {}

def reset_logs():
    ss.search_log = []
    ss.detail_log = []
    ss.search_log_data = {}
    ss.detail_log_data = {}

def search(from_date, to_date, _log=True, tabs=0):
    print(f'{"\t" * tabs}Searching {from_date} to {to_date}...')
    # resp = requests.post(search_url, headers={'cookie': cookie}, data=search_req_body(from_date, to_date))
    resp = requests.post(search_url, headers={'cookie': cookie}, json=search_req_body(from_date, to_date))
    print(f'{"\t" * tabs}...finished searching with {resp.status_code}')
    ss.search_log.append(resp)
    if resp.status_code == 204:
        return {'data': [], 'overThreshold': False}

    if resp.status_code != 200:
        print('!!! Error searching from', from_date, 'to', to_date, ' !!!')

    j = resp.json()
    # If there's more data, recurse in halfs, and combine the data
    if j['overThreshold']:
        print(f'{"\t" * tabs}Over threshold, recursing...')
        half = from_date + ((to_date - from_date) // 2)
        half1 = search(from_date, half, _log=False, tabs=tabs+1)
        half2 = search(half, to_date, _log=False, tabs=tabs+1)
        j = {'data': half1['data'] + half2['data'], 'overThreshold': False}
        print(f'{"\t" * tabs}...finished recursing')

    if _log:
        ss.search_log_data[(from_date, to_date)] = j

    return j

def details(deal_id):
    print(f'Fetching details for {deal_id}...')
    resp = requests.get(detail_url.format(deal_id=deal_id), headers={'cookie': cookie})
    print(f'...finished fetching details with {resp.status_code}')
    ss.detail_log.append(resp)
    j = resp.json()
    ss.detail_log_data[deal_id] = j
    return j

@st.cache_data
def get_schools(from_date, to_date):
    schools = []
    # Search by month
    i = from_date
    while i <= to_date:
        schools += search(i, i+timedelta(days=30))['data']
        i += timedelta(days=30)
    try:
        return pl.DataFrame(schools)
    except:
        return schools

@st.cache_data
def get_deals(deal_ids):
    school_bonus = []
    coupons = []
    for deal_id in deal_ids:
        d = details(deal_id)['data']

        dd = d['dealDetails']
        for i in dd:
            i['deal_id'] = deal_id
        school_bonus += dd

        cd = d['cusipDetails']
        for i in cd:
            i['deal_id'] = deal_id
        coupons += cd
    try:
        return pl.DataFrame(coupons), pl.DataFrame(school_bonus)
    except:
        return coupons, school_bonus

from cryptography.fernet import Fernet
import base64
import hashlib

def password_to_key(password: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

def decrypt_file_to_df(encrypted_path: str, password: str):
    key = password_to_key(password)
    fernet = Fernet(key)

    with open(encrypted_path, 'rb') as f:
        encrypted = f.read()

    try:
        decrypted = fernet.decrypt(encrypted)
    except Exception as e:
        raise ValueError("Invalid password or corrupted file.") from e

    from io import BytesIO
    buffer = BytesIO(decrypted)

    return pl.read_csv(buffer)


def get_preloaded_data():
    # return pl.read_csv('backup_data/target.csv'),
    return decrypt_file_to_df('backup_data/bonds_encrypted.bin', st.secrets['PASSWORD']), decrypt_file_to_df('backup_data/coupons_encrypted.bin', st.secrets['PASSWORD'])

def show_target(bonds, coupons, key='1'):
    base = st.number_input("Coupon rate you think you can get them right now", value=5.25, key=key+'base')
    coupons = coupons.with_columns(reward=(pl.col('coupon_rate') - base) * pl.col('par_amount'))
    target_coupons = coupons.filter(pl.col('coupon_rate') > base)
    target = target_coupons.rename({'deal_id': 'dealId'}).join(bonds, on='dealId', how='left').sort('reward', descending=True)
    column_order = ['cusip', 'dealId', 'coupon_rate', 'par_amount', 'issuer', 'corpProject', 'leadManager', 'reward']
    # Put samples with a null par_amount at the bottom
    df = target.filter(pl.col('par_amount').is_not_null())
    target = pl.concat([df, target.filter(pl.col('par_amount').is_null())])
    st.write(target
        .select(column_order + [col for col in target.columns if col not in column_order])
    , key=key+'target')

with st.expander("Generate a custom dataset from a given daterange", expanded=False):
    """
    Warning: large dateranges (>1 or 2 months) can take a while. Ranges > ~1 year may fail. Have Cope generate those manually
    """
    try:
        from_date, to_date = st.date_input("From Date", value=(dt(2025, 1, 1), dt(2025, 2, 1)))
    except ValueError:
        st.error("Invalid date range")
        st.stop()



    # search button
    if st.button("Scrape"):
        if len(cookie) < 50:
            st.error("Invalid cookie: ask Cope needs to get you a new one")
            st.stop()
        try:
            schools = get_schools(from_date, to_date)
            coupons, bonus_school_data = get_deals(schools['dealId'].to_list())

            try:
                additional_data = schools.select('first_coupon_date', 'first_call_date', 'security_type', 'bank_qualified', 'sp_rating', 'bond_insurance', 'paying_agent', 'deal_id').rename({'deal_id': 'dealId'})
            except Exception as e:
                bonds = schools.unique('dealId')
            else:
                bonds = schools.unique('dealId').join(additional_data, on='dealId', how='left')
            show_target(bonds, coupons)

        except Exception as e:
            st.error('Something went wrong. Please download this file and give it to Cope:')
            # TODO: delete this later
            st.exception(e)
            # Pickle everything and stuff it in a single zip file
            st.download_button("Download logs",
                data=pickle.dumps({
                    'from_date': from_date,
                    'to_date': to_date,
                    'cookie': cookie,
                    'error': e,
                    'error_str': str(e),
                    'search_log': ss.search_log,
                    'detail_log': ss.detail_log,
                    'search_log_data': ss.search_log_data,
                    'detail_log_data': ss.detail_log_data
                }),
                file_name=f"log-{now}.pkl"
            )

            # """ Search logs: """
            # st.write(ss.search_log)

            # st.write(ss.detail_log)
            # st.write(ss.search_log_data)
            """ Data gathered so far: """
            # st.write(ss.detail_log_data)

with st.expander("Pre-filtered data from 2016-12-21 to 2025-07-16", expanded=True):
    bonds, coupons = get_preloaded_data()
    show_target(bonds, coupons, key='2')