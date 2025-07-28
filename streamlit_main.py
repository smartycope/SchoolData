import streamlit as st
from streamlit import session_state as ss
from datetime import datetime as dt
from datetime import timedelta
import requests
import json
# import pandas as pd
import polars as pl
import pickle

from cryptography.fernet import Fernet
import base64
import hashlib

password_hash = {
    'salt': 'iB8/Yzann0b8AXIB2vt9RQ==',
    'hash': 'lC0xsC5D8lpNfg1d8DwFSDMaSVYncZcjjqd0Lb/M6eE=',
    'iterations': 10000
}

def verify_password(password, hash):
    salt = base64.b64decode(hash['salt'])
    expected_hash = base64.b64decode(hash['hash'])
    test_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, hash['iterations'])
    return test_hash == expected_hash


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

st.set_page_config(page_title="School Data", layout="wide", page_icon="🏫", initial_sidebar_state="collapsed")

if True:
    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Password authentication
    if not st.session_state["authenticated"]:
        given_password = st.text_input("Enter password", key="password_input", type="password")
        try:
            st.session_state["authenticated"] = verify_password(given_password, password_hash)
            st.rerun()
        except:
            st.write("Incorrect password. Please try again.")
        st.stop()


@st.cache_data
def get_preloaded_data():
    return decrypt_file_to_df('backup_data/bonds_encrypted.bin', given_password), decrypt_file_to_df('backup_data/coupons_encrypted.bin', given_password)




    # search button
    # if st.button("Scrape"):
    #     if len(cookie) < 50:
    #         st.error("Invalid cookie: ask Cope needs to get you a new one")
    #         st.stop()
    #     try:
    #         schools = get_schools(from_date, to_date)
    #         coupons, bonus_school_data = get_deals(schools['dealId'].to_list())

    #         try:
    #             additional_data = schools.select('first_coupon_date', 'first_call_date', 'security_type', 'bank_qualified', 'sp_rating', 'bond_insurance', 'paying_agent', 'deal_id').rename({'deal_id': 'dealId'})
    #         except Exception as e:
    #             bonds = schools.unique('dealId')
    #         else:
    #             bonds = schools.unique('dealId').join(additional_data, on='dealId', how='left')
    #         show_target(bonds, coupons)

    #     except Exception as e:
    #         st.error('Something went wrong. Please download this file and give it to Cope:')
    #         # TODO: delete this later
    #         st.exception(e)
    #         # Pickle everything and stuff it in a single zip file
    #         st.download_button("Download logs",
    #             data=pickle.dumps({
    #                 'from_date': from_date,
    #                 'to_date': to_date,
    #                 'cookie': cookie,
    #                 'error': e,
    #                 'error_str': str(e),
    #                 'search_log': ss.search_log,
    #                 'detail_log': ss.detail_log,
    #                 'search_log_data': ss.search_log_data,
    #                 'detail_log_data': ss.detail_log_data
    #             }),
    #             file_name=f"log-{now}.pkl"
    #         )

    #         # """ Search logs: """
    #         # st.write(ss.search_log)

    #         # st.write(ss.detail_log)
    #         # st.write(ss.search_log_data)
    #         """ Data gathered so far: """
            # st.write(ss.detail_log_data)

# with st.expander("Pre-filtered data from 2016-12-21 to 2025-07-16", expanded=True):
bonds, coupons = get_preloaded_data()

st.title("School Bonds")
cookie = ''
with st.sidebar:
    # cookie = st.text_input("Auth Cookie (ask Cope, only lasts an hour or 2)")
    # if cookie:
    #     st.success("Cookie loaded")
    precision = st.number_input("Precision", value=1, min_value=1, max_value=10, step=1, format='%d')
    max_date = dt.strptime(bonds['datedDate'].max(), '%Y-%m-%d')
    if max_date < dt.now() - timedelta(days=365):
        box = st.error
    elif max_date < dt.now() - timedelta(days=6*30):
        box = st.warning
    else:
        box = st.success
    l, r = st.columns([2, 1])
    with l:
        box(f"Data is <{int((dt.now() - max_date).days)*-1} days old")
    if r.button('Update'):
        st.warning('This button doesn\'t currently do anything, but it will soon!')

    f"""
    Data range: {bonds['datedDate'].min()} to {max_date.strftime('%Y-%m-%d')}

    - To download: there's a download button that shows up over the data as you hover over it
    - To see which columns are hidden: click the eye icon that shows up over the data as you hover over it
    - To sort: click the column headers
    - To search for a specific row: use the search icon

    The data is filtered by reward (`(coupon rate - specified rate) * par amount`) by default.
    """


now = dt.now().strftime("%Y-%m-%d")
search_url = "https://workspace.refinitiv.com/api/tm3-backend/muni-data-analysis/deal-search/search"
detail_url = 'https://workspace.refinitiv.com/api/tm3-backend/muni-data-analysis/common/deal-analysis?dealId={deal_id}&evaluationDate='+now
# TM3_URL = "https://workspace.refinitiv.com/muni-data-analysis/deal-search/deal/{deal_id}"
TM3_DEAL_URL = "https://workspace.refinitiv.com/web/rap/tm3-app/muni-data-analysis/deal-search/details/{deal_id}/deal-analysis?evaluationDate="+now
TM3_CUSIP_URL = 'https://workspace.refinitiv.com/web/rap/tm3-app/muni-data-analysis/cusip-search/details/{cusip}/main?evaluationDate=' + now
EMMA_URL = 'https://emma.msrb.org/QuickSearch/Results?quickSearchText={cusip}'

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

@st.cache_data
def get_target(coupons, bonds):
    df = (coupons
        .rename({'deal_id': 'dealId'})
        .join(bonds, on='dealId', how='left')
        .with_columns(
            state=pl.col('state').cast(pl.Categorical),
            datedDate=pl.col('datedDate').cast(pl.Date),
            maturity_date=pl.col('maturity_date').cast(pl.Date),
            first_coupon_date=pl.col('first_coupon_date').str.strptime(pl.Date, '%m/%d/%Y', strict=False),
            first_call_date=pl.col('first_call_date').str.strptime(pl.Date, '%m/%d/%Y', strict=False),
            saleDate=pl.col('saleDate').cast(pl.Date),
            TM3_url=pl.col('dealId').map_elements(lambda x: TM3_DEAL_URL.format(deal_id=x)),
            TM3_cusip_url=pl.col('cusip').map_elements(lambda x: TM3_CUSIP_URL.format(cusip=x)),
            emma_url=pl.col('cusip').map_elements(lambda x: EMMA_URL.format(cusip=x)),
        )
        # Get the sums
        .with_columns(total_par_amount=pl.col('par_amount').sum().over('dealId'))
        # Get the weights
        .with_columns(contrib=pl.col('par_amount') / pl.col('total_par_amount'))
        # Weight the rates
        .with_columns(weighted_rate=pl.col('coupon_rate') * pl.col('contrib'))
    )

    grouped = (df
        # Sum the weighted rates
        .group_by('dealId')
        .agg(tic=pl.col('weighted_rate').sum(), cusips=pl.col('cusip'))
    )
    # Join the aggregated data back to the target
    # I have no idea why this expands to the original number of rows, it shouldn't
    df = grouped.join(df, on='dealId', how='left')

    df = df.with_columns(num_cusips=pl.col('cusips').list.len())

    # Put samples with a null par_amount at the bottom
    tmp = df.filter(pl.col('par_amount').is_not_null())
    df = pl.concat([tmp, df.filter(pl.col('par_amount').is_null())])

    return df

hide = [
    'contrib',
    'par_amount',
    'bank_qualified',
    'bond_insurance',
    'sp_rating',
    'moody_rating',
    'sand_rating',
    'fitch_rating',
    'offering_type',
    'issue_type',
    'weighted_rate',
]

# Dicts are ordered since Python 3.7+
cc = st.column_config
column_config = {
    'TM3_url': cc.LinkColumn('TM3 URL', width='small'),
    'TM3_cusip_url': cc.LinkColumn('TM3 CUSIP URL', width='small'),
    'emma_url': cc.LinkColumn('Emma URL', width='small'),
    'cusip': 'CUSIP',
    'par_amount': cc.NumberColumn('Coupon Par Amount ($K)', format='localized'),
    'total_par_amount': cc.NumberColumn('Deal Par Amount ($K)', format='localized'),
    'tic': cc.NumberColumn('TIC', format=f'%.{precision}f%%'),
    'coupon_rate': cc.NumberColumn('Coupon Rate', format=f'%.{precision}f%%'),

    'corpProject': 'Corp Project',
    'leadManager': 'Lead Manager',
    'issuer': 'Issuer',

    'dealAmount': cc.NumberColumn('Deal Amount ($K)', format='localized'),
    'series': 'Series',
    'state': 'State',

    'reward': 'Reward Score',
    'weighted_rate': 'Weighted Rate',
    'num_cusips': 'Number of Coupons',
    'cusips': 'Associated Coupons',

    'datedDate': cc.DateColumn('Dated Date'),
    'maturity_date': cc.DateColumn('Maturity Date'),
    'first_coupon_date': cc.DateColumn('First Coupon Date'),
    'first_call_date': cc.DateColumn('First Call Date'),
    'saleDate': cc.DateColumn('Sale Date'),

    'security_type': 'Security Type',
    'bank_qualified': 'Bank Qualified',
    'sp_rating': 'SP Rating',
    'bond_insurance': 'Bond Insurance',
    'taxStatus': 'Tax Status',
    'moody_rating': 'Moody Rating',
    'sand_rating': 'Sand Rating',
    'fitch_rating': 'Fitch Rating',
    'issue_type': 'Issue Type',
    'offering_type': 'Offering Type',
    'priceTo': 'Price To',

    'original_price': 'Original Price',
    'original_yield': 'Original Yield',
    'evaluated_price': 'Evaluated Price',
    'evaluated_yield': 'Evaluated Yield',
    'spread': 'Spread',
    'spread_interpolated': 'Spread Interpolated',
    'spread_to_mmd': 'Spread To MMD',
    'spread_to_mmd_interpolated': 'Spread To MMD Interpolated',
    'paying_agent': 'Paying Agent',

    'dealId': 'Deal ID',
}

hide_nulls = [
    'tic',
    'coupon_rate',
    'dealId',
    'cusip',
    'state',
    'dealAmount',
    'total_par_amount',
]

target = get_target(coupons, bonds)
# with st.form("Pre-filtered data"):
with st.container(border=True):
    l, m, r = st.columns(3)
    state = l.selectbox("State", ['All'] + bonds['state'].unique().sort().to_list())
    range_col = "total_par_amount"
    range_scale = 1_000

    ml, mr = m.columns(2)
    until = ml.date_input("Show deals up until...", value=dt.now()-timedelta(days=6*30))
    date_columns = ['datedDate', 'maturity_date', 'first_coupon_date', 'first_call_date', 'saleDate']
    until_col = mr.selectbox("...using", date_columns, format_func=lambda x: column_config[x]['label'])
    min_tic = r.number_input("Don't show deals with a TIC lower than", value=5.25, help='This is also what reward is based on')

    l, r = st.columns([5, 1], vertical_alignment='center')
    min, max = int(target[range_col].min()//range_scale), 50_000
    range_min, range_max = l.slider("Deal Par Amount Range", min, max, (0, max), format='$%dK', step=1_000)
    dont_limit_range = r.checkbox("Don't limit", value=False, help="I picked 50K, because deal par amounts are less than that. The really big ones are outliers.")

    min, max = target['num_cusips'].min(), 25
    l, r = st.columns([5, 1], vertical_alignment='center')
    min_cusips, max_cusips = l.slider("Number of Coupons", min, max, (min, max))
    dont_limit_cusips = r.checkbox("Don't limit", value=False, help="I picked 25, because most deals have less than that. Ones that have a lot are outliers.")

    exclude_null = hide_nulls.copy()
    with st.expander("Null values"):
        st.write("Check to exclude the schools with null values in the following columns:")
        columns = st.columns(4)
        for cnt, tmp in enumerate(column_config.items()):
            col, name = tmp
            name = name if isinstance(name, str) else name['label']
            if columns[cnt % 4].checkbox(name, value=col in hide_nulls, key=col):
                exclude_null.append(col)
    # l, r = st.columns(2)
    show_coupons = st.checkbox("Show individual coupons", value=False)
    # l.form_submit_button("Submit")

if not show_coupons:
    hide.append('cusip')
    hide.append('TM3_cusip_url')
    hide.append('coupon_rate')
else:
    hide.append('tic')

deals = target.unique('dealId') if not show_coupons else target
shown = (deals
    .with_columns(
        reward=(pl.col('coupon_rate') - min_tic) * pl.col('par_amount'),
        total_par_amount=pl.col('total_par_amount')/1000,
        par_amount=pl.col('par_amount')/1000,
        dealAmount=pl.col('dealAmount')/1000,
    )
    .filter(pl.col('tic') > min_tic)
    .filter(pl.col(until_col) < until)
    .filter(pl.col('state') == state if state != 'All' else True)
    .filter((pl.col('num_cusips') >= min_cusips) & (pl.col('num_cusips') <= max_cusips))
    .filter((pl.col(range_col) >= range_min) & (pl.col(range_col) <= range_max))
    .sort('reward', descending=True)
)
# st.write(exclude_null)
for col in exclude_null:
    shown = shown.filter(pl.col(col).is_not_null())

st.write(f'`{len(shown)}`/`{len(deals if not show_coupons else target)}` {"schools" if not show_coupons else "coupons"} selected')

st.dataframe(shown,
    column_config=column_config | {name: None for name in hide},
    column_order=column_config.keys(),
)

# import plotly.express as px
# fig = px.histogram(data_frame=deals, x='num_cusips')
# st.plotly_chart(fig)

