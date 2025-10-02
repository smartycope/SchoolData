import base64
import hashlib
from datetime import datetime as dt
from datetime import timedelta
from io import BytesIO
import narwhals as nw
import numpy as np
import pandas as pd
import pprp
from cryptography.fernet import Fernet
import streamlit as st
from streamlit import session_state as ss
from copy import copy

password_hash = {
    'salt': 'yITsoJop3O7QiNnzSdeTSA==',
    'hash': '1KuqXs83yQ6v7vGqYvE2LJG0N4PpvNAYUbk5dHJ8Mes=',
    'key_size': 32,
}

now = dt.now().strftime("%Y-%m-%d")
search_url = "https://workspace.refinitiv.com/api/tm3-backend/muni-data-analysis/deal-search/search"
detail_url = 'https://workspace.refinitiv.com/api/tm3-backend/muni-data-analysis/common/deal-analysis?dealId={deal_id}&evaluationDate='+now
TM3_DEAL_URL = "https://workspace.refinitiv.com/web/rap/tm3-app/muni-data-analysis/deal-search/details/{deal_id}/deal-analysis?evaluationDate="+now
TM3_CUSIP_URL = 'https://workspace.refinitiv.com/web/rap/tm3-app/muni-data-analysis/cusip-search/details/{cusip}/main?evaluationDate=' + now
EMMA_URL = 'https://emma.msrb.org/QuickSearch/Results?quickSearchText={cusip}'

if 'search_log' not in ss: ss.search_log = []
if 'detail_log' not in ss: ss.detail_log = []
# Indexed by daterange
if 'search_log_data' not in ss: ss.search_log_data = {}
# Indexed by deal_id
if 'detail_log_data' not in ss: ss.detail_log_data = {}

# Read-only state from query params
STATE = {
    'state': 'All', # st.query_params.get('state', 'All'))
    'filter_sale_date': False,
    # None means use the min and max of the data (unspecified)
    'sale_date_min': None,
    'sale_date_max': None,
    'exclude_matured': True,
    'exclude_taxable': True,
    'exclude_nulls': [],
    # None means use the min and max of the data (unspecified)
    'maturity_date_min': None,
    'maturity_date_max': None,
}

# Get the current state, serializable into query params
# TODO: this is unfinished
def get_state():
    return st.session_state


def verify_password(password: str, stored: dict):
    password_bytes = password.encode('utf-8')
    salt = base64.b64decode(stored["salt"])
    expected = base64.b64decode(stored["hash"])
    dk = pprp.pbkdf2(password_bytes, salt, stored["key_size"])
    return dk == expected

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

    buffer = BytesIO(decrypted)

    return nw.from_native(pd.read_csv(buffer))

st.set_page_config(page_title="School Data", layout="wide", page_icon="🏫", initial_sidebar_state="expanded")

PASSWORD_FOR_DEBUGGING = 'round_table_rocks'
DEBUG = bool(PASSWORD_FOR_DEBUGGING)
if DEBUG:
    st.session_state["authenticated"] = PASSWORD_FOR_DEBUGGING
else:
    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Password authentication
    if not st.session_state["authenticated"]:
        given_password = st.text_input("Enter password", key="password_input", type="password")
        if given_password and verify_password(given_password, password_hash):
            st.session_state["authenticated"] = given_password
            st.rerun()
        elif given_password:
            st.error("Incorrect password. Please try again.")
        st.stop()


@st.cache_data
def get_preloaded_data():
    return (
        decrypt_file_to_df('backup_data/bonds_encrypted.bin', st.session_state["authenticated"]),
        decrypt_file_to_df('backup_data/coupons_encrypted.bin', st.session_state["authenticated"]),
    )

bonds, coupons = get_preloaded_data()

@st.cache_data
def get_target():
    global bonds, coupons

    print(bonds.columns)
    print(coupons.columns)

    # Convert to Narwhals DataFrame and rename column
    df = coupons.rename({'deal_id': 'dealId'})

    # Join with bonds
    df = df.join(bonds, on='dealId', how='left')

    def format_strings(func, col):
        return np.array([func(i) for i in list(col)])

    # Convert date columns and create URL columns
    df = df.with_columns(
        state=nw.col('state').cast(nw.String),
        datedDate=nw.col('datedDate').str.to_datetime(),
        maturity_date=nw.col('maturity_date').str.to_datetime(),
        first_coupon_date=nw.col('first_coupon_date').str.to_datetime(),
        first_call_date=nw.col('first_call_date').str.to_datetime(),
        saleDate=nw.col('saleDate').str.to_datetime(),
        # I hate these lines
        TM3_url=format_strings(lambda i: TM3_DEAL_URL.format(deal_id=i), df['dealId']),
        TM3_cusip_url=format_strings(lambda i: TM3_CUSIP_URL.format(cusip=i), df['cusip']),
        emma_url=format_strings(lambda i: EMMA_URL.format(cusip=i), df['cusip']),
    )

    # Calculate total par amount per deal
    total_par = df.group_by('dealId').agg(total_par_amount=nw.col('par_amount').sum())
    df = df.join(total_par, on='dealId', how='left')

    def pretty_delta(seconds: int, include_days=False, years_threshold=5, decimal_years=False):
        s = abs(seconds)
        if np.isnan(s):
            return
        years = int(s // (365 * 24 * 60 * 60))
        years_decimal = (s % (365 * 24 * 60 * 60)) / (365 * 24 * 60 * 60)
        months = int((s % (365 * 24 * 60 * 60)) // (30 * 24 * 60 * 60))
        days = int((s % (30 * 24 * 60 * 60)) // (24 * 60 * 60))

        rtn = ""
        if years:
            if years > years_threshold:
                if decimal_years:
                    rtn += f"{years + years_decimal:.1f} years"
                else:
                    rtn += f"~{years} years"
            else:
                rtn += f"{years:2d} years"
        if months and years < years_threshold:
            rtn += f" {months:2d} months"
        # If years and days is 0, then forcibly include days, if there are any
        if include_days or (not rtn and days):
            rtn += f" {days:2d} days"

        if not rtn:
            return "today"

        if seconds > 0:
            rtn = f"{rtn} ago"
        elif seconds < 0:
            rtn = f"In {rtn}"
        return rtn

    # Calculate weights and weighted rates
    df = df.with_columns(
        contrib=nw.col('par_amount') / nw.col('total_par_amount'),
        weighted_rate=nw.col('coupon_rate') * (nw.col('par_amount') / nw.col('total_par_amount')),
        saleDelta=(dt.now() - nw.col('saleDate')).dt.total_seconds(),
        maturityDelta=(dt.now() - nw.col('maturity_date')).dt.total_seconds(),
        first_call_delta=(dt.now() - nw.col('first_call_date')).dt.total_seconds(),
        first_coupon_delta=(dt.now() - nw.col('first_coupon_date')).dt.total_seconds(),
    )

    df = df.with_columns(
        saleDelta=format_strings(pretty_delta, df['saleDelta']),
        maturityDelta=format_strings(pretty_delta, df['maturityDelta']),
        first_call_delta=format_strings(pretty_delta, df['first_call_delta']),
        first_coupon_delta=format_strings(pretty_delta, df['first_coupon_delta']),
    )

    # Group by dealId and calculate tic and collect cusips
    grouped = df.group_by('dealId').agg(
        tic=nw.col('weighted_rate').sum(),
        # num_cusips=nw.col('cusip').count(),
        num_cusips=nw.col('cusip').n_unique(),
        # cusips=nw.col('cusip').collect(),
        # TODO: come back to this
        # cusips=nw.col('cusip').unique().cast(nw.List(nw.String))
    )

    # Join the aggregated data back to the original dataframe
    df = df.join(grouped, on='dealId', how='left')

    # Calculate number of cusips
    # TODO: come back to this
    # df = df.with_columns(num_cusips=nw.col('cusips').list.len())

    # Put samples with a null par_amount at the bottom
    # tmp = df.filter(~nw.col('par_amount').is_null())
    # nulls = df.filter(nw.col('par_amount').is_null())
    # df = nw.concat([tmp, nulls])

    return df


target = get_target()
precision = st.sidebar.number_input("Precision (digits)", value=1, min_value=1, max_value=10, step=1, format='%d')

hide_nulls = [
    'tic',
    'coupon_rate',
    'dealId',
    'cusip',
    'state',
    'dealAmount',
    'total_par_amount',
]

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
    'par_amount': cc.NumberColumn('Coupon Par Amount ($K)', format='localized'),
    'total_par_amount': cc.NumberColumn('Deal Par Amount ($K)', format='localized'),
    'tic': cc.NumberColumn('TIC', format=f'%.{precision}f%%'),
    'coupon_rate': cc.NumberColumn('Coupon Rate', format=f'%.{precision}f%%'),
    'saleDate': cc.DateColumn('Sale Date'),
    'saleDelta': '...Sale Date is',
    'state': 'State',

    'num_cusips': '# Coupons',

    'maturity_date': cc.DateColumn('Maturity Date'),
    'maturityDelta': '...Maturity Date is',

    'first_call_date': cc.DateColumn('First Call Date'),
    'first_call_delta': '...First Call Date is',

    'corpProject': 'Corp Project',
    'leadManager': 'Lead Manager',
    'issuer': 'Issuer',

    'dealAmount': cc.NumberColumn('Deal Amount ($K)', format='localized'),
    'series': 'Series',

    'first_coupon_date': cc.DateColumn('First Coupon Date'),
    'first_coupon_delta': '...First Coupon Date is',
    'datedDate': cc.DateColumn('Dated Date'),

    'cusip': 'CUSIP',
    'dealId': 'Deal ID',

    'reward': 'Reward Score',
    'weighted_rate': 'Weighted Rate',
    'cusips': 'Associated Coupons',

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

    'original_price':  cc.NumberColumn('Original Price', format='dollar'),
    'original_yield':  cc.NumberColumn('Original Yield', format=f'%.{precision}f%%'),
    'evaluated_price': cc.NumberColumn('Evaluated Price', format='dollar'),
    'evaluated_yield': cc.NumberColumn('Evaluated Yield', format=f'%.{precision}f%%'),
    'spread': cc.NumberColumn('Spread', format=f'%.{precision}f'),
    'spread_interpolated': cc.NumberColumn('Spread Interpolated', format=f'%.{precision}f'),
    'spread_to_mmd': cc.NumberColumn('Spread To MMD', format=f'%.{precision}f'),
    'spread_to_mmd_interpolated': cc.NumberColumn('Spread To MMD Interpolated', format=f'%.{precision}f'),
    'paying_agent': 'Paying Agent',
}

############### Sidebar ###############
st.title("School Bonds")
with st.sidebar:
    max_date = target['saleDate'].max()
    if max_date < dt.now() - timedelta(days=365):
        box = st.error
    elif max_date < dt.now() - timedelta(days=6*30):
        box = st.warning
    else:
        box = st.success
    l, r = st.columns([2, 1])
    with l:
        box(f"Data is ~{int((dt.now() - max_date).days)} days old")

    f"""
    Sale date range: {target['saleDate'].min().strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}

    - To download: there's a download button that shows up over the data as you hover over it
    - To see which columns are hidden: click the eye icon that shows up over the data as you hover over it
    - To sort: click the column headers
    - To search for a specific row: use the search icon

    The data is filtered by reward (`coupon rate * par amount`) by default.
    """

    # Nulls
    exclude_null = hide_nulls.copy()
    with st.expander("Null values", expanded=False):
        st.write("Check to exclude the schools with values which aren't provided in the following columns:")
        st.caption("Basically, if you ever find yourself going \"I keep seeing `None`, I wish those ones would go away\", use this.")
        num_cols = 2
        columns = st.columns(num_cols)
        for cnt, tmp in enumerate(column_config.items()):
            col, name = tmp
            name = name if isinstance(name, str) else name['label']
            if columns[cnt % num_cols].checkbox(name, value=col in hide_nulls, key=col):
                exclude_null.append(col)

############### UI ###############
show_coupons = st.checkbox("Show individual coupons", value=False)
with st.container(border=True):
    now = dt.now()
    six_months_ago = now - pd.offsets.DateOffset(months=6)
    six_months_from_now = now + pd.offsets.DateOffset(months=6)

    l, m, r = st.columns([1, 2, 2])
    # State box
    with l:
        options = ['All'] + bonds['state'].unique().sort().to_list()
        state = st.selectbox("State", options, index=options.index(STATE['state']))
    # Sale date range
    with m:
        filter_sale_date = st.checkbox("Filter by sale date", value=STATE['filter_sale_date'])
        val0, val1 = STATE['sale_date_min'] or target['saleDate'].min(), STATE['sale_date_max'] or target['saleDate'].max()
        sale_date_min, sale_date_max = st.date_input(
            "Allowed Sale Date Range",
            value=(val0, val1),
            disabled=not filter_sale_date)
    # Maturity date range & checkbox
    with r:
        exclude_matured = st.checkbox(
            "Exclude matured bonds, and bonds about to mature",
            value=STATE['exclude_matured'],
            help="If a coupon has a maturity date in the past, or in the next 6 months, don't include it. But if a bond has at least one coupon expiring more than 6 months in the future, include it."
        )
        val0, val1 = STATE['maturity_date_min'] or target['maturity_date'].min(), STATE['maturity_date_max'] or target['maturity_date'].max()
        maturity_date_min, maturity_date_max = st.date_input(
            "Allowed Maturity Date Range",
            value=(val0, val1),
            disabled=exclude_matured
        )

    l, r = st.columns([1, 3])
    with l:
        if show_coupons:
            options = ['Coupon Rate', 'Par Amount', "Nothing"]
        else:
            options = ['TIC', 'Coupon Rate', 'Par Amount', "Nothing"]
        filter_by = st.selectbox("Filter based on", options, key='filter_by')
    with r:
        # TIC box
        if filter_by == 'TIC':
            min_tic = st.number_input("Don't show deals with a TIC lower than", value=5.25)
        elif filter_by == 'Par Amount':
            range_col = "total_par_amount" if not show_coupons else "par_amount"
            range_scale = 1_000
            min, max = int(target[range_col].min()//range_scale), 50_000
            range_min, range_max = st.slider(f"{range_col.replace('_', ' ').title()} Range", min, max, (0, max), format='$%dK', step=range_scale)
        elif filter_by == 'Coupon Rate':
            # scale = 1
            # min, max = int(target['coupon_rate'].min()//scale), 50_000
            min, max = target['coupon_rate'].min(), target['coupon_rate'].max()
            coupon_range_min, coupon_range_max = st.slider(f"Coupon Rate Range", min, max, (min, max), format='%f%%', step=.1, help='If looking at deals instead of induvidual coupons, it only filters out deals where *all* of it\'s coupons are outside the range.')
            # min_coupon_rate = st.number_input("Don't show deals with a coupon rate lower than", value=6.5)

    # Number of coupons slider
    min, max = target['num_cusips'].min(), 20
    l, r = st.columns([1, 5], vertical_alignment='center')
    limit_cusips = l.checkbox("Filter by number of coupons", value=False)#, help="I picked 25, because most deals have less than that. Ones that have a lot are outliers.")
    min_cusips, max_cusips = r.slider("Number of Coupons", min, max, (min, max), disabled=not limit_cusips)

    # Checkboxes
    exclude_taxable = st.checkbox("Exclude taxable", value=True)

############### Data Filtering ###############
if not show_coupons:
    hide.append('cusip')
    hide.append('TM3_cusip_url')
    hide.append('coupon_rate')
else:
    hide.append('total_coupon_rate')
    hide.append('num_cusips')
    hide.append('tic')

deals = target
if not show_coupons:
    num_unfiltered = len(deals.unique('dealId'))
else:
    num_unfiltered = len(deals)
breakdown = [('Total Coupons', len(deals))]
shown = (
    deals.with_columns(
        reward=nw.col('coupon_rate') * nw.col('par_amount'),
        total_par_amount=nw.col('total_par_amount')/1000,
        par_amount=nw.col('par_amount')/1000,
        dealAmount=nw.col('dealAmount')/1000,
    )
    # We have to make until a datetime, not a date, cause narwals in Pyodide can't support a date type column
    # .filter(nw.col(until_col) < dt.combine(until, dt.min.time()))
    .filter(
        (nw.col('saleDate') <= dt.combine(sale_date_max, dt.min.time())) &
        (nw.col('saleDate') >= dt.combine(sale_date_min, dt.min.time()))
    )
    .sort('reward', descending=True)
)
breakdown.append(('Sale Date out of range', len(shown)))

if filter_by == 'TIC':
    shown = shown.filter(nw.col('tic') > min_tic)
    breakdown.append(('TIC out of range', len(shown)))
elif filter_by == 'Par Amount':
    # if not dont_limit_range:
    shown = shown.filter(
        (nw.col('par_amount') >= range_min) &
        (nw.col('par_amount') <= range_max)
    )
    breakdown.append(('Par Amount out of range', len(shown)))
elif filter_by == 'Coupon Rate':
    shown = shown.filter(
        (nw.col('coupon_rate') >= coupon_range_min) &
        (nw.col('coupon_rate') <= coupon_range_max)
    )
    breakdown.append(('Coupon Rate out of range', len(shown)))

if exclude_taxable:
    shown = shown.filter(nw.col('taxStatus') != 'US FEDERAL TAXABLE')
    breakdown.append(('Taxable', len(shown)))
if limit_cusips:
    shown = shown.filter(
        (nw.col('num_cusips') >= min_cusips) &
        (nw.col('num_cusips') <= max_cusips)
    )
    breakdown.append(('Number of Coupons', len(shown)))
if state != 'All':
    shown = shown.filter(nw.col('state') == state)
    breakdown.append(('State', len(shown)))
if exclude_matured:
    shown = shown.filter(nw.col('maturity_date') > six_months_from_now)
    breakdown.append(('Already Matured', len(shown)))
else:
    shown = shown.filter(
        (nw.col('maturity_date') <= dt.combine(maturity_date_max, dt.min.time())) &
        (nw.col('maturity_date') >= dt.combine(maturity_date_min, dt.min.time()))
    )
    breakdown.append(('Maturity Date out of range', len(shown)))

# st.write(exclude_null)
for col in exclude_null:
    shown = shown.filter(~nw.col(col).is_null())
    breakdown.append((col.replace('_', ' ').title() + ' not provided', len(shown)))

total_coupons = len(shown)
if not show_coupons:
    shown = shown.unique('dealId')

l, r = st.columns([1, 2])
l.write(f'`{len(shown)}`/`{num_unfiltered}` {"schools" if not show_coupons else "coupons"} selected')
with r.expander("Breakdown"):
    prev = 0
    for name, cnt in breakdown:
        if cnt == prev:
            continue
        st.write(f"`{cnt - prev}` {name}")
        prev = cnt
    st.divider()
    if not show_coupons:
        st.write(f"= `{total_coupons}` total coupons")
    st.write(f"{'=' if show_coupons else '->'} `{len(shown)}` {'unique schools' if not show_coupons else 'coupons'}")

shown_df = shown.to_pandas()

# Display the data
st.dataframe(
    shown_df,
    column_config=column_config | {name: None for name in hide},
    column_order=column_config.keys(),
    hide_index=True,
)

# import plotly.express as px
# fig = px.histogram(data_frame=deals, x='num_cusips')
# st.plotly_chart(fig)
