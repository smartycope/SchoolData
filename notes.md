build in authentication

if the issuer is the same and corpProject is the same, and series has the same year, they're the same project, and they're duplicates



# Done
filter by state -- add a dropdown box
filter out first_call_date values which are more than a year away -- interactive.
group by schools, not by coupons -- add a checkbox?
add a box to sort by - drop dropdown
only show ones with single/mutiple coupons
filter by min and max par_amount
series is relevant
tic - the wighted (by par_amount) average of the intrest rates of the coupons
make "cusips" a list column
double check the par_amount is induvidual to each coupon, not each bond







State conduit issuer:
    access the tax exempt market

conduit: an entity that handles tax exempt transactions
    thing which issues conduit debt

exceptions: natinal issuers exist
    can issue debt in other states


use of proceeds: EDUC charter school

CUSIP: bond coupon id (pk, likely)

Goal:
    find all charter schools which have bonds which have coupon rates > 6%


Original yeild:
    if coupon == yeild,


Only look at the last 6 months of minutes


# authorizers
meet once a month
give permission to schools
try to find a list here: https://qualitycharters.org/

# TODO:
do, in fact, be able to scrape .doc files as well

https://www.dropbox.com/scl/fi/729ldwhy1cn232hq3lklt/Nationwide-Charter-School-List.xlsx?e=1&oref=e&r=ACcrax5yeGJ0waCzVig4UtOdH97m7tc8ZtFnN9rMB9EhbUEMnE7jUUED8-UoEti6K1UsQroCYWm7VaI-36oyAhdX3ZnYRu5pIAEInocwES7k8DPmqhZS52pIL4fj4YTH87h0Xg6wT_GruXNNvDY6sOTOdyeCTksYQ3786OLwWTOAqx7BUXD6KkiUJvkv-9nFLyN5yFG6Vdg8zMseoQoNN7fb&sm=1&dl=0


List of authorizer names as of 2016: https://qualitycharters.org/wp-content/uploads/2017/01/Index_of_Essential_Practices_2016_By_State_FINAL.pdf?pdf=Index_of_Essential_Practices_2016_By_State_FINAL.pdf

# Questions:
1. About the authorizers sheet, I'm looking at it (the one you sent me with authorizer, type, minutes & tips columns), and I'm not sure what "type" is? Is that describing what the minutes link points to?
2. Will the adgenda have the same information as minutes? Is an adgenda sufficient?
    - good substitute, try minutes first
3. You found the links on the authorizers sheet by just googling, right?
4. Am I looking at commission meetings, or commitee meetings? Both? What's the difference?
    - commission is a parent of committee
    - comittee is likely better
    - commission is going to be less specific, but still potentially useful
    -
5. I'm looking into (but am not set on) using a search engine API to find minutes pages. These cost money however. Am I okay to go with this route if it ends up useful, and you'll reimburse me? It should be <$50, and will most likely be a 1 time cost. There's also other options.
6. "Approved charter contracts". Is that relevant at all?
    - Yes. Scrape with separate keywords list
7. To start, I'm just finding authroizer minutes, right? not looking for charter school links and minutes?
8. Should I gather "modification application" documents as well? I can write a separate scraper for that easily
8.8. Do all authorizers have modification applications?
    - Maybe scrape.
9. WHat's a letter of intent? Do I need to gather those at all?
    - definite yes. Scrape these, separate keywords list
    - letter of intent comes first
10. Don't worry about scraping renewal applications

letter of intent
application
approval
charter renewal every 5 years
modification application

NEG: Non-Educational Government Entity
NFP: Nonprofit Organization
SEA: State Education Agency
HEI: Higher Education Institution
ICB: Independent Chartering Board
LEA: Local Education Agency


# Most recent notes
[x] Instead of TIC use value of coupon -- leave both, but have a dropdown box (either or, not both)
[x] change number of coupons max to 10
[x] add option to exclude coupons with maturity dates in the past (if any one coupon is not matured, still include the bond, but exclude from bonds)
[x] set the "show deals up until" box autofill with the current date
[x] add all float columns use precision (original price, origial yield , eval price, eval yield, spread, etc)
[x] change default TIC to 6.5?
[x] Allow "show deals up until" to go out to 2100
[x] make saleDate have it's own daterange specifically
    [x] Sale dates are often far in the future
    [x] first call date is still helpful
        [x] don't filter by it, but move the column closer to the front - very important
[x] move sale date closer to the front as well
[x] move state closer to the front (but not too close)

Find an estimate for using AI for the minutes scraper
build a scraper that finds specific urls from a minutes link



You can't refinance until the first call date -- maybe



# Next notes to ask him
* I did "exclude coupons with maturity dates within 6 months", would they like "exclude coupons with maturity dates in the past" instead?
* Deal Par Amount already has a range slider?
* is .1 a good step for coupon rate?
* are the columns in a good order?

